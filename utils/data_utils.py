import pandas as pd
import os
from tqdm import tqdm, notebook
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
import sys
import random

from sklearn.preprocessing import StandardScaler

import rdkit
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset


###################### Multi-conformational generation and charge calculation #######################

def calc_geom_center(xyz):
    """  
    Calculate the geometric center of a molecule
    output: [x, y, z]
    """
    return np.average(xyz, axis=0)


def calc_inertia_tensor(xyz, mass=1):
    """  
    Calculate the moment of inertia tensor of a molecule
    input:
        xyz The atomic coordinates of the molecule
        mass The corresponding atomic mass is 1 by default
    """
    I_xx = (mass * np.sum(np.square(xyz[:,1:3:1]),axis=1)).sum()
    I_yy = (mass * np.sum(np.square(xyz[:,0:3:2]),axis=1)).sum()
    I_zz = (mass * np.sum(np.square(xyz[:,0:2:1]),axis=1)).sum()
    I_xy = (-1 * mass * np.prod(xyz[:,0:2:1],axis=1)).sum()
    I_yz = (-1 * mass * np.prod(xyz[:,1:3:1],axis=1)).sum()
    I_xz = (-1 * mass * np.prod(xyz[:,0:3:2],axis=1)).sum()
    I = np.array([[I_xx, I_xy, I_xz],
                  [I_xy, I_yy, I_yz],
                  [I_xz, I_yz, I_zz]])
    return I


def mol_align(xyz):
    """
    input:
        xyz The molecular coordinates that need to be transformed
    """
    geom_center = calc_geom_center(xyz)
    new_xyz = xyz - geom_center
    I = calc_inertia_tensor(new_xyz)
    eigenvalue, featurevector = np.linalg.eigh(I)
    return np.dot(new_xyz, featurevector)


def conf_gen(output_path, smiles, CHEMBL_ID, num_threads=6, num_confs=20, maxiter=1000, rms_thresh=0.5):
    for i, s in enumerate(tqdm(smiles, ncols=100)):
        m = Chem.MolFromSmiles(s)
        m3d = Chem.AddHs(m)
        num_iters = 0
        while m3d.GetNumConformers() < num_confs:
             # Limit the maximum number of iterations
            if num_iters >= maxiter:
                print(f'Generate Conformers\'s iterations over {maxiter}!')
                break
            num_iters += 1
            
            conf = Chem.AddHs(m)
            AllChem.EmbedMolecule(conf)
            if conf.GetNumConformers() == 0:
                print(f'{s} is not a nice molecule...')
                break
            AllChem.MMFFOptimizeMoleculeConfs(conf, numThreads=num_threads)
            # Conformational deduplication after optimization
            if m3d.GetNumConformers() == 0:
                m3d.AddConformer(conf.GetConformer(), assignId=True)
            else:
                cid = m3d.AddConformer(conf.GetConformer(), assignId=True)
                for j in range(cid):
                    if AllChem.GetConformerRMS(m3d, j, cid) < rms_thresh:
                        m3d.RemoveConformer(cid)
                        break
        # Make the Cartesian coordinates parallel to the inertial spindle
        if m3d.GetNumConformers() == 0:
            continue
        conf = m3d.GetConformer(id=0)
        xyz = conf.GetPositions()
        xyz = mol_align(xyz)
        for j in range(m3d.GetNumAtoms()):
            p3d = Geometry.Point3D(xyz[j, 0], xyz[j, 1], xyz[j, 2])
            conf.SetAtomPosition(j, p3d)
        
        # Inter-conformation alignment
        Chem.rdMolAlign.AlignMolConformers(m3d)
        cm = Chem.AddHs(m)

        save_path = output_path + CHEMBL_ID[i] + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for j, conf in enumerate(m3d.GetConformers()):
            cm.AddConformer(conf)
            charges = rdkit.Chem.rdMolDescriptors.CalcEEMcharges(cm)
            xyz = cm.GetConformer().GetPositions()
            symbol = [a.GetSymbol() for a in cm.GetAtoms()]
            res = np.c_[symbol, xyz, np.array(charges)]
            mol = pd.DataFrame(res, columns=['atom', 'x', 'y', 'z', 'charge'])
            mol.to_csv(save_path + 'cid_' + str(j) + '.csv', index=False)
            rdkit.Chem.rdchem.Mol.RemoveAllConformers(cm)

################ Get the molecular van der Waals surface #########################

class get_data():
    def __init__(self) -> None:
        self.atom_radius = {'H':1.2, 'Li':1.82, 'Na':2.27, 'K':2.75, 'Mg':1.73, 'B':2.13, 'Al':2.51, 'C':1.7, 'Si':2.1, 'Sn':2.27, 'N':1.55, 'P':1.8, 'O':1.52, 'S':1.8, 'F':1.47, 'Cl':1.75, 'Br':1.85, 'I':1.98}
        
    def get_points(self, x=0, y=0, z=0, r=1, l:int=10, d:int=20) -> list:
        """
        Gets the points of the circular surface
        """
        res = []
        l = np.linspace(start=0, stop=np.pi, num=l + 2)
        d = np.linspace(start=0, stop=np.pi * 2, num=d + 1)
        for i in l[1:-1]:
            for j in d[1:]:
                res.append([x + r * np.sin(i) * np.cos(j), y + r * np.sin(i) * np.sin(j), z + r * np.cos(i)])
        return res

    def drop_dump(self, points, x, y, z, radius, charges=None):
        """
        Remove overlapping parts of the circle
        """
        res = []
        for p in points:
            flag = False
            cha = 0
            for i in range(len(radius)):
                distance = (x[i] - p[0])**2 + (y[i] - p[1])**2 + (z[i] - p[2])**2
                if distance + 1e-9 < radius[i]**2:
                    flag = True
                    break
                if charges is not None:
                    cha += charges[i] / (distance**2)
            if flag is False:
                if charges is not None:
                    p.append(cha)
                res.append(p)
        return res

def gen_surface(pro_id, input_path, output_path, filenames, f):
    """
    Get the molecular van der Waals surface point
    """
    for filename in tqdm(filenames, ncols=80, desc='Perform Tasks' + str(pro_id) + ' pid:' + str(os.getpid())):
        mol_path = input_path + filename + '/'
        save_path = output_path + filename + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        flag = True
        for conf in os.listdir(mol_path):
            mol = pd.read_csv(mol_path + conf)
            x = mol['x'].values
            y = mol['y'].values
            z = mol['z'].values
            charge = mol['charge'].values
            atom = mol['atom'].values
            radius = []
            for i in atom:
                if i in f.atom_radius:
                    radius.append(f.atom_radius[i])
                else:
                    flag = False
                    break

            if flag is False:
                print(filename + ' has abnormal atoms !')
                break

            if charge[0] > 10 or charge[0] < -10:
                print(filename + '\'s charge is not normal !')
                break

            points = []
            for i in range(len(x)):
                r = radius[i]
                l = int(r / 1.52 * 20)
                points.append(f.get_points(x[i], y[i], z[i], r, l, 2 * l))
            points_drop = []
            for p in points:
                points_drop.append(f.drop_dump(p, x, y, z, radius, charge))
            res = []
            for p in points_drop:
                res.extend(p)
            df = pd.DataFrame(res, columns=['x', 'y', 'z', 'ele'])
            df.to_csv(save_path + conf, index=False)


def multi_pro_gen_surface(input_path, output_path, cpu_num):
    """
    Multi-process acquisition of molecular van der Waals surface points
    """
    filenames = os.listdir(input_path)
    f = get_data()

    p = mp.Pool(cpu_num)
    cnt = len(filenames) // cpu_num

    if len(filenames) < cpu_num:
        print('Too many cpus!!!')
        sys.exit()

    for i in range(cpu_num):
        if i == cpu_num - 1:
            p.apply_async(gen_surface, args=(i, input_path, output_path, filenames[i * cnt:], f))
        else:
            p.apply_async(gen_surface, args=(i, input_path, output_path, filenames[i * cnt:(i + 1) * cnt], f))
    p.close()
    p.join()
    print('done')

######################## Surface point sampling #############################

def farthest_point_sample(xyz, npoint):
    """
    The farthest point sampling (FPS)
    input:
        npoint: Number of sampling points
    output:
        centroids: Sample point subscript [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Returns the point cloud data indexed by the input point cloud data and index
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # repeat 复制tensor
    new_points = points[batch_indices, idx, :]
    return new_points


def get_bag_raw(input_path, source_file, npoints, id_name, target_name, method='fps', seed=123, sample_num=-1):
    """
    Get the bag raw data
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # 随机采样
    random.seed(seed)
    if sample_num == -1:
        sample_list = os.listdir(input_path)
    else:
        sample_list = random.sample(os.listdir(input_path), sample_num)
    npoints = npoints

    source = pd.read_csv(source_file)

    raw_X = []
    raw_y = []
    mol_names = []

    for mol in tqdm(sample_list, ncols=100):
        points = []
        for conf in os.listdir(input_path + mol):
            t = pd.read_csv(input_path + mol + '/' + conf).values
            if ((t[:, 3] < -1e10).sum() or np.isnan(t).sum()):
                print(f'{mol}_{conf} got a outlier or a nan!')
                continue
            
            if method == 'random':
                points_list = [j for j in range(len(t))]
                points_list = np.random.choice(points_list, size=npoints, replace=True)
                t = t[points_list, :]
            elif method == 'fps':
                t = torch.from_numpy(t).float().to(device).unsqueeze(0)
                idx = farthest_point_sample(t[:, :, :3], npoints) # (1, npoints)
                t = index_points(t, idx).squeeze(0) # (npoints, C)
            
            points.append(t.tolist())
        if len(points) != 0:
            raw_y.append(source[source[id_name] == mol][target_name].item())
            raw_X.append(np.array(points))
            mol_names.append(mol)
    
    return raw_X, np.array(raw_y), mol_names


################ Data processing and packaging ###########################

def scaler_data(X_train, X_test):
    """
    standardization
    """
    scaler = StandardScaler()
    X_trans = np.concatenate([x[0] for x in X_train])
    scaler.fit(X_trans)

    for i, mol in enumerate(X_train):
        for j, conf in enumerate(mol):
            X_train[i][j] = scaler.transform(conf)

    for i, mol in enumerate(X_test):
        for j, conf in enumerate(mol):
            X_test[i][j] = scaler.transform(conf)
    return X_train, X_test, scaler


class BagsDataset(Dataset):
    """
    The Dataset: [Bag1, Bag2, ..., BagN]
    """
    def __init__(self, bags, targets):
        # 转化为tensor
        self.bags = [torch.from_numpy(d).float() for d in bags]
        self.targets = torch.from_numpy(targets).float()
        
    def __getitem__(self, index):
        return self.bags[index], self.targets[index]
    
    def __len__(self):
        return len(self.bags)


def BagsLoader(X_train, X_test, y_train, y_test, sampler=None):
    """
    DataLoader
    """
    if sampler is not None:
        train_loader = DataLoader(BagsDataset(X_train, y_train), batch_size=1, num_workers=4, sampler=sampler, drop_last=True)
    else:
        train_loader = DataLoader(BagsDataset(X_train, y_train), batch_size=1, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(BagsDataset(X_test, y_test), batch_size=1, shuffle=False, num_workers=4)

    return train_loader, valid_loader


