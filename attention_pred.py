from utils import model_utils, data_utils, plot_utils
from models.Attention import Attention

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset

import rdkit
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score, balanced_accuracy_score, precision_score,\
    recall_score, f1_score, average_precision_score, matthews_corrcoef, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import numpy as np
import pickle
import pandas as pd
import random
import os
import sys
import argparse
import copy
from tqdm import notebook, tqdm


############# PARAMETERS #################
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('run')
    parser.add_argument('--use_cpu', action='store_true',
                        default=False, help='use cpu mode')
    parser.add_argument('--source_path', type=str, default=None,
                        help='the dataset path that needs to be predicted')
    parser.add_argument('--confs_path', type=str,
                        default='./data/confs/', help='the path of conformations')
    parser.add_argument('--surface_path', type=str,
                        default='./data/surface/', help='the path of point cloud')
    parser.add_argument('--smiles_name', type=str, default='SMILES',
                        help='column name of smiles in the data')
    parser.add_argument('--id_name', type=str, default='CAS',
                        help='column name of molecules id')
    parser.add_argument('--num_cpu', type=int, default=4,
                        help='the number of CPUs used')
    parser.add_argument('--raw_data_path', type=str,
                        default='./data/raw_20_cls.pkl', help='the path of raw data')
    parser.add_argument('--num_point', type=int, default=4096,
                        help='the number of points sampled')
    parser.add_argument('--pred_name', type=str, default='Activity',
                        help='column name of predicted value')
    parser.add_argument('--sample_method', type=str,
                        default='fps', help='sample method [fps, random]')
    parser.add_argument('--cls', type=bool, default=True,
                        help='classification')
    parser.add_argument('--xyz', type=bool, default=False,
                        help='only (x, y, z) dimensions are used')
    parser.add_argument('--seed', default=3, type=int,
                        help='seed')
    parser.add_argument('--epoch', default=200, type=int,
                        help='number of epoch in training')
    parser.add_argument('--step_size', default=4, type=int,
                        help='number of gradient accumulations')
    parser.add_argument('--learning_rate', default=1e-5,
                        type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float,
                        default=1e-2, help='decay rate')
    parser.add_argument('--model', default='attention', 
                        help='model name [attention, max, mean]')
    return parser.parse_args()


############ DataLoader #############
def get_dataloader(raw_X, raw_y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        raw_X, raw_y, test_size=test_size)
    X_train, X_test = data_utils.scaler_data(X_train, X_test)
    train_loader, valid_loader = data_utils.BagsLoader(
        X_train, X_test, y_train, y_test)
    return train_loader, valid_loader


def main(args):
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ############ Conformation generation ############
    source = pd.read_csv(args.source_path)
    smiles = source[args.smiles_name]
    ID = source[args.id_name]
    data_utils.conf_gen(args.confs_path, smiles, ID)

    ############# Generate van der Waals surface points ###############
    data_utils.multi_pro_gen_surface(args.confs_path, args.surface_path, args.num_cpu)

    ############# Surface point sampling ###############
    raw_X, raw_y, _ = data_utils.get_bag_raw(args.surface_path, args.source_path, args.num_point, \
                                                     args.id_name, args.pred_name, args.sample_method)

    if args.cls is False:
        act_index = (raw_y < 6.0)
        raw_y = raw_y[act_index]
        raw_X = np.array(raw_X)[act_index]
    else:
        raw_X = np.array(raw_X)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    ############ DataLoader #############
    X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_y, test_size=0.3, random_state=seed)
    X_train, X_test, y_train, y_test = copy.deepcopy(X_train), copy.deepcopy(X_test), copy.deepcopy(y_train), copy.deepcopy(y_test)
    X_train, X_test, _ = data_utils.scaler_data(X_train, X_test)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.3333, random_state=seed)
    train_loader, valid_loader = data_utils.BagsLoader(X_train, X_valid, y_train, y_valid)
    _, test_loader = data_utils.BagsLoader(X_valid, X_test, y_valid, y_test)

    ############## train ################
    model = Attention(in_channels=4, norm='layer', N=args.num_point, res_layer=True, \
                                        layer_list=[64, 128, 256, 1024], cls=args.cls)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, \
                           betas=(0.9, 0.999), weight_decay=args.decay_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    train_loss, valid_loss = model_utils.train(args.epoch, train_loader, valid_loader, model, optimizer, \
                                               device, scheduler, step_size=args.step_size, cls=args.cls)

    ############## eval ################
    plot_utils.plot_loss(train_loss, valid_loss)
    if args.cls:
        checkpoint = torch.load('out/models/attention_cls.pth')
    else:
        checkpoint = torch.load('out/models/attention_reg.pth')
    best_model = checkpoint['model']
    if args.cls:
        accuracy, auc, balanced_accuracy, recall, mcc = model_utils.cal_score( \
            best_model, test_loader, args.cls)
        print(f'accuracy: {accuracy:.4f} auc: {auc:.4f} balanced_accuracy: {balanced_accuracy:.4f} \
              recall: {recall:.4f} mcc: {mcc:.4f}' +
              ' epoch: ' + str(checkpoint['epoch']))
    else:
        r2, mse, rmse, mae, me = model_utils.cal_score(best_model, test_loader, args.cls)
        print(f'r2: {r2:.4f} mse: {mse:.4f} mae: {mae:.4f} rmse: {rmse:.4f} me: {me:.4f} epoch: ' +
              str(checkpoint['epoch']))


if __name__ == '__main__':
    args = parse_args()
    main(args)
