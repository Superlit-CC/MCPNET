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


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('run')
    parser.add_argument('--source_path', type=str, default=None,
                        help='the dataset path that needs to be predicted')
    parser.add_argument('--surface_path', type=str,
                        default='./data/surface/', help='the path of point cloud')
    parser.add_argument('--id_name', type=str, default='CAS',
                        help='column name of molecules id')
    parser.add_argument('--pred_name', type=str, default='Activity',
                        help='column name of predicted value')
    return parser.parse_args()


def main(args):
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    seed = 1
    n_splits = 5
    in_channels = 4
    npoints = 4096
    cls = True
    epochs = 5

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    fpr_list, tpr_list = [], []
    accuracy_list, auc_list, balanced_accuracy_list, recall_list, mcc_list = [], [], [], [], []
    r2_list, mse_list, rmse_list, mae_list, me_list = [], [], [], [], []
    train_loss_list, valid_loss_list = [], []

    KF = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # get raw data
    raw_X, raw_y, _ = data_utils.get_bag_raw(args.surface_path, args.source_path, npoints,
                                             args.id_name, args.pred_name, 'fps')
    raw_X, raw_y = np.array(raw_X), np.array(raw_y)

    for train_index, test_index in KF.split(raw_X):
        X_train, X_test = raw_X[train_index], raw_X[test_index]
        y_train, y_test = raw_y[train_index], raw_y[test_index]
        X_train, X_test, y_train, y_test = copy.deepcopy(X_train), copy.deepcopy(
            X_test), copy.deepcopy(y_train), copy.deepcopy(y_test)
        X_train, X_test, _ = data_utils.scaler_data(X_train, X_test)
        train_loader, valid_loader = data_utils.BagsLoader(
            X_train, X_test, y_train, y_test)
        
        # train
        model = Attention(in_channels=in_channels, norm='layer', N=npoints,
                          res_layer=True, layer_list=[64, 128, 256, 1024], cls=cls)
        optimizer = optim.Adam(model.parameters(), lr=1e-5,
                               betas=(0.9, 0.999), weight_decay=1e-2)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5)
        train_loss, valid_loss = model_utils.train(
            epochs, train_loader, valid_loader, model, optimizer, device, scheduler, step_size=4, cls=cls)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        # eval
        if cls:
            checkpoint = torch.load('out/models/attention_cls.pth')
        else:
            checkpoint = torch.load('out/models/attention_reg.pth')
        best_model = checkpoint['model']
        if cls:
            accuracy, auc, balanced_accuracy, recall, mcc = model_utils.cal_score(
                best_model, valid_loader, cls)
            y_true, y_scores = model_utils.cal_true_scores(best_model, valid_loader)
            fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

            accuracy_list.append(accuracy)
            auc_list.append(auc)
            balanced_accuracy_list.append(balanced_accuracy)
            recall_list.append(recall)
            mcc_list.append(mcc)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        else:
            r2, mse, rmse, mae, me = model_utils.cal_score(
                best_model, valid_loader, cls)

            r2_list.append(r2)
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            me_list.append(me)
    if cls:
        print(f'accuracy {np.mean(accuracy_list):.4f} auc {np.mean(auc_list):.4f} balanced_accuracy {np.mean(balanced_accuracy_list):.4f} mcc {np.mean(mcc_list):.4f} recall {np.mean(recall_list):.4f}')
    else:
        print(f'r2 {np.mean(r2_list):.4f} mse {np.mean(mse_list):.4f} rmse {np.mean(rmse_list)} mae {np.mean(mae_list)} me {np.mean(me_list)}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
