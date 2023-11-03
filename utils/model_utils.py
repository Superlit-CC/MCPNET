import numpy as np
from tqdm import tqdm, notebook
import copy
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score, balanced_accuracy_score, precision_score,\
recall_score, f1_score, average_precision_score, matthews_corrcoef, mean_absolute_error

import torch
import torch.nn as nn


def train(epochs, train_loader, valid_loader, model, optimizer, device, scheduler, step_size=5, cls=False):
    model = model.to(device)
    train_loss, valid_loss = [], []
    best_score = 0
    if cls:
        loss_fun = nn.BCELoss()
    else:
        loss_fun = nn.MSELoss()
    for epoch in notebook.tqdm(range(epochs)):
        epoch_loss = []
        # Train
        model.train()
        optimizer.zero_grad()
        for i, data in enumerate(train_loader):
            points, target = data
            points, target = points.squeeze(0).permute(0, 2, 1).to(device), target.to(device)
            pred, A, max_xyz = model(points)
            loss = loss_fun(pred.flatten(), target.flatten())
            # Progressive gradient descent
            (loss / step_size).backward()
            epoch_loss.append(loss.item())
            if ((i + 1) % step_size == 0):
                optimizer.step()
                optimizer.zero_grad()
        train_loss.append((sum(epoch_loss) / len(epoch_loss)))
        scheduler.step()
        # Evaluate
        model.eval()
        epoch_loss = []
        y_pred = []
        y_true = []
        with torch.no_grad():
            for data_v in valid_loader:
                points_v, target_v = data_v
                points_v, target_v = points_v.squeeze(0).permute(0, 2, 1).to(device), target_v.to(device)
                pred, A, max_xyz = model(points_v)
                loss = loss_fun(pred.flatten(), target_v.flatten())
                epoch_loss.append(loss.item())
                y_pred.extend(pred.flatten().tolist())
                y_true.extend(target_v.flatten().tolist())
            valid_loss.append((sum(epoch_loss) / len(epoch_loss)))
            if cls:
                accuracy = accuracy_score(y_true, [y > 0.5 for y in y_pred])
                auc = roc_auc_score(y_true, y_pred)
                balanced_accuracy = balanced_accuracy_score(y_true, [y > 0.5 for y in y_pred])
                if balanced_accuracy > best_score:
                    best_score = balanced_accuracy
                    torch.save({
                    'epoch': epoch,
                    'model': model,
                    }, 'out/models/attention_cls.pth')
            else:
                r2 = r2_score(y_true, y_pred)
                if r2 > best_score:
                    best_score = r2
                    torch.save({
                    'epoch': epoch,
                    'model': model,
                    }, 'out/models/attention_reg.pth')
        if (epoch + 1) % 5 == 0:
            if cls:
                print('[%d/%d] train loss %f valid loss %f accuracy %f auc %f balanced_accuracy %f' % (epoch + 1, epochs, train_loss[-1], valid_loss[-1], accuracy, auc, balanced_accuracy))
            else:
                print('[%d/%d] train loss %f valid loss %f r2 %f' % (epoch + 1, epochs, train_loss[-1], valid_loss[-1], r2))
    return train_loss, valid_loss

def mean_relative_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true))

def max_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.max(np.abs(y_pred - y_true))

def cal_score(model, loader, cls):
    """
    Calculate metrics
    """
    device = next(model.parameters()).device
    model.eval()
    y_scores = []
    y_true = []
    with torch.no_grad():
        for data_v in loader:
            points_v, target_v = data_v
            points_v, target_v = points_v.squeeze(0).permute(0, 2, 1).to(device), target_v.to(device)
            pred, A, max_xyz = model(points_v)
            y_scores.append(pred.item())
            y_true.append(target_v.item())
    # Classification
    if cls:
        y_pred = [y > 0.5 for y in y_scores]
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_scores)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        return accuracy, auc, balanced_accuracy, recall, mcc
    # Regression
    else:
        r2 = r2_score(y_true, y_scores)
        mse = mean_squared_error(y_true, y_scores)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_scores)
        me = max_error(y_true, y_scores)
        return r2, mse, rmse, mae, me

def get_important_points(model, points, scaler):
    """
    Get important feature points
    inputs:
        points (B, N, C)
    """
    device = next(model.parameters()).device
    tpoints = copy.deepcopy(points)
    for i, conf in enumerate(tpoints):
            tpoints[i] = scaler.transform(conf)
    tpoints = torch.from_numpy(tpoints).float()
    model.eval()
    with torch.no_grad():
        pred, A, max_xyz = model(tpoints.permute(0, 2, 1).to(device))
        conf = torch.max(A, 1)[1]
        return conf.item(), max_xyz.detach().cpu().numpy() # (1024, C)

def cal_true_scores(model, data_loader):
    device = next(model.parameters()).device
    model.eval()
    y_scores = []
    y_true = []
    with torch.no_grad():
        for data_v in data_loader:
            points_v, target_v = data_v
            points_v, target_v = points_v.squeeze(0).permute(0, 2, 1).to(device), target_v.to(device)
            pred, A, max_xyz = model(points_v)
            y_scores.append(pred.item())
            y_true.append(target_v.item())
    return y_true, y_scores