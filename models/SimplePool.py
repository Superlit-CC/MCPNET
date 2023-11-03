import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
from tqdm import tqdm, notebook
import pickle

from models.ExtrPointFeat import ExtrPointFeat


class SimplePool(nn.Module):
    def __init__(self, pool='max', in_channels=3, norm='layer', N=4096, res_layer=False, layer_list=[64, 128, 1024], cls=False, seed=123) -> None:
        super(SimplePool, self).__init__()
        
        self.feature_extraction = ExtrPointFeat(in_channels=in_channels, norm=norm, N=N, res_layer=res_layer, layer_list=layer_list)

        self.pool = pool
        self.cls = cls

        self.reg = nn.Sequential(
            nn.Linear(1024, 256), nn.Dropout(p=0.3), nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.dropout = nn.Dropout(p=0.3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        H, max_xyz = self.feature_extraction(x) # (B, L)
        A = None

        if self.pool == 'max':
            H, A = torch.max(H, 0)
        elif self.pool == 'mean':
            H = torch.mean(H, 0)
        
        y_pred = self.reg(H)
        if self.cls:
            y_pred = self.sig(y_pred)
        return y_pred, A, max_xyz