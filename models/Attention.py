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


class Attention(nn.Module):
    def __init__(self, in_channels=4, norm='layer', N=4096, res_layer=True, layer_list=[64, 128, 1024], cls=True) -> None:
        super(Attention, self).__init__()

        self.L = 1024
        self.D = 256
        self.K = 1
        self.cls = cls

        self.feature_extraction = ExtrPointFeat(in_channels=in_channels, norm=norm, N=N, res_layer=res_layer, layer_list=layer_list)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.sig = nn.Sigmoid()

        self.down = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(self.L*self.K, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        inpurt:
            x (B, k, N)
        """
        H, max_xyz = self.feature_extraction(x) # (B, L)

        A = self.attention(H) # (B, K)
        A = torch.transpose(A, 1, 0) # (K, B)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, H) # (K, L)

        Y_prob = self.down(M)

        if self.cls:
            Y_prob = self.sig(Y_prob)

        return Y_prob, A, max_xyz

class ScaledDotProductAttention(nn.Module):
    def __init__(self, in_channels=3, norm='layer', N=4096, res_layer=False, layer_list=[64, 128, 1024], cls=False) -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.L = 1024
        self.M = 1024
        self.cls = cls

        self.feature_extraction = ExtrPointFeat(in_channels=in_channels, norm=norm, N=N, res_layer=res_layer, layer_list=layer_list)

        self.lin1 = nn.Linear(self.L, self.M)
        self.lin2 = nn.Linear(self.L, self.M)
        self.lin3 = nn.Linear(self.L, self.M)

        self.sig = nn.Sigmoid()

        self.reg = nn.Sequential(
            nn.Linear(self.M, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        inpurt:
            x (B, k, N)
        """
        H, max_xyz = self.feature_extraction(x) # (B, L)
        Q, K, V = self.lin1(H), self.lin2(H), self.lin3(H) # (B, M)
        d = torch.tensor(Q.shape[-1])
        X = F.softmax(torch.mm(Q, K.T) / torch.sqrt(d), dim=-1) # (B, B)
        A = torch.mm(X, V) # (B, M)

        Y_prob = torch.mean(self.reg(A))

        if self.cls:
            Y_prob = self.sig(Y_prob)

        return Y_prob, A, max_xyz