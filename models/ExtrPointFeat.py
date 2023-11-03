import numpy as np
import pandas as pd
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable


class ExtrPointFeat(nn.Module):
    def __init__(self, in_channels:int=4, norm='layer', N=4096, res_layer=False, layer_list=[64, 128, 1024]):
        super(ExtrPointFeat, self).__init__()
        self.res_layer = res_layer
        # mlp
        self.layer_list = layer_list
        self.conv = nn.ModuleList([nn.Conv1d(in_channels, self.layer_list[0], 1)])
        layer_sizes = zip(self.layer_list[:-1], self.layer_list[1:])
        self.conv.extend([nn.Conv1d(h1, h2, 1) for h1, h2 in layer_sizes])
        # norm
        if norm == 'batch':
            self.norm = nn.ModuleList(nn.BatchNorm1d(i) for i in self.layer_list)
        elif norm == 'layer':
            self.norm = nn.ModuleList(nn.LayerNorm([i, N]) for i in self.layer_list)
        else:
            print('No such norm!')
            sys.exit('Done')
        if self.res_layer:
            self.constant_layers = nn.ModuleList([nn.Conv1d(h, h, 1) for h in self.layer_list[:-1]])
    
    def forward(self, x):
        """
        x (B, k, N)
        """
        x = F.relu(self.norm[0](self.conv[0](x))) # (B, 64, N)
        for i, conv in enumerate(self.conv):
            if i == 0:
                continue
            x = conv(x)
            if self.res_layer and i != len(self.conv) - 1:
                x = x + self.constant_layers[i](x)
            x = F.relu(self.norm[i](x))
        
        temp = x
        x = torch.max(x, 2)[0] # (B, 1024)
        max_xyz = torch.max(temp, 2)[1]
        return x, max_xyz