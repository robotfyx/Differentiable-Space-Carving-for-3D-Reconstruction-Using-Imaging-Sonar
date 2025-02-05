# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:10:04 2023
akutam_fyx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 input_ch=3
                 ):
        super(network, self).__init__()
        self.input_ch = input_ch
        # prob network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        prod_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 # 1 prod
            else:
                out_dim = hidden_dim
            
            prod_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.prod_net = nn.ModuleList(prod_net)
    
    def forward(self, x):
        h = x
        for l in range(self.num_layers):
            h = self.prod_net[l](h)
            if l != self.num_layers-1:
                h = F.relu(h, inplace=True)
        h = F.sigmoid(h)
        return h.clone()

if __name__ == '__main__':
    x = torch.ones((10,3))
    y = network()(x)
    print(y)
    