import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import os, sys, pdb, tqdm, random, json, gzip, bz2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import importlib
import copy
import argparse
from torchvision import transforms, datasets




class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)
    
class allcnn_t(nn.Module):
    def __init__(self, num_classes = 10, c1=96, c2=144):
        super().__init__()
        d = 0.5

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co, affine=True))

        self.m = nn.Sequential(
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            convbn(c2,num_classes,1,1),
            nn.AvgPool2d(8),
            View(num_classes))

        # print('Num parameters: ', sum([p.numel() for p in self.m.parameters()]))

    def forward(self, x):
        return self.m(x)





























# x = torch.randn([500, 3, 32, 32]).float()

# net = allcnn_t(2, 96, 192)

# print(net(x).shape)

