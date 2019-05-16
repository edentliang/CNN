#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:21:42 2019

@author: haoxingliang
"""

import torch.nn as nn

cfgs = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, version, num_classes = 10):
        super(VGG, self).__init__()
        self.feature = self._make_layers(cfgs[version])
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
                nn.Linear(7 * 7 * 512, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(4096, num_classes))
    
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1) #batch dimension
        x = self.classifier(x)
        
        return x
        
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            else :
                layers += [nn.Conv2d(in_channels, l, kernel_size = 3, padding = 1), 
                           nn.BatchNorm2d(l),
                           nn.ReLU(inplace = True)]
                in_channels = l
        return nn.Sequential(*layers)
                