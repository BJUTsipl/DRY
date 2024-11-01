#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from cmath import log
import torch
import torch.nn as nn

from collections import OrderedDict

class GDIFD_CGM(nn.Module):
    """
    Gated Domain-Invariant Feature Disentanglement 
    """
    def __init__(
        self, 
        num_convs=3, 
        in_channels=256,
        s_init=0.9
    ):
        super().__init__()

        CGM_tower = []
        for i in range(num_convs):
            CGM_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            CGM_tower.append(nn.InstanceNorm2d(in_channels))
            CGM_tower.append(nn.ReLU())

        self.add_module('CGM_tower', nn.Sequential(*CGM_tower))

        # self.special_init_layer = nn.Sequential(OrderedDict([
        #   ('si_conv', nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1)),
        #   ('si_act', nn.Sigmoid())
        # ]))
        special_init_layer = []
        special_init_layer.append(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        special_init_layer.append(nn.Sigmoid())
        self.add_module('special_init_layer', nn.Sequential(*special_init_layer))

        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        # initialization
        for modules in self.CGM_tower:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.b_init = -log((1-s_init)/s_init)

        print("=====================")
        print("GDIFD_CGM s_init = {}".format(s_init))
        print("GDIFD_CGM b_init = {}".format(self.b_init))
        print("=====================")

        for modules in self.special_init_layer:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, self.b_init)

    def forward(self, feature):
        b, c, _, _ = feature.size()
        x = self.CGM_tower(feature)
        x = self.special_init_layer(x)
        s = self.GAP(x).view(b, c, 1, 1)
        return s

