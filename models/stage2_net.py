import os
import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class Temporal_I3D_SGA_STD(nn.Module):
    def __init__(self,dropout_rate):
        super(Temporal_I3D_SGA_STD, self).__init__(k=1, soft_voting=True)
        self.k = k
        self.soft_voting = soft_voting
        self.Regressor=[nn.Sequential(nn.Linear(4096,512),nn.ReLU(), nn.Dropout(dropout_rate),nn.Linear(512, 32),nn.Dropout(dropout_rate),nn.Linear(32, 2)) for i in range(self.k)]
        self.Softmax=nn.Softmax(dim=-1)

    def train(self,mode=True):
        super(Temporal_I3D_SGA_STD, self).train(mode)
        return self

    def forward(self,x,act=True,extract=False):
        logits_all = []
        for i in range(self.k):
            logits=self.Regressor[i](x)
            if act:
                logits=self.Softmax(logits)
            logits_all.append(logits)
        if self.soft_voting:
            logits_all = torch.stack(logits_all, 1)  #### should be "dimension Batchsize, 2, k"
            logits_final = logits_all.mean(-1)
            pseudo_labels = torch.argmax(logits_final, 1)
        else:
            result_all = []
            zero_num = torch.zeros([x.size(0)]).cuda()
            one_num = torch.zeros([x.size(0)]).cuda()
            for i in range(self.k):
                result = torch.argmax(logits_all[i], 1)  ####, batchsize, 2
                zero_num += (result==0)   #### batchsize, 1
                one_num += (result==1)    #### batchsize, 1
            nums = torch.cat([zero_num, one_num], -1)    #### batchsize, 2
            pseudo_labels = torch.argmax(nums, 1)
        return pseudo_labels

