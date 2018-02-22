# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:40:32 2018

@author: jzhu8
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

#Define Model: simple Logistic Regression
class EHR_LR(nn.Module):
    #def __init__(self, input_size, hidden_size, n_layers=1):#hidden_size = out_size in Logistic Regression 
    def __init__(self,input_size, hidden_size = 2, n_layers =1):    
        super(EHR_LR, self).__init__()
        
        #input_size = args.input_size

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.out = nn.Linear(self.hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input,hidden):
        embedded = self.embedding(input).view(-1, self.hidden_size) #modified, instead of (-1,1, self.hidden_size) => use (-1,self.hidden_size
        embedded = torch.sum(embedded, dim=0).view(1,-1) #modified,instead of (1,1,-1) => use .view(1,-1)
        #output = embedded
        #for i in range(self.n_layers):
            #output, hidden = self.gru(output, hidden)
            
        output = self.sigmoid(self.out(embedded))
        hidden = self.sigmoid(self.out(embedded)) #modified,hidden == out for Logistic Regression 
        return output, hidden 

    def initHidden(self):
        result = Variable(torch.zeros(1,self.hidden_size)) #modified, instead of (1,1,self.hidden_size) => use (1,self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result