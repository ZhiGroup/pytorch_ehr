# -*- coding: utf-8 -*-
"""
Created on Mon Apr  30 11:50:18 2018

@author: jzhu8
"""

#main execution file 
#stops after 12 stagnant epoches
#only calculates test AUC when it is the best validation AUC

from __future__ import print_function, division
from io import open
import string
import re
import random

import os
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
#from torchviz import make_dot, make_dot_from_trace

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
    
#import self-defined modules
import model_final as model 
import Loaddata
import TrainVaTe_Final as TVT

# check GPU availability
use_cuda = torch.cuda.is_available()


parser = argparse.ArgumentParser(description='Predictive Analytics on EHR using Pytorch: LR, RNN')
# learning
parser.add_argument('-lr', type=float, default=np.exp(-9.0790), help='initial learning rate [default: 0.0001]')
parser.add_argument('-L2', type=float, default=np.exp(-7.3835) help='L2 regularization [default: 0.0006]')
parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 30]')
parser.add_argument('-batch_size', type=int, default=1, help='batch size for training [default: 1]')
parser.add_argument('-seq_file', type = str, default = 'Data/h143.visits' , help='the path to the Pickled file containing visit information of patients')
parser.add_argument('-label_file', type = str, default = 'Data/h143.labels', help='the path to the Pickled file containing label information of patients')
# model
parser.add_argument('-which_model', type = str, default = 'LR', help='choose from {"LR", "RNN"}')
parser.add_argument('-input_size', type = int, default =20000, help='input dimension [default: 20000]')
parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-hidden_size', type=int, default=128, help='size of hidden layers [default: 128]')
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.1]')
parser.add_argument('-eb_mode', type=str, default='sum', help= "embedding mode [default: 'sum']")
args = parser.parse_args()


# load and prepare data
set_x = pickle.load(open(args.seq_file, 'rb'), encoding='bytes')
set_y = pickle.load(open(args.label_file, 'rb'),encoding='bytes')

#preprocessing
# LR needs to have input format of list; list of list for NN models
if args.which_model == 'LR':
    model_x = []
    for patient in set_x:
        model_x.append([each for visit in patient for each in visit])  
else: 
    model_x = set_x     
    
merged_set= [[set_y[i],model_x[i]] for i in range(len(set_y))] 
print("\nLoading and preparing data...")    
train1, valid1, test1 = Loaddata.load_data(merged_set)
print("\nSample data after split:")  
print(train1[0])

# model loading part: choose which model to use 
if args.which_model == 'LR': 
    ehr_model = model.EHR_LR(args.input_size)
else:
    ehr_model = model.EHR_RNN(args.input_size, args.embed_dim, args.dropout, args.eb_mode)  
if use_cuda:
    ehr_model = ehr_model.cuda()

optimizer = optim.Adam(ehr_model.parameters(), lr=args.lr, weight_decay=args.L2)

#small function to track the models 
def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()

logFile='models/EHR_LR.log'
header = 'Model|EmbSize|LR|L2|BestValidAUC|TestAUC|atEpoch'
print2file(header, logFile)

## train validation and test part
"""current_loss_allep=[]
all_losses_allep=[]
avg_losses_allep=[]
train_auc_allep =[]
valid_auc_allep =[]
test_auc_allep=[]"""

bestValidAuc = 0.0
bestTestAuc = 0.0
bestValidEpoch = 0
  

# train, validation, and test for each epoch 
for ep in range(args.epochs):
    current_loss, train_loss, _ = TVT.train(train1, model= ehr_model, optimizer = optimizer, batch_size = args.batch_size) 
    avg_loss = np.mean(train_loss)
    print ('\n Current running on: Epoch ', ep,'Training loss:',' Average loss', avg_loss)
    TVT.showPlot(train_loss)
    valid_auc, y_real, y_hat, _  = TVT.calculate_auc(model = ehr_model, data = valid1, which_model = args.which_model, batch_size = args.batch_size)
    if valid_auc > bestValidAuc: 
          bestValidAuc = valid_auc
          bestValidEpoch = ep
          best_model= ehr_model

    if ep - bestValidEpoch >12:
          break

    bmodel_pth='models/'
    bestTestAuc, y_real, y_hat,_ = TVT.calculate_auc(model = best_model, data = test1, which_model = w_model, batch_size = batch_size)
    torch.save(best_model, bmodel_pth) 
    buf = '|%f |%f |%d ' % (bestValidAuc, bestTestAuc, bestValidEpoch )
    pFile= w_model+'|'+str(embed_dim)+'|'+str(lr)+'|'+str(l2)+'|'+ buf
    print2file(pFile, logFile)    
    
    """
    #compared to what we used to do:
    train_auc, y_real, y_hat, _ = TVT.calculate_auc(model= ehr_model, data = train1, which_model = args.which_model, batch_size = args.batch_size)
    TrainVaTe.auc_plot(y_real, y_hat)
    valid_auc, y_real, y_hat, _  = TVT.calculate_auc(model = ehr_model, data = valid1, which_model = args.which_model, batch_size = args.batch_size)
    #print ('\n Current running on: Epoch ', ep,' validation auc:', valid_auc)
    TrainVaTe.auc_plot(y_real, y_hat)
    test_auc, y_real, y_hat, _ = TrainVaTe.calculate_auc(model = ehr_model, data = test1, which_model = args.which_model, batch_size = args.batch_size)
    #print ('\n Current running on: Epoch ', ep,' test auc:', test_auc)
    TrainVaTe.auc_plot(y_real, y_hat)
    current_loss_allep.append(current_loss)
    all_losses_allep.append(train_loss)
    avg_losses_allep.append(avg_loss)
    train_auc_allep.append(train_auc)
    valid_auc_allep.append(valid_auc)
    test_auc_allep.append(test_auc)"""
