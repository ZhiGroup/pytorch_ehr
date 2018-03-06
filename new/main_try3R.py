# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:25:59 2018

@author: jzhu8
"""
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
import model # we have all models in this file
import Loaddata
import TrainVaTe

# check GPU availability
use_cuda = torch.cuda.is_available()



parser = argparse.ArgumentParser(description='30 Hospital Readmission Model with Pytorch: LR, RNN, CNN')
# learning
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.0001]')
parser.add_argument('-L2', type=float, default=0, help='L2 regularization [default: 0]')
parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 20]')
parser.add_argument('-batch_size', type=int, default=200, help='batch size for training [default: 200]')
#parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
#parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-seq_file', type = str, default = 'Data/h143.visits' , help='the path to the Pickled file containing visit information of patients')
parser.add_argument('-label_file', type = str, default = 'Data/h143.labels', help='the path to the Pickled file containing label information of patients')
parser.add_argument('-validation_ratio', type = float, default = 0.1, help='validation data size [default: 0.1]')
parser.add_argument('-test_ratio', type = float, default = 0.2, help='test data size [default: 0.2]')
# model
parser.add_argument('-which_model', type = str, default = 'RNN', help='choose from {"LR", "RNN", "CNN"}')
parser.add_argument('-mb', type = bool, default = True, help='whether train on mini batch (True) or not (False) [default: False ]') #at train
parser.add_argument('-input_size', type = int, default =20000, help='input dimension [default: 20000]')
parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
#parser.add_argument('-hidden_size', type=int, default=128, help='size of hidden layers [default: 128]')
parser.add_argument('-ch_out', type=int, default=64, help='number of each kind of kernel [default; 64]')
parser.add_argument('-kernel_sizes', type=list, default=[3], help='comma-separated kernel size to use for convolution [default:[3]')
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.1]')
parser.add_argument('-eb_mode', type=str, default='sum', help= "embedding mode [default: 'sum']")

# option
#parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
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
    
merged_set= [[set_y[i],model_x[i]] for i in range(len(set_y))] #list of list or list of lists of lists
print("\nLoading and preparing data...")    
train1, valid1, test1 = Loaddata.load_data(merged_set)
print("\nSample data after split:")  
print(train1[0])

# model loading part: choose which model to use 
if args.which_model == 'LR': 
    ehr_model = model.EHR_LR(args.input_size)
elif args.which_model == 'RNN':
    ehr_model = model.EHR_RNN(args.input_size, args.embed_dim, args.dropout, args.eb_mode) 
else: 
    ehr_model = model.EHR_CNN(args.input_size, args.embed_dim, args.dropout, args.eb_mode, args.ch_out, args.kernel_sizes)    
if use_cuda:
    ehr_model = ehr_model.cuda()

optimizer = optim.Adam(ehr_model.parameters(), lr=args.lr, weight_decay=args.L2)

## train validation and test part
#epochs=args.epochs
#batch_size=args.batch_size
#current_loss_allep=[]
#all_losses_allep=[]

# train, validation, and test for each epoch 
for ep in range(args.epochs):
    current_loss, train_loss = TrainVaTe.train(train1, model= ehr_model, optimizer = optimizer, batch_size = args.batch_size)#, mb=args.mb)
    print ('\n Current running on: Epoch ', ep,'Training loss:')
    print(train_loss)
    TrainVaTe.showPlot(train_loss)
    train_auc, y_real, y_hat = TrainVaTe.calculate_auc(model= ehr_model, data = train1, batch_size = args.batch_size)
    print ('\n Current running on: Epoch ', ep,' Training auc:', train_auc)
    TrainVaTe.auc_plot(y_real, y_hat)
    valid_auc, y_real, y_hat  = TrainVaTe.calculate_auc(model = ehr_model, data = valid1, batch_size = args.batch_size)
    print ('\n Current running on: Epoch ', ep,' validation auc:', valid_auc)
    TrainVaTe.auc_plot(y_real, y_hat)
    test_auc, y_real, y_hat = TrainVaTe.calculate_auc(model = ehr_model, data = test1, batch_size = args.batch_size)
    print ('\n Current running on: Epoch ', ep,' test auc:', test_auc)
    TrainVaTe.auc_plot(y_real, y_hat)
    #current_loss_allep.append(current_loss)
    #all_losses_allep.append(train_losses)
    
"""    
for all_losses_a in all_losses_l:
    showPlot(all_losses_a)    
##from torchviz import make_dot, make_dot_from_trace
output , label_tensor = model(train_sl[0:10])
make_dot (output)    

#useful model saving technique 

torch.save(the_model.state_dict(), PATH)
#Then later:

the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))



#or saving the whole model by doing:
The second saves and loads the entire model:

torch.save(the_model, PATH)
Then later:

the_model = torch.load(PATH)


"""



##Saving models after the best epoch?
