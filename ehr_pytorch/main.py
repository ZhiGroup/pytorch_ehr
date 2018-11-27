# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:57:40 2018

@author: ginnyzhu
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
from torch.utils.data import Dataset, DataLoader

#from sklearn.metrics import roc_auc_score  
#from sklearn.metrics import roc_curve 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
    
#import self-defined modules
#models, utils, and Dataloader
import models as model 
#import EHRDataloader as dataloader
from EHRDataloader import EHRdataFromPickles, EHRdataloader  #do modifications later
import utils as ut #:)))) 
#from embedding import EHRembeddings
from EHREmb import EHREmbeddings

#silly ones
from termcolor import colored

# check GPU availability
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

def main():
    #this is where you define all the things you wanna run in your main file
    parser = argparse.ArgumentParser(description='Predictive Analytics on EHR with Pytorch')
    
    #EHRdataloader 
    parser.add_argument('-root_dir', type = str, default = '../data/' , help='the path to the folders with pickled file(s)')
    parser.add_argument('-file', type = str, default = 'hf.train' , help='the name of pickled files')
    parser.add_argument('-test_ratio', type = float, default = 0.2, help='test data size [default: 0.2]')
    parser.add_argument('-valid_ratio', type = float, default = 0.1, help='validation data size [default: 0.1]')
    
    #EHRmodel
    parser.add_argument('-which_model', type = str, default = 'DRNN', help='choose from {"RNN","DRNN","QRNN","LR"}') #Do I want to keep LR here?
    parser.add_argument('-cell_type', type = str, default = 'GRU', help='For RNN based models, choose from {"RNN", "GRU", "LSTM", "QRNN" (for QRNN model only)}')
    ####Think about whether you want to keep this RNN or LR based, or just call all different models
    parser.add_argument('-input_size', type = list, default =[15817], help='''input dimension(s), decide which embedding types to use. 
                        If len of 1, then  1 embedding; len of 3, embedding medical, diagnosis and others separately (3 embeddings) [default:[15817]]''')
    parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-hidden_size', type=int, default=128, help='size of hidden layers [default: 128]')
    parser.add_argument('-dropout_r', type=float, default=0.1, help='the probability for dropout[default: 0.1]')
    parser.add_argument('-n_layers', type=int, default=3, help='number of Layers, for Dilated RNNs, dilations will increase exponentialy with mumber of layers [default: 1]')
    parser.add_argument('-bii', type=bool, default=False, help='indicator of whether Bi-directin is activated. [default: False]')
    parser.add_argument('-time', type=bool, default=False, help='indicator of whether time is incorporated into embedding. [default: False]')
    parser.add_argument('-preTrainEmb', type= str, default='', help='path to pretrained embeddings file. [default:'']')
    parser.add_argument("-output_dir",type=str, default= '../models/', help="The output directory where the best model will be saved and logs written [default: we will create'../models/'] ")
    
    # training 
    parser.add_argument('-lr', type=float, default=10**-4, help='learning rate [default: 0.0001]')
    parser.add_argument('-L2', type=float, default=10**-4, help='L2 regularization [default: 0.0001]')
    parser.add_argument('-epochs', type=int, default= 100, help='number of epochs for training [default: 100]')
    parser.add_argument('-patience', type=int, default= 20, help='number of stagnant epochs to wait before terminating training [default: 20]')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for training, validation or test [default: 128]')
    parser.add_argument('-optimizer', type=str, default='adam', choices=  ['adam','adadelta','adagrad', 'adamax', 'asgd','rmsprop', 'rprop', 'sgd'], 
                        help='Select which optimizer to train [default: adam]. Upper/lower case does not matter') 
    #maybe later? choose the GPU working on 
    #parser.add_argument('-cuda', type= bool, default=True, help='whether GPU is available [default:True]')
    args = parser.parse_args()
    
    
    ## Move LR processing to a different module
    '''
    ## Move LR processing to a different module?? Maybe
    ## simple load before giving to loader 
    if args.which_model == 'LR':
        #call another function to clean up the data first before feeding it into the loader 
        model_x = []
        for patient in set_x:
            model_x.append([each for visit in patient for each in visit])  
    else: 
        model_x = set_x     
    '''
    
    ####Step1. Data preparation
    print(colored("\nLoading and preparing data...", 'green'))    
    data = EHRdataFromPickles(root_dir = args.root_dir, 
                              file = args.file, 
                              sort= False,
                              test_ratio = args.test_ratio, 
                              valid_ratio = args.valid_ratio) #prevent shuffle before splitting
    #see an example
    #can comment out 
    print(data.__getitem__(40, seeDescription = True)) #get a smaller one please 
    
    # Dataloader splits
    train, test, valid = data.__splitdata__() #this time, sort is true
    # can comment out this part if you dont want to know what's going on here
    print(colored("\nSample data after split:", 'green'))
    print(
      "train: {}".format(train[-1]),
      "test: {}".format(test[-1]),
      "validation: {}".format(valid[-1]), sep='\n')
    print(colored("\nSample data lengths for train, test and validation:", 'green'))
    print(len(train), len(test), len(valid))
    #separate loader for train, test, validation 
    trainloader = EHRdataloader(train) 
    validloader = EHRdataloader(valid)
    testloader = EHRdataloader(test)
    
    
    #####Step2. Model loading
    if args.which_model == 'RNN': 
        ehr_model = model.EHR_RNN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  dropout_r=args.dropout_r,
                                  cell_type=args.cell_type,
                                  bii= args.bii,
                                  time= args.time,
                                  preTrainEmb= args.preTrainEmb) 
    elif args.which_model == 'DRNN': 
        ehr_model = model.EHR_DRNN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  dropout_r=args.dropout_r, #default =0 
                                  cell_type=args.cell_type, #default = 'GRU'
                                  bii= False,
                                  time = args.time, 
                                  preTrainEmb= args.preTrainEmb)     
    elif args.which_model == 'QRNN': 
        ehr_model = model.EHR_DRNN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  dropout_r=args.dropout_r, #default =0.1
                                  cell_type= 'QRNN', #doesn't support normal cell types
                                  bii= False, #QRNN doesn't support bi
                                  time = args.time, 
                                  preTrainEmb= args.preTrainEmb)  
    else: 
        ehr_model = model.EHR_LR_emb(input_size = args.input_size,
                                     embed_dim = args.embed_dim,
                                     preTrainEmb= args.preTrainEmb)
    #make sure cuda is working
    if use_cuda:
        ehr_model = ehr_model.cuda() 
    #model optimizers to choose from. Upper/lower case dont matter
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(ehr_model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.L2)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(ehr_model.parameters(), 
                                   lr=args.lr, 
                                   weight_decay=args.L2)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(ehr_model.parameters(), 
                                  lr=args.lr, 
                                  weight_decay=args.L2) 
    elif args.optimizer.lower() == 'adamax':
        optimizer = optim.Adamax(ehr_model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.L2)
    elif args.optimizer.lower() == 'asgd':
        optimizer = optim.ASGD(ehr_model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.L2)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(ehr_model.parameters(), 
                                  lr=args.lr, 
                                  weight_decay=args.L2)
    elif args.optimizer.lower() == 'rprop':
        optimizer = optim.Rprop(ehr_model.parameters(), 
                                lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(ehr_model.parameters(), 
                              lr=args.lr, 
                              weight_decay=args.L2)
    else:
        raise NotImplementedError
 
    
    #######Step3. Train, validation and test. default: batch shuffle = true 
    try:
        ut.epochs_run(args.epochs, 
                      train = trainloader, 
                      valid = validloader, 
                      test = testloader, 
                      model = ehr_model, 
                      optimizer = optimizer,
                      shuffle = True, 
                      batch_size = args.batch_size, 
                      which_model = args.which_model, 
                      patience = args.patience,
                      output_dir = args.output_dir)
    #we can keyboard interupt now 
    except KeyboardInterrupt:
        print(colored('-' * 89, 'green'))
        print(colored('Exiting from training early','green'))
    
#do the main file functions and runs 
if __name__ == "__main__":
    main()    