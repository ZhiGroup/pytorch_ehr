
# coding: utf-8

# * This notebook is to demonstrate how to use Bayesian Optimization package on github <https://github.com/fmfn/BayesianOptimization> locally to tune hyperparamters for our models(LR, embedding dimension=2) for predicting heart failure onset risk on cerner sample data
# * For this demonstration, the data is the original 1 hospital (h143) previously used by retain, with 42,729 patients in total
# * The logistic regression model has the architecture of an embedding layer (embedding dimension =2), a linear activation and sigmoid transformation. The hyperparameters to be tuned are: learning rate: lr, and l2 regularization 
# * To implement this, first you need to install the package: however we modify the package file a bit to bypass error and keep on iterating. The modified files could be found at Experiments/modifiedBO
# * Then **important**: you need to define a function (in our case LR_tune()) which takes in the hyperparameters: l2, lr on logscale, run the model using models, Loaddata, and TrainVaTe modules and return the best validation auc
# * Be ware that this BO package will search float parameters, so if you have int or categorical parameters you want to tune, you might want to transform those values in your function before giving those to your models
# * Then **important**: call BO function and pass your LR_tune(), a search range for each parameter ((-16, 1) means -16 and 1 inclusive), and give it points to explore (points that will give you large target values) if you want to, and call maximize() and pass number of iterations you want to run BO
# * Then you will get results of your initial designated explored points(if any), 5 initializations, and plus number of BO iterations
# * For our results: it improved our best validation auc **from manually tuned 0.76980** (l2 = np.exp(-11), lr = np.exp(-9)) to **0.78329** (l2 = np.exp(-9.0790), lr = np.exp(-7.3835))


from __future__ import print_function
from __future__ import division

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

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle


from bayes_opt import BayesianOptimization



import model as model 
import Loaddata as Loaddata
import TrainVaTe as TVT

# check GPU availability
use_cuda = torch.cuda.is_available()
use_cuda



# Load data set and target values
set_x = pickle.load(open('Data/h143.visits', 'rb'), encoding='bytes')
set_y = pickle.load(open('Data/h143.labels', 'rb'),encoding='bytes')

model_x = []
for patient in set_x:
    model_x.append([each for visit in patient for each in visit])  
    
merged_set= [[set_y[i],model_x[i]] for i in range(len(set_y))]
print("\nLoading and preparing data...")    
train1, valid1, test1 = Loaddata.load_data(merged_set)
print("\nSample data after split:")  
print(train1[0])
print("model is", 'LR') 



#function to record comprehensive searching results 
def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()

logFile='testLR_final.log'                                                                                                                                                                                            
header = 'model|emb_dim|l2|lr|BestValidAUC|TestAUC|atEpoch'
print2file(header, logFile)


#Hyperparamters to tune for LR: l2, lr. Define a function to return the best validation AUC of the model 
def LR_tune(l2, lr):
    #little transformations to use the searched values                                       
    l2 = np.exp(l2) #base e
    lr = np.exp(lr) #base e
    ehr_model = model.EHR_LR(embed_dim = 2)  
    if use_cuda:
        ehr_model = ehr_model.cuda()
    optimizer = optim.Adam(ehr_model.parameters(), lr=lr, weight_decay=l2)
    
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    
    for ep in range(25): 
        current_loss, train_loss, _ = TVT.train(train1, model= ehr_model, optimizer = optimizer, batch_size = 1) 
        avg_loss = np.mean(train_loss)
        valid_auc, y_real, y_hat, _  = TVT.calculate_auc(model = ehr_model, data = valid1, which_model = 'LR', batch_size = 1)  
        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            bestTestAuc, y_real, y_hat,_ = TVT.calculate_auc(model = ehr_model, data = test1, which_model = 'LR', batch_size = 1)

        if ep - bestValidEpoch >10:
            break   
                
    buf = '|%f |%f |%d ' % (bestValidAuc, bestTestAuc, bestValidEpoch)
    pFile= 'LR'+'|'+'|'+str(l2)+'|'+str(lr)+'|'+buf    
    print2file(pFile, logFile)      

    return bestValidAuc


if __name__ == "__main__":
    gp_params = {"alpha": 1e-4}

    LRBO = BayesianOptimization(LR_tune,
        {'l2': (-16, 1), 'lr': (-11, -2) })
    LRBO.explore({'l2': [-11], 'lr': [-9]})

    LRBO.maximize(n_iter=30, **gp_params)

    print('-' * 53)
    print('Final Results')
    print('LR: %f' % LRBO.res['max']['max_val'])

