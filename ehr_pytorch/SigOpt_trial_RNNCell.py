
# coding: utf-8

# This is to demonstrate how to use SigOpt software for hyperparameter tuning: 
# * After importing necessary modules, first create a connection to SigOpt API, you will need an account that you can sign up here: <https://sigopt.com/edu> and get the client token under the tab **API Tokens**
# * Do the normal steps of Bayesian Optimization up till the creation of the function model_tune() which takes in the hyperparameters and return the best validation AUC (sidenote: since everything will be recorded at SigOpt API and you can download it as csv file later, you can get rid of recording searched parameters on your own)
# * Then **important**: create an experiment with the connection to SigOpt API specifying the experiment name, the hypreparameters searching space, the type of hyperparameters  (int, float, categorical), and finally observation budget (which is the number of iterations). I have all types in here already, so you can just copy as needed 
# * You can print out the experiment id link so you can easily check out experiment results
# * Then **important**: you will need an function to pass the parameters you created in the experiments to model_tune(). Assignments are the default results from experiments, so you need to call 'assignments' dictionary for different hyperparameters 
# * Then **important**: create iterations(observations), the API will automatically update suggestions based on best result
# * Then fetch the best experiment results after all iterations are done, but you can also manually retrieve the result at the experiment id link above
# * Additional infor that might be need: delete an experiment (free trial account has 10 experiments limit, so be sure to manage your unwanted ones.


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
    
import pandas as pd

from sigopt import Connection
conn = Connection(client_token='QULCGRMSXQAMRSTUPJPLAHDCHJMEMLXPSPZSQMWNEYKHLYJF')

torch.cuda.set_device(0)
torch.backends.cudnn.enabled=False


import model_HPS_V2 as model #this changed
import Loaddata_final as Loaddata
import TrVaTe_V2 as TVT #This changed 

# check GPU availability
use_cuda = torch.cuda.is_available()
use_cuda



# Load data set and target values
set_x = pickle.load(open('Data/h143.visits', 'rb'), encoding='bytes')
set_y = pickle.load(open('Data/h143.labels', 'rb'),encoding='bytes')

"""
model_x = []
for patient in set_x:
    model_x.append([each for visit in patient for each in visit])  
    
"""
model_x = set_x  #this is for the rest of the models
    
merged_set= [[set_y[i],model_x[i]] for i in range(len(set_y))] #list of list or list of lists of lists
print("\nLoading and preparing data...")    
train1, valid1, test1 = Loaddata.load_data(merged_set)
print("\nSample data after split:")  
print(train1[1])
print("model is", 'RNN') #can change afterwards, currently on most basic RNN

epochs = 100


#test on RNN with celltype plain RNN, with differnent optimizers also as an option 
def model_tune(embdim_exp, hid_exp, layers_n, dropout, l2_exp , lr_exp, eps_exp, opt_code):
    #little transformations to use the searched values
    embed_dim = 2** int(embdim_exp)
    hidden_size = 2** int(hid_exp)
    n_layers = int(layers_n)
    dropout = round(dropout, 4)
    l2 = 10** int(l2_exp)
    lr = 10** int(lr_exp)
    eps = 10** (eps_exp)
      
       
    ehr_model = model.EHR_RNN(20000, embed_dim, hidden_size, n_layers, dropout, cell_type = 'RNN')

    if use_cuda:
        ehr_model = ehr_model.cuda(0)   
    
    if opt_code == 'Adadelta':
        opt= 'Adadelta'
        optimizer = optim.Adadelta(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps) ## rho=0.9
    elif opt_code == 'Adagrad':
        opt= 'Adagrad'
        optimizer = optim.Adagrad(ehr_model.parameters(), lr=lr, weight_decay=l2) ##lr_decay no eps
    elif opt_code =='Adam':
        opt= 'Adam'
        optimizer = optim.Adam(ehr_model.parameters(), lr=lr, weight_decay=l2,eps=eps ) ## Beta defaults (0.9, 0.999), amsgrad (false)
    elif opt_code =='Adamax':
        opt= 'Adamax'
        optimizer = optim.Adamax(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps) ### Beta defaults (0.9, 0.999)
    elif opt_code =='RMSprop':
        opt= 'RMSprop'
        optimizer = optim.RMSprop(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps)                
    elif opt_code =='ASGD':
        opt= 'ASGD'
        optimizer = optim.ASGD(ehr_model.parameters(), lr=lr, weight_decay=l2 ) ### other parameters
    elif opt_code =='SGD':
        opt= 'SGD'
        optimizer = optim.SGD(ehr_model.parameters(), lr=lr, weight_decay=l2 ) ### other parameters

     
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
  
    for ep in range(epochs):
        current_loss, train_loss = TVT.train(train1, model= ehr_model, optimizer = optimizer, batch_size = 200)
        avg_loss = np.mean(train_loss)
        valid_auc, y_real, y_hat  = TVT.calculate_auc(model = ehr_model, data = valid1, which_model = 'RNN', batch_size = 200)
        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            best_model= ehr_model
           

        if ep - bestValidEpoch > 12:
            break
      
  
    bestTestAuc, y_real, y_hat = TVT.calculate_auc(model = best_model, data = test1, which_model = 'RNN', batch_size = 200)
    print(bestTestAuc)
    return bestValidAuc



# create 3 different experiments for 3 different cells, keep the optimizer categorical 
experiment = conn.experiments().create(
  name='RNN_AUC_Optimization',
  parameters=[
    dict(name='embdim_exp', type='int', bounds=dict(min=5, max=9)),
    dict(name='hid_exp', type='int', bounds=dict(min=5, max=9)),
    dict(name='layers_n', type='int', bounds=dict(min=1, max=3)),
    dict(name='dropout', type='double', bounds=dict(min=0.1000, max= 0.9000)),
    dict(name='lr_exp', type='int', bounds=dict(min=-7, max=-2)), 
    dict(name='l2_exp', type='int', bounds=dict(min=-7, max=-1)),
    dict(name='eps_exp', type='int', bounds=dict(min=-9, max=-4)),  
    dict(name='opt_code', type='categorical', categorical_values=[dict(name='Adadelta'), dict(name='Adagrad'),
                                                                  dict(name='Adam'), dict(name='Adamax'),dict(name='RMSprop'),
                                                                  dict(name='ASGD'), dict(name='SGD')]),
  ],
    observation_budget=300
)


print("Created experiment: https://sigopt.com/experiment/" + experiment.id);


# Evaluate your model with the suggested parameter assignments
def evaluate_model(assignments):
  return model_tune(assignments['embdim_exp'], assignments['hid_exp'],assignments['layers_n'],assignments['dropout'],
                    assignments['lr_exp'],assignments['l2_exp'],assignments['eps_exp'],assignments['opt_code'])


#run the experiments, no need to modify
for i in range(300):
  suggestion = conn.experiments(experiment.id).suggestions().create()
  value = evaluate_model(suggestion.assignments)
  conn.experiments(experiment.id).observations().create(
    suggestion=suggestion.id,
    value=value,
  )



#Wrapping up the experiments and get the results
best_assignments_list = (
    conn.experiments(experiment.id)
        .best_assignments()
        .fetch()
)
if best_assignments_list.data:
    best_assignments = best_assignments_list.data[0].assignments

#delete unwanted experiments 
experiment = conn.experiments('someid').delete()

