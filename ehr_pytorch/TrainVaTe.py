# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:50:18 2018

@author: jzhu8
"""
 #train, validation and test related
from __future__ import print_function, division
from io import open
import string
import re
import random
import math 
import time 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data

#from torchviz import make_dot, make_dot_from_trace  
#I dont have this on anaconda 

from sklearn.metrics import roc_auc_score  #modified,import this for auc 
from sklearn.metrics import roc_curve 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


use_cuda = torch.cuda.is_available()


#less important: small pieces of functions
"""
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
"""

#loss plot 
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
 

#major part 
#One sample or minibatch
def trainsample(sample, model,  optimizer, criterion = nn.BCELoss()): #, mb = False):
    #hidden = model.initHidden()  #modified the class name in this section
    model.zero_grad()

    #for i in range(len(ehr_seq_tensor)):
        #output, hidden = model(ehr_seq_tensor[i],hidden) #generalized for different models
    output, label_tensor = model(sample)    
    loss = criterion(output, label_tensor)
        
    #if optimizer != None:
    loss.backward()
    optimizer.step()
        
    return output, loss.data[0]    


def train(data, model, optimizer, batch_size = 1, print_every = 10, plot_every = 5): # mb = False:  #just trainsample one by one 
    #for nn models: laila's default parameters are like: print_every = 10, plot_every = 5
    #for LR model: laila's default parameters are like: print_every = 5000, plot_every = 1000
    data.sort(key=lambda pt:len(pt[1]))
    
    current_loss = 0 
    all_losses =[]
    n_iter = 0 
    
    n_batches = int(np.ceil((len(data)) / int(batch_size)))
    #start = time.time()
    
    for index in random.sample(range(n_batches), n_batches):
        batch = data[index*batch_size:(index+1)*batch_size]
        output, loss = trainsample(batch, model, optimizer)
        current_loss += loss
        n_iter +=1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0    
            
    return current_loss, all_losses


def calculate_auc(model, data, which_model, batch_size=1):  #here should be different for different models, for Laila's should be reshape, and extend 
    n_batches = int(np.ceil(int(len(data)) / int(batch_size)))
    y_real =[]
    y_hat= []
    for index in range(n_batches):
            batch = data[index*batch_size:(index+1)*batch_size]
            output, label_tensor = model(batch)
            if which_model == 'LR':
                y_hat.append(output.cpu().data.numpy()[0][0])
                y_real.append(label_tensor.cpu().data.numpy()[0][0])
            else:
                y_hat.extend(output.cpu().data.view(-1).numpy())  #modify something!
                y_real.extend(label_tensor.cpu().data.view(-1).numpy())
    #print (labelVec, y_hat)
    auc = roc_auc_score(y_real, y_hat)
    return auc, y_real, y_hat 

def auc_plot(y_real, y_hat):
    fpr, tpr, _ = roc_curve(y_real,  y_hat)
    auc = roc_auc_score(y_real, y_hat)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    
    
    