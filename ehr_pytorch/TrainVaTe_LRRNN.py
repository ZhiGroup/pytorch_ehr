# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:50:18 2018

@author: jzhu8
"""
#train, validation and test related
#for LR, embeddings can be plotted visually 
#for hyperparameter tuning and embbeding and general purposes
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


from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


use_cuda = torch.cuda.is_available()

#loss plot 
def showPlot(points):
    fig, ax = plt.subplots()
    plt.plot(points)
    plt.show()
 

#major part 
#One sample or minibatch
def trainsample(sample, model,  optimizer, criterion = nn.BCELoss()): 
    model.zero_grad()
    
    output, label_tensor, em = model(sample)   #forward function output 
    loss = criterion(output, label_tensor)    
    loss.backward()   #backward propagation for training
    optimizer.step()
        
    return output, loss.data[0], em   


def train(data, model, optimizer, batch_size = 1, print_every = 5000, plot_every = 1000): #last two parameters for plotting purposes
    data.sort(key=lambda pt:len(pt[1])) #sort data first according to length of visits 
    
    current_loss = 0 
    all_losses =[]
    n_iter = 0 
    
    n_batches = int(np.ceil((len(data)) / int(batch_size)))
    
    for index in random.sample(range(n_batches), n_batches):  #shuffle 
        batch = data[index*batch_size:(index+1)*batch_size]
        output, loss, em = trainsample(batch, model, optimizer) 
        current_loss += loss
        n_iter +=1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0    
            
    return current_loss, all_losses, em 


def calculate_auc(model, data, which_model, batch_size=1):  #this part is for 2D embedding dimension, modify as needed for 1D
    n_batches = int(np.ceil(int(len(data)) / int(batch_size))) #for getting results without visualization, comment out embedding related
    y_real =[]
    y_hat= []
    emb= np.array([[999.,999.]]) #placeholder
    for index in range(n_batches):
            batch = data[index*batch_size:(index+1)*batch_size]
            output, label_tensor, em = model(batch)
            if which_model == 'LR':
                y_hat.append(output.cpu().data.numpy()[0][0])
                y_real.append(label_tensor.cpu().data.numpy()[0][0])
                em = em.cpu().data.numpy()[0] #embedding matrix
                emb =np.vstack((emb,em))
            else: 
                y_hat.extend(output.cpu().data.view(-1).numpy())  
                y_real.extend(label_tensor.cpu().data.view(-1).numpy())
    
    auc = roc_auc_score(y_real, y_hat)
    return auc, y_real, y_hat, emb

def auc_plot(y_real, y_hat):
    fpr, tpr, _ = roc_curve(y_real,  y_hat)
    auc = roc_auc_score(y_real, y_hat)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    
    
    