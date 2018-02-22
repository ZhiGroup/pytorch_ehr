# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:14:40 2018

@author: jzhu8
"""

 #train, validation and test related
# model loading part
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

from sklearn.metrics import roc_auc_score  #modified,import this for auc 
from sklearn.metrics import roc_curve 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


use_cuda = torch.cuda.is_available()


#less important: small pieces of functions
#clock
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#loss plot 
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
 
# roc_auc curve plot
def auc_plot(y_real, y_cal):
    fpr, tpr, _ = roc_curve(y_real,  y_cal)
    auc = roc_auc_score(y_real, y_cal)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.show()


#major part 
# One sample, train or evaluation
#def onesample(label_tensor, ehr_seq_tensor, model, optimizer, criterion, whichdata = 'train'):#define whichdata 'train', 'eval'
def onesample(label_tensor, ehr_seq_tensor, model,  optimizer=None, criterion = nn.BCELoss()):    
    hidden = model.initHidden()  #modified the class name in this section
    model.zero_grad()

    for i in range(len(ehr_seq_tensor)):
        output, hidden = model(ehr_seq_tensor[i],hidden) #generalized for different models
        loss = criterion(output, label_tensor)
        
    if optimizer != None:
            loss.backward()
            optimizer.step()
        
    return output, loss.data[0]    

# model input: {'LR', 'RNN',...} 
def variableFromEHRSeq(ehr_seq, which_model):
    # ehr_seq is a list for LR, and a list of list for deep learning models, only different between LR and deep learning 
    result = []
    if which_model == 'LR':
        if use_cuda:
            result.append( Variable(torch.LongTensor([int(v) for v in ehr_seq])).cuda() ) 
        else:
            result.append( Variable(torch.LongTensor([int(v) for v in ehr_seq])) )
    else: 
        if use_cuda:
            for i in range(len(ehr_seq)):
                result.append( Variable(torch.LongTensor([int(v) for v in ehr_seq[i]])).cuda() )
        else:
            for i in range(len(ehr_seq)):
                result.append( Variable(torch.LongTensor([int(v) for v in ehr_seq[i]])) )
    return result


def train(data, model, optimizer,n_iters = 100000, print_every = 5000, plot_every = 1000):
    current_loss = 0 
    all_losses =[]
    start = time.time()
    
    for iter in range(1, n_iters + 1):  
        label, ehr_seq = random.choice(data)  #modified, using the train set
        label_tensor = Variable(torch.FloatTensor([[float(label)]]))
        if use_cuda:
            label_tensor = label_tensor.cuda() 
        ehr_seq_tensor = variableFromEHRSeq(ehr_seq, which_model ='LR') 
    
        output, loss = onesample(label_tensor, ehr_seq_tensor,  model , optimizer)
        current_loss += loss

    # Print iter number, training loss, name and guess
        if iter % print_every == 0:
            print('%d %d%% (%s) %.4f ' % (iter, iter / n_iters * 100, timeSince(start), loss))

     # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0    
            
    return all_losses

def evaluation(data, model):  #data could be validation or test
    y_cal=[]
    y_real=[]
    datasize = len(data)
    for i in range(datasize):  
        label, ehr_seq = data[i]  #predict one by one
        y_real.append(label)        #get the true labels 
        label_tensor = Variable(torch.FloatTensor([[float(label)]])) 
        if use_cuda:
            label_tensor = label_tensor.cuda() 
        ehr_seq_tensor = variableFromEHRSeq(ehr_seq, which_model='LR')
    
        output, _= onesample(label_tensor, ehr_seq_tensor, model) # Get the predicted labels
        output_y = output.data.cpu().numpy()[0][0]   #modified, trace back to the tensor, to.cpu() to numpy array, and take the element 
        y_cal.append(output_y)
        
    return y_real, y_cal   #as lists


