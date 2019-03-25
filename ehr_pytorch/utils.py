# -*- coding: utf-8 -*-
# coding: utf-8
"""
Created on Mon Oct 29 12:57:40 2018

@author: ginnyzhu
"""
from __future__ import print_function, division
from io import open
#import string
#import re
#import random
import math 
import time 
import os

import torch
import torch.nn as nn
#from torch.autograd import Variable
#from torch import optim
#import torch.nn.functional as F
#import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 

import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np

#Rename afterwards
import EHRDataloader
from EHRDataloader import iter_batch2
#silly ones
from termcolor import colored


#check this later
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

###### minor functions, plots and prints
#loss plot
def showPlot(points):
    fig, ax = plt.subplots()
    plt.plot(points)
    plt.show()

 
#auc_plot 
def auc_plot(y_real, y_hat):
    fpr, tpr, _ = roc_curve(y_real,  y_hat)
    auc = roc_auc_score(y_real, y_hat)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.show();

        
#time Elapsed
def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)    


#print to file function
def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


###### major model training utilities
def trainsample(sample, model, optimizer, criterion = nn.BCELoss()): 
    model.zero_grad()
    output, label_tensor = model(sample)   
    loss = criterion(output, label_tensor)    
    loss.backward()   
    optimizer.step()
    #print(loss.item())
    return output, loss.item()


#train with loaders

def trainbatches(loader, model, optimizer, shuffle = True):#,we dont need this print print_every = 10, plot_every = 5): 
    current_loss = 0
    all_losses =[]
    plot_every = 5
    n_iter = 0 
    if shuffle: 
         #we shuffle batches if shuffle is true
         loader = iter_batch2(loader, len(loader))
    for i,batch in enumerate(loader):
        #batch.to(device) #see if it works
        output, loss = trainsample(batch, model, optimizer, criterion = nn.BCELoss())
        current_loss += loss
        n_iter +=1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0    
        #print(all_losses) #check    
    return current_loss, all_losses 



def calculate_auc(model, loader, which_model = 'RNN', shuffle = True): # batch_size= 128 not needed

    y_real =[]
    y_hat= []
    if shuffle: 
         loader = iter_batch2(loader, len(loader)) 
    for i,batch in enumerate(loader):
            #batch.to(device) #check you want it or not 
            output, label_tensor = model(batch)
            if which_model != 'LR':
                y_hat.extend(output.cpu().data.view(-1).numpy())  
                y_real.extend(label_tensor.cpu().data.view(-1).numpy())
         
            else: 
                #The minor case, basically embedding LR and GRU-LR case. Do we want to keep it?
                y_hat.append(output.cpu().data.numpy()[0][0])
                y_real.append(label_tensor.cpu().data.numpy()[0][0])
    
    auc = roc_auc_score(y_real, y_hat)
    return auc, y_real, y_hat 

    
#define the final epochs running, use the different names

def epochs_run(epochs, train, valid, test, model, optimizer, shuffle = True, which_model = 'RNN', patience = 20, output_dir = '../models/', model_prefix = 'hf.train', model_customed= ''):  
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    #header = 'BestValidAUC|TestAUC|atEpoch'
    #logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    #print2file(header, logFile)
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = trainbatches(loader = train, model= model, optimizer = optimizer)

        train_time = timeSince(start)
        #epoch_loss.append(train_loss)
        avg_loss = np.mean(train_loss)
        train_auc, _, _ = calculate_auc(model = model, loader = train, which_model = which_model, shuffle = shuffle)
        valid_start = time.time()
        valid_auc, _, _ = calculate_auc(model = model, loader = valid, which_model = which_model, shuffle = shuffle)
        valid_time = timeSince(valid_start)
        print(colored('\nCurrent running on Epoch (%s), Average_loss (%s)'%(ep, avg_loss), 'green'))
        print(colored('Train_auc (%s), Valid_auc (%s)'%(train_auc, valid_auc),'green'))
        print(colored('Train_time (%s), Valid_time (%s)'%(train_time, valid_time),'green'))
        if valid_auc > bestValidAuc: 
              bestValidAuc = valid_auc
              bestValidEpoch = ep
              best_model= model          
        if ep - bestValidEpoch > patience:
              break
          
    bestTestAuc, _, _ = calculate_auc(model = best_model, loader = test, which_model = which_model, shuffle = shuffle)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #save model & parameters
    torch.save(best_model, output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    torch.save(best_model.state_dict(), output_dir + model_prefix + model_customed + 'EHRmodel.st')
    '''
    #later you can do to load previously trained model:
    best_model= torch.load(args.output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    best_model.load_state_dict(torch.load(args.output_dir + model_prefix + model_customed + 'EHRmodel.st'))
    best_model.eval()
    '''
    #Record in the log file
    header = 'BestValidAUC|TestAUC|atEpoch'
    logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    print2file(header, logFile)
    pFile = '|%f |%f |%d ' % (bestValidAuc, bestTestAuc, bestValidEpoch)
    print2file(pFile, logFile) 
    print(colored('BestValidAuc %f has a TestAuc of %f at epoch %d ' % (bestValidAuc, bestTestAuc, bestValidEpoch),'green'))
    print(colored('Details see ../models/%sEHRmodel.log' %(model_prefix + model_customed),'green'))

    
    