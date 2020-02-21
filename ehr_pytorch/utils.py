# -*- coding: utf-8 -*-
# coding: utf-8
# -*- coding: utf-8 -*-
"""
This Class is mainly for the creation of the EHR patients' visits embedding
which is the key input for all the deep learning models in this Repo
@authors: Lrasmy , Jzhu @ DeguiZhi Lab - UTHealth SBMI
Last revised Feb 20 2020
"""
from __future__ import print_function, division
from io import open
#import string
#import re
import random
import math 
import time 
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 

import matplotlib.pyplot as plt
import numpy as np

#Rename afterwards
import EHRDataloader
from EHRDataloader import iter_batch2
from termcolor import colored



use_cuda = torch.cuda.is_available()


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
def trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion = nn.BCELoss()): 
    model.train() ## LR added Jul 10, that is the right implementation
    model.zero_grad()
    output = model(sample,seq_l, mtd)   
    loss = criterion(output, label_tensor)    
    loss.backward()   
    optimizer.step()
    # print(loss.item())
    return output, loss.item()


#train with loaders

def trainbatches(mbs_list, model, optimizer, shuffle = True):#,we dont need this print print_every = 10, plot_every = 5): 
    current_loss = 0
    all_losses =[]
    plot_every = 5
    n_iter = 0 
    if shuffle: 
         # you can also shuffle batches using iter_batch2 method in EHRDataloader
        #  loader = iter_batch2(mbs_list, len(mbs_list))
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mtd = batch
        output, loss = trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion = nn.BCELoss())
        current_loss += loss
        n_iter +=1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0    
    return current_loss, all_losses 



def calculate_auc(model, mbs_list, which_model = 'RNN', shuffle = True): # batch_size= 128 not needed
    model.eval() ## LR added Jul 10, that is the right implementation
    y_real =[]
    y_hat= []
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):

        sample, label_tensor, seq_l, mtd = batch
        output = model(sample, seq_l, mtd)
        y_hat.extend(output.cpu().data.view(-1).numpy())  
        y_real.extend(label_tensor.cpu().data.view(-1).numpy())
         
    auc = roc_auc_score(y_real, y_hat)
    return auc, y_real, y_hat 

    
#define the final epochs running, use the different names

def epochs_run(epochs, train, valid, test, model, optimizer, shuffle = True, which_model = 'RNN', patience = 20, output_dir = '../models/', model_prefix = 'dhf.train', model_customed= ''):  
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    #header = 'BestValidAUC|TestAUC|atEpoch'
    #logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    #print2file(header, logFile)
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = trainbatches(mbs_list = train, model= model, optimizer = optimizer)

        train_time = timeSince(start)
        #epoch_loss.append(train_loss)
        avg_loss = np.mean(train_loss)
        valid_start = time.time()
        train_auc, _, _ = calculate_auc(model = model, mbs_list = train, which_model = which_model, shuffle = shuffle)
        valid_auc, _, _ = calculate_auc(model = model, mbs_list = valid, which_model = which_model, shuffle = shuffle)
        valid_time = timeSince(valid_start)
        print(colored('\n Epoch (%s): Train_auc (%s), Valid_auc (%s) ,Training Average_loss (%s), Train_time (%s), Eval_time (%s)'%(ep, train_auc, valid_auc , avg_loss,train_time, valid_time), 'green'))
        if valid_auc > bestValidAuc: 
              bestValidAuc = valid_auc
              bestValidEpoch = ep
              best_model= model 
              if test:      
                      testeval_start = time.time()
                      bestTestAuc, _, _ = calculate_auc(model = best_model, mbs_list = test, which_model = which_model, shuffle = shuffle)
                      print(colored('\n Test_AUC (%s) , Test_eval_time (%s) '%(bestTestAuc, timeSince(testeval_start)), 'yellow')) 
                      #print(best_model,model) ## to verify that the hyperparameters already impacting the model definition
                      #print(optimizer)
        if ep - bestValidEpoch > patience:
              break
    #if test:      
    #   bestTestAuc, _, _ = calculate_auc(model = best_model, mbs_list = test, which_model = which_model, shuffle = shuffle) ## LR code reorder Jul 10
    
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
    #Record in the log file , modify below as you like, this is just as example
    header = 'BestValidAUC|TestAUC|atEpoch'
    logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    print2file(header, logFile)
    pFile = '|%f |%f |%d ' % (bestValidAuc, bestTestAuc, bestValidEpoch)
    print2file(pFile, logFile) 
    if test:
        print(colored('BestValidAuc %f has a TestAuc of %f at epoch %d ' % (bestValidAuc, bestTestAuc, bestValidEpoch),'green'))
    else: 
        print(colored('BestValidAuc %f at epoch %d ' % (bestValidAuc,  bestValidEpoch),'green'))
        print('No Test Accuracy')
    print(colored('Details see ../models/%sEHRmodel.log' %(model_prefix + model_customed),'green'))

    
    
