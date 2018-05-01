# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 10:16:05 2018

@author: jzhu8
"""
import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


# Model 1: simple Logistic Regression
class EHR_LR(nn.Module):
    def __init__(self,input_size, embed_dim= 128, preTrainEmb=False):    
        super(EHR_LR, self).__init__()
    
        self.embed_dim = embed_dim
        #self.embedding = nn.Embedding(input_size,embed_dim)
        self.out = nn.Linear(self.embed_dim,1)
        self.sigmoid = nn.Sigmoid()
        self.preTrainEmb=preTrainEmb
        #Need to modify here 
        if self.preTrainEmb:
           self.embed_dim = emb_t.size(1)
           input_size = emb_t.size(0)
           self.embedding = nn.EmbeddingBag(emb_t.size(0),emb_t.size(1))#,mode= 'sum')
           self.embedding.weight.data= emb_t
           self.embedding.weight.requires_grad=False
         
        else:
           self.embedding = nn.Embedding(input_size, self.embed_dim) #mode= 'sum')
        

    def forward(self, input):  #lets say the input the one sample data of the merged set  
        label, ehr_seq = input[0] 
        #print(input[0]) 
        label_tensor = Variable(torch.FloatTensor([[float(label)]]))
        if use_cuda:
            label_tensor = label_tensor.cuda()
        if use_cuda:
            result = Variable(torch.LongTensor([int(v) for v in ehr_seq])).cuda() 
        else:
            result = Variable(torch.LongTensor([int(v) for v in ehr_seq])) 
        embedded = self.embedding(result).view(-1, self.embed_dim) #modified, instead of (-1,1, self.hidden_size) => use (-1,self.hidden_size)
        embedded = torch.sum(embedded, dim=0).view(1,-1)#modified,instead of (1,1,-1) => use .view(1,-1) 
        output = self.sigmoid(self.out(embedded))
        
        return output, label_tensor #return output and also label tensor 



# Model 2:RNN     
class EHR_RNN(nn.Module):
    def __init__(self, input_size, embed_dim, dropout, eb_mode, n_layers=1, hidden_size =128):
        super(EHR_RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.embedBag = nn.EmbeddingBag(input_size, self.embed_dim, mode= eb_mode)
        self.gru = nn.GRU(self.embed_dim, hidden_size, dropout= dropout )
        self.out = nn.Linear(self.hidden_size,1)
        self.sigmoid = nn.Sigmoid()


        
    def EmbedPatient_MB(self, seq_mini_batch): # x is a ehr_seq_tensor
        
        lp= len(max(seq_mini_batch, key=lambda xmb: len(xmb[1]))[1]) # max number of visitgs within mb ??? verify again
        #print ('longest',lp)
        tb= torch.FloatTensor(len(seq_mini_batch),lp,self.embed_dim) 
        lbt1= torch.FloatTensor(len(seq_mini_batch),1)

        for pt in range(len(seq_mini_batch)):
              
            lbt ,pt_visits =seq_mini_batch[pt]
            lbt1[pt] = torch.FloatTensor([[float(lbt)]])
            ml=(len(max(pt_visits, key=len))) ## getting the visit with max no. of codes ##the max number of visits for pts within the minibatch
            txs= torch.LongTensor(len(pt_visits),ml)
            
            b=0
            for i in pt_visits:
                pd=(0, ml-len(i))
                txs[b] = F.pad(torch.from_numpy(np.asarray(i)).view(1,-1),pd,"constant", 0).data
                b=b+1
            
            if use_cuda:
                txs=txs.cuda()
                
            emb_bp= self.embedBag(Variable(txs)) ### embed will be num_of_visits*max_num_codes*embed_dim 
            #### the embed Bag dim will be num_of_visits*embed_dim
            
            zp= nn.ZeroPad2d((0,0,0,(lp-len(pt_visits))))
            xzp= zp(emb_bp)
            tb[pt]=xzp.data

        tb= tb.permute(1, 0, 2) ### as my final input need to be seq_len x batch_size x input_size
        emb_m=Variable(tb)
        label_tensor = Variable(lbt1)

        if use_cuda:
                label_tensor = label_tensor.cuda()
                emb_m = emb_m.cuda()
                
        return emb_m , label_tensor

    def forward(self, input):
        
        x_in , lt = self.EmbedPatient_MB(input)
        
        for i in range(self.n_layers):
                output, hidden = self.gru(x_in) # input (seq_len, batch, input_size) need to check torch.nn.utils.rnn.pack_padded_sequence() 
                                                          #h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.

        output = self.sigmoid(self.out(output[0]))
        return output, lt     #output, and (TRUE) label tensor

#Model3: CNN
class EHR_CNN(nn.Module):
    
    def __init__ (self, input_size, embed_dim, dropout,eb_mode, ch_out, kernel_sizes):
        super(EHR_CNN, self).__init__()
        
        P = input_size ## number of predictors in our model
        D = embed_dim
        self.D = D
        C = 1
        Ci = 1
        Co = ch_out ## number of Channels out
        Ks = kernel_sizes ## a list for different kernel(filters dimension)

        self.embedBag = nn.EmbeddingBag(P, D,mode= eb_mode)
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=2) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.sigmoid = nn.Sigmoid()
   
    def EmbedPatient_MB(self, seq_mini_batch): # x is a ehr_seq_tensor
        
       
        lp= len(max(seq_mini_batch, key=lambda xmb: len(xmb[1]))[1])
        #print ('longest',lp)
        tb= torch.FloatTensor(len(seq_mini_batch),lp,self.D) 
        lbt1= torch.FloatTensor(len(seq_mini_batch),1)

        for pt in range(len(seq_mini_batch)):
              
            lbt ,pt_visits =seq_mini_batch[pt]
            lbt1[pt] = torch.FloatTensor([[float(lbt)]])
            ml=(len(max(pt_visits, key=len))) ## getting the max number of visits for pts within the minibatch
            txs= torch.LongTensor(len(pt_visits),ml)
            
            b=0
            for i in pt_visits:
                pd=(0, ml-len(i))
                txs[b] = F.pad(torch.from_numpy(np.asarray(i)).view(1,-1),pd,"constant", 0).data
                b=b+1
            
            if use_cuda:
                txs=txs.cuda()
                
            emb_bp= self.embedBag(Variable(txs)) ### embed will be num_of_visits*max_num_codes*embed_dim 
            #### the embed Bag dim will be num_of_visits*embed_dim
            
            zp= nn.ZeroPad2d((0,0,0,(lp-len(pt_visits))))
            xzp= zp(emb_bp)
            tb[pt]=xzp.data

        tb= tb.permute(1, 0, 2) ### as my final input need to be seq_len x batch_size x input_size
        emb_m=Variable(tb)
        label_tensor = Variable(lbt1)

        if use_cuda:
                label_tensor = label_tensor.cuda()
                emb_m = emb_m.cuda()
                
        return emb_m , label_tensor


    def forward(self, input):
        #x = self.EmbedPatient(input) # [seqlen*batchsize*embdim]

        x , lt = self.EmbedPatient_MB(input)
        x = x.permute(1,2,0) # [N, Co, W]
        
        x = [F.relu(conv(x)) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        y = self.sigmoid(logit)
        
        return y , lt