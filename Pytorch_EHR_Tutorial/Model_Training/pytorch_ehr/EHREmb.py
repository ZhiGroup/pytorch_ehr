"""
This Class is mainly for the creation of the EHR patients' visits embedding
which is the key input for all the deep learning models in this Repo

@authors: Lrasmy , Jzhu @ DeguiZhi Lab - UTHealth SBMI
Last revised Feb 20 2020

"""

import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
use_cuda = torch.cuda.is_available()

#construct a whole embedding class from pytorch nn.module
#then we call this class in models after we define it 
class EHREmbeddings(nn.Module):
    #initialization and then the forward and things
    #DRNN has no bi, QRNN no bi, TLSTM has no bi, but DRNN has other cell-types 
    #cel_type are different for each model variation 
    def __init__(self, input_size, embed_dim ,hidden_size, n_layers=1,dropout_r=0.1,cell_type='LSTM', bii=False, time=False , preTrainEmb='', packPadMode = True, surv = False , hlf=False, cls_dim=1):
        super(EHREmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.time=time
        self.preTrainEmb=preTrainEmb
        self.packPadMode = packPadMode
        self.surv = surv
        self.cls_dim=cls_dim
        self.hlf=hlf

        if bii: 
            self.bi=2 
        else: 
            self.bi=1
            
        if len(input_size)==1:
            self.multi_emb=False
            if len(self.preTrainEmb)>0:
                emb_t= torch.FloatTensor(np.asmatrix(self.preTrainEmb))
                self.embed= nn.Embedding.from_pretrained(emb_t)#,freeze=False) 
                self.in_size= embed_dim ### need to be updated to be automatically capyured from the input
            else:
                input_size=input_size[0]
                self.embed= nn.Embedding(input_size, self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
                self.in_size= embed_dim
        else:
            if len(input_size)!=3: 
                raise ValueError('the input list is 1 length')
            else: 
                self.multi_emb=True
                self.diag=self.med=self.oth=1

        #self.emb = self.embed.weight  LR commented Jul 10 19
        if self.time: self.in_size= self.in_size+1 
        
               
        if self.cell_type == "GRU":
            self.cell = nn.GRU
        elif self.cell_type == "RNN":
            self.cell = nn.RNN
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTM
        elif self.cell_type == "QRNN":
            from torchqrnn import QRNN
            self.cell = QRNN
        elif self.cell_type == "TLSTM":
            from tplstm import TPLSTM
            self.cell = TPLSTM 
        else:
            raise NotImplementedError
       
        if self.cell_type == "QRNN": 
            self.bi=1 ### QRNN not support Bidirectional, DRNN should not be BiDirectional either.
            self.rnn_c = self.cell(self.in_size, self.hidden_size, num_layers= self.n_layers, dropout= self.dropout_r)
        elif self.cell_type == "TLSTM":
            self.bi=1 
            self.rnn_c = self.cell(self.in_size, hidden_size)
        else:
            self.rnn_c = self.cell(self.in_size, self.hidden_size, num_layers=self.n_layers, dropout= self.dropout_r, bidirectional=bii, batch_first=True) 
         
        self.out = nn.Linear(self.hidden_size*self.bi,self.cls_dim)
        if self.cls_dim==1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax(dim=1) 

        self.relu= nn.ReLU()
        self.logsoftmax= nn.LogSoftmax()
        
                            
    #let's define this class method
    def EmbedPatients_MB(self,mb_t, mtd): #let's define this
        self.bsize=len(mb_t) ## no of pts in minibatch
        embedded = self.embed(mb_t)  ## Embedding for codes
        embedded = torch.sum(embedded, dim=2) 
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: 
                mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        if use_cuda:
            out_emb.cuda() 
        if self.hlf: 
            out_emb.half()
            #print('hlf applied')
        return out_emb
    
