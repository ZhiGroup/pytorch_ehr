# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:57:40 2018

@author: ginnyzhu
"""
import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch import optim
import torch.nn.functional as F
#from torchqrnn import QRNN
#import bnlstm


##### modify the name of module accordingly
#from embedding import EHRembeddings 
from EHREmb import EHREmbeddings

use_cuda = torch.cuda.is_available()
##################TO DO: cell-type cleanup!
#### For DRNN & QRNN: should always override self.bi ==1 (ASK)

# Model 1:RNN & Variations: GRU, LSTM, Bi-RNN, Bi-GRU, Bi-LSTM
class EHR_RNN(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers=1,dropout_r=0.1,cell_type='GRU',bii=False ,time=False, preTrainEmb='',packPadMode = True):

       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers, dropout_r, cell_type, bii, time , preTrainEmb, packPadMode)



    #embedding function goes here 
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self, input)
    
    def EmbedPatient_SMB(self, input):
        return EHREmbeddings.EmbedPatients_SMB(self, input)       
     
    def init_hidden(self):
        
        h_0 = Variable(torch.rand(self.n_layers*self.bi,self.bsize, self.hidden_size))
        if use_cuda: 
            h_0.cuda()
        if self.cell_type == "LSTM":
            result = (h_0,h_0)
        else: 
            result = h_0
        return result
    
    def forward(self, input):
        #print(type(input))
        if self.multi_emb:
            x_in , lt ,x_lens = self.EmbedPatient_SMB(input)
        else: 
            x_in , lt ,x_lens = self.EmbedPatient_MB(input) 
        x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)   
        #h_0= self.init_hidden()
        output, hidden = self.rnn_c(x_inp)#,h_0) 
        if self.cell_type == "LSTM":
            hidden=hidden[0]
        if self.bi==2:
            output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1)))
        else:
            output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze(), lt.squeeze()


#Model 2: DRNN, DGRU, DLSTM
class EHR_DRNN(EHREmbeddings): 
    def __init__(self,input_size,embed_dim, hidden_size, n_layers, dropout_r=0.1,cell_type='GRU', bii=False, time=False, preTrainEmb='', packPadMode = False):

        EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers, dropout_r, cell_type, time , preTrainEmb, packPadMode)

        #super(DRNN, self).__init__()
        #The additional parameters that norma RNNs don't have

        self.dilations = [2 ** i for i in range(n_layers)]
        self.layers = nn.ModuleList([])
        if self.bi == 2:
            print('DRNN only supports 1-direction, implementing 1-direction instead')
        self.bi =1  #Enforcing 1-directional
        self.packPadMode = False #Enforcing no packpadded indicator 
        
        for i in range(n_layers):
            if i == 0:
                c = self.cell(self.in_size, self.hidden_size, dropout=self.dropout_r)
            else:
                c = self.cell(self.hidden_size, self.hidden_size, dropout=self.dropout_r)
            self.layers.append(c)
        self.cells = nn.Sequential(*self.layers)
        #check if DRNN can only be 1 directional, if that is the case then we always have self.bi = 1 
        #self.out = nn.Linear(hidden_size,1)

    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self,input)
    
    def EmbedPatient_SMB(self, input):
        return EHREmbeddings.EmbedPatients_SMB(self,input)    
    
    def forward(self, inputs, hidden=None):
        if self.multi_emb: 
            x , lt ,_ = self.EmbedPatient_SMB(inputs)
        else: 
            x , lt ,_ = self.EmbedPatient_MB(inputs)

        x=x.permute(1,0,2)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                x,_ = self.drnn_layer(cell, x, dilation) ############## WE HAVE THE DRNN LAYER HERE. LET"S TRY THINGS. IN THE SAME CLASS WE DEFINE EHREMBEDDINGD
            else:
                x,hidden[i] = self.drnn_layer(cell, x, dilation, hidden[i]) #######DRNN_LAYER YAAAAAAAAAAA
            
        outputs=x[-dilation:]
        x=self.sigmoid(self.out(torch.sum(outputs,0))) #change from F to self.sigmoid, should be the same
        return x.squeeze(), lt.squeeze()

        
######Dilated RNN related methods
    def drnn_layer(self, cell, inputs, rate, hidden=None):

        #n_steps = len(inputs)
        n_steps = inputs.size(0)
        #print('n_steps',n_steps) 
        #batch_size = inputs[0].size(0)
        batch_size = inputs.size(1)
        #print('batch size',batch_size) --verified correct
        hidden_size = cell.hidden_size
        #print('hidden size',hidden_size) --verified correct

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden
       
    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):

        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):

        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):

        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):

        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, Variable(zeros_)))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps


    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs
    
    def init_hidden(self, batch_size, hidden_size):
        c = Variable(torch.zeros(batch_size, hidden_size)) ############## hidden_dim??? hidden_size?? Also where is other batch_size 
        if use_cuda:
            c = c.cuda()
        if self.cell_type == "LSTM":
            m = Variable(torch.zeros(batch_size, hidden_size))  #batch_size should be part of self.batch_size I think 
            if use_cuda:
                m = m.cuda()
            return (c, m)
        else:
            return c        




# Model 3: QRNN
class EHR_QRNN(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers =1 ,dropout_r=0.1, cell_type='QRNN', bii=False, time=False, preTrainEmb='', packPadMode = False):

        EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers, dropout_r, cell_type, time , preTrainEmb, packPadMode)

        #super(EHR_QRNN, self).__init__()
        #basically, we dont allow cell_type and bii choices
        #let's enfroce these:
        if (self.cell_type !='QRNN' or self.bi !=1):
            print('QRNN only supports 1-direction & QRNN cell_type implementation. Implementing corrected parameters instead')
        self.cell_type = 'QRNN'
        self.bi = 1 #enforcing 1 directional
        self.packPadMode = False #enforcing correct packpaddedmode
        
    #embedding function goes here
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self,input)
    
    def EmbedPatient_SMB(self, input):
        return EHREmbeddings.EmbedPatients_SMB(self,input)    
    
    def forward(self, input):
        x_in , lt ,x_lens = self.EmbedPatient_MB(input)
        x_in = x_in.permute(1,0,2) ## QRNN not support batch first
        output, hidden = self.rnn_c(x_in)#,h_0) 
        output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze(), lt.squeeze()



# Model 4: T-LSTM
class EHR_TLSTM(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers =1 ,dropout_r=0.1, cell_type='TLSTM', bii=False, time=False, preTrainEmb=''):

        EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers, dropout_r, cell_type, time , preTrainEmb)

        #test on EHR_TLSTM() parameters please
        #modify something here to make sure everything runs correctly
        '''ask laila if i Implemented the right model parameters regarding, time, bii, and pretrained,
        '''
        if self.cell_type !='TLSTM' or self.bi != 1:
            print("TLSTM only supports 'TSTM' cell and 1 direction. Implementing corrected parameters instead")
        self.cell_type = 'TLSTM'
        self.bi = 1 #enforcing 1 directional

        
        #check on the packpadded sequence part and others
    def EmbedPatient_MB(self, input):
        return EHREmbeddings.EmbedPatients_MB(self, input)
    
    def EmbedPatient_SMB(self, input):
        return EHREmbeddings.EmbedPatients_SMB(self, input)       

  
    def init_hidden(self):
        h_0 = Variable(torch.rand(self.n_layers*self.bi,self.bsize, self.hidden_size))
        if use_cuda:
            h_0= h_0.cuda()
        if self.cell_type == "LSTM"or self.cell_type == "TLSTM":
            result = (h_0,h_0)
        else: 
            result = h_0
        return result
   
    
    def forward(self, input):
        if self.multi_emb: x_in , lt ,x_lens = self.EmbedPatient_SMB(input)
        else: x_in , lt ,x_lens = self.EmbedPatient_MB(input)
        x_in = x_in.permute(1,0,2) ##
        #x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)
        h_0 = self.init_hidden()
        output, hidden,_ = self.rnn_c(x_in,h_0) 
        if self.cell_type == "LSTM" or self.cell_type == "TLSTM":
            hidden=hidden[0]
        if self.bi==2:
            output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1)))
        else:
            output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze(), lt.squeeze()



# Model 5: Logistic regression (with embeddings) Do we want to keep it here?  
class EHR_LR_emb(nn.Module):
    def __init__(self,input_size, embed_dim= 128, preTrainEmb= ''):    
        super(EHR_LR_emb, self).__init__()
        self.embed_dim = embed_dim
        #self.embedding = nn.Embedding(input_size,embed_dim)
        self.out = nn.Linear(self.embed_dim,1)
        self.sigmoid = nn.Sigmoid()
        self.preTrainEmb=preTrainEmb
        #Need to modify here 
        if len(self.preTrainEmb) >0 :
           emb_t= torch.FloatTensor(np.asmatrix(self.preTrainEmb))
           self.embed_dim = emb_t.size(1)
           input_size = emb_t.size(0)
           self.embedding = nn.EmbeddingBag(emb_t.size(0),emb_t.size(1))
           self.embedding.weight.data= emb_t
           self.embedding.weight.requires_grad=False
         
        else:
           self.embedding = nn.Embedding(input_size, self.embed_dim) 
        

    def forward(self, input):  
        label, ehr_seq = input[0] 
        #print(input[0]) #real-time check
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

