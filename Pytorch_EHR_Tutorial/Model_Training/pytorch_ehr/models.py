# -*- coding: utf-8 -*-
"""
This Class is mainly for the creation of the EHR patients' visits embedding
which is the key input for all the deep learning models in this Repo
@authors: Lrasmy , Jzhu, Htran, Xin128 @ DeguiZhi Lab - UTHealth SBMI

V.3.0: adding survival and multiclass classification, as well as MLP options 
@authors: Lrasmy , BMao @ DeguiZhi Lab - UTHealth SBMI

Last revised Nov 08 2020
"""
import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
#from torchqrnn import QRNN



#from embedding import EHRembeddings 
from .EHREmb import EHREmbeddings

use_cuda = torch.cuda.is_available()

# Model 1:RNN & Variations: GRU, LSTM, Bi-RNN, Bi-GRU, Bi-LSTM
class EHR_RNN(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers=1,dropout_r=0.1,cell_type='GRU',bii=False ,time=False, preTrainEmb='',packPadMode = True, surv=False, hlf=False, cls_dim=1):

       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type, bii=bii, time=time , preTrainEmb=preTrainEmb, packPadMode=packPadMode, surv=surv,hlf=hlf, cls_dim=cls_dim)


    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    def init_hidden(self):
        
        h_0 = Variable(torch.rand(self.n_layers*self.bi,self.bsize, self.hidden_size))
        if use_cuda: 
            h_0.cuda()
        if self.cell_type == "LSTM":
            result = (h_0,h_0)
        else: 
            result = h_0
        return result
    
    def forward(self, input, x_lens, mtd):
        #print('input',input)
        x_in  = self.EmbedPatient_MB(input, mtd) 
        #print('x_in',x_in)
        ### uncomment the below lines if you like to initiate hidden to random instead of Zero which is the default
        #h_0= self.init_hidden()
        #if use_cuda: h_0.cuda()
        if self.packPadMode: 
            x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)  
            #print (x_inp)
            output, hidden = self.rnn_c(x_inp)#,h_0) 
        else:
            output, hidden = self.rnn_c(x_in)#,h_0) 
        
        if self.cell_type == "LSTM":
            hidden=hidden[0]
        if self.bi==2:
            if ((self.surv) & (self.cls_dim==1)) : 
                output = self.out(torch.cat((hidden[-2],hidden[-1]),1))
                #print('surv_bi')
            else: output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1))) ## if multiclass will be softmax
        
        else:
            if ((self.surv) & (self.cls_dim==1)) : 
                output = self.out(hidden[-1])
                #print('surv')  
            else: output = self.sigmoid(self.out(hidden[-1]))## if multiclass will be softmax
                
        return output.squeeze()

#Model 2: DRNN, DGRU, DLSTM
class EHR_DRNN(EHREmbeddings): 
    def __init__(self,input_size,embed_dim, hidden_size, n_layers, dropout_r=0.1,cell_type='GRU', bii=False, time=False, preTrainEmb='', packPadMode = False):

       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type,bii=False, time=time , preTrainEmb=preTrainEmb, packPadMode=False)


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
        #self.out = nn.Linear(hidden_size,1)
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
        
## Uncomment if using the mutiple embeddings version ---not included in this release    
#     def EmbedPatient_SMB(self, input,mtd):
#         return EHREmbeddings.EmbedPatients_SMB(self, input,mtd)     
    
    def forward(self, input,  x_lens, mtd, hidden=None):
        x  = self.EmbedPatient_MB(input,mtd) 

        x=x.permute(1,0,2)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                x,_ = self.drnn_layer(cell, x, dilation)
            else:
                x,hidden[i] = self.drnn_layer(cell, x, dilation, hidden[i]) 
            
        outputs=x[-dilation:]
        x=self.sigmoid(self.out(torch.sum(outputs,0))) #change from F to self.sigmoid, should be the same
        return x.squeeze()

        
######Dilated RNN related methods
    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = inputs.size(0)
        batch_size = inputs.size(1)
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
        c = Variable(torch.zeros(batch_size, hidden_size))
        if use_cuda:
            c = c.cuda()
        if self.cell_type == "LSTM":
            m = Variable(torch.zeros(batch_size, hidden_size))   
            if use_cuda:
                m = m.cuda()
            return (c, m)
        else:
            return c        


# Model 3: QRNN
class EHR_QRNN(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers =1 ,dropout_r=0.1, cell_type='QRNN', bii=False, time=False, preTrainEmb='', packPadMode = False):

       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type, bii=bii, time=time , preTrainEmb=preTrainEmb, packPadMode=packPadMode)

        #super(EHR_QRNN, self).__init__()
        #basically, we dont allow cell_type and bii choices
        #let's enfroce these:
        if (self.cell_type !='QRNN' or self.bi !=1):
            print('QRNN only supports 1-direction & QRNN cell_type implementation. Implementing corrected parameters instead')
        self.cell_type = 'QRNN'
        self.bi = 1 #enforcing 1 directional
        self.packPadMode = False #enforcing correct packpaddedmode
        
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    def forward(self, input, x_lens, mtd):    
        x_in  = self.EmbedPatient_MB(input,mtd) 
        x_in = x_in.permute(1,0,2) ## QRNN not support batch first
        output, hidden = self.rnn_c(x_in)#,h_0) 
        output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze()



# Model 4: T-LSTM
class EHR_TLSTM(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers =1 ,dropout_r=0.1, cell_type='TLSTM', bii=False, time=True, preTrainEmb=''):

        EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers, dropout_r, cell_type, time , preTrainEmb)
       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type, bii=False, time=True , preTrainEmb=preTrainEmb, packPadMode=False)
        
        if self.cell_type !='TLSTM' or self.bi != 1:
            print("TLSTM only supports Time aware LSTM cell type and 1 direction. Implementing corrected parameters instead")
        self.cell_type = 'TLSTM'
        self.bi = 1 #enforcing 1 directional
        self.packPadMode = False

        
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    def EmbedPatient_SMB(self, input,mtd):
        return EHREmbeddings.EmbedPatients_SMB(self, input, mtd)   

  
    def init_hidden(self):
        h_0 = Variable(torch.rand(self.n_layers*self.bi,self.bsize, self.hidden_size))
        if use_cuda:
            h_0= h_0.cuda()
        if self.cell_type == "LSTM"or self.cell_type == "TLSTM":
            result = (h_0,h_0)
        else: 
            result = h_0
        return result
   
    
    def forward(self, input, x_lens, mtd):
        x_in  = self.EmbedPatient_MB(input,mtd) 
        x_in = x_in.permute(1,0,2) 
        #x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)### not well tested
        h_0 = self.init_hidden()
        output, hidden,_ = self.rnn_c(x_in,h_0) 
        if self.cell_type == "LSTM" or self.cell_type == "TLSTM":
            hidden=hidden[0]
        if self.bi==2:
            output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1)))
        else:
            output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze()

# Model 5: Logistic regression (with embeddings):
class EHR_LR_emb(EHREmbeddings):
    def __init__(self, input_size,embed_dim, time=False, cell_type= 'LR',preTrainEmb='',surv=False):
        
         EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size = embed_dim,surv=surv)
         
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
## Uncomment if using the mutiple embeddings version ---not included in this release    
#     def EmbedPatient_SMB(self, input,mtd):
#         return EHREmbeddings.EmbedPatients_SMB(self, input,mtd)     
    def forward(self, input, x_lens, mtd):
#         if self.multi_emb:
#             x_in  = self.EmbedPatient_SMB(input,mtd)
#         else: 
        x_in  = self.EmbedPatient_MB(input,mtd) 
        if self.surv:
            output = self.out(torch.sum(x_in,1))
        else:    
            output = self.sigmoid(self.out(torch.sum(x_in,1)))
        return output.squeeze()

# Model 6: MLP (with embeddings):
class EHR_MLP(EHREmbeddings):
    def __init__(self, input_size,embed_dim,hidden_size, time=False, cell_type= 'LR',preTrainEmb='',surv=False):
        
         EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size = hidden_size,surv=surv)
         self.sequential=nn.Sequential( nn.Linear(self.in_size, self.hidden_size), nn.Sigmoid(), nn.Linear(self.hidden_size, self.hidden_size))
            
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
## Uncomment if using the mutiple embeddings version ---not included in this release    
#     def EmbedPatient_SMB(self, input,mtd):
#         return EHREmbeddings.EmbedPatients_SMB(self, input,mtd)     
    def forward(self, input, x_lens, mtd):
#         if self.multi_emb:
#             x_in  = self.EmbedPatient_SMB(input,mtd)
#         else: 
        x_in  = self.EmbedPatient_MB(input,mtd) 
        if self.surv:
            output = self.out(torch.sum(self.sequential(x_in),1))
        else:    
            output = self.sigmoid(self.out(torch.sum(self.sequential(x_in),1)))
        return output.squeeze()

    

# Model 6:Retain Model
class RETAIN(EHREmbeddings):
    def __init__(self, input_size, embed_dim, hidden_size, n_layers,bii=True):
        
        EHREmbeddings.__init__(self,input_size = input_size, embed_dim=embed_dim ,hidden_size=hidden_size)
        
        if bii: self.bi=2 
        else: self.bi=1
        
        self.embed_dim = embed_dim
        self.RNN1 = nn.RNN(embed_dim,hidden_size,1,batch_first=True,bidirectional=bii)
        self.RNN2 = nn.RNN(embed_dim,hidden_size,1,batch_first=True,bidirectional=bii)
        self.wa = nn.Linear(hidden_size*self.bi,1,bias=False)
        self.Wb = nn.Linear(hidden_size*self.bi,self.embed_dim,bias=False)
        self.W_out = nn.Linear(self.embed_dim,n_layers,bias=False)
        self.sigmoid = nn.Sigmoid()
        #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    
    def forward(self, input, x_lens, mtd):
        # get embedding using self.emb
        b = len(input)
        x_in  = self.EmbedPatient_MB(input,mtd) 
            
        h_0 = Variable(torch.rand(self.bi,self.bsize, self.hidden_size))
        if use_cuda:
            x_in = x_in.cuda()
            h_0 = h_0.cuda()
      
        # get alpha coefficients
        outputs1 = self.RNN1(x_in,h_0) # [b x seq x 128*2]
        b,seq,_ = outputs1[0].shape
        E = self.wa(outputs1[0].contiguous().view(-1, self.hidden_size*self.bi)) # [b*seq x 1]     
        alpha = F.softmax(E.view(b,seq),1) # [b x seq]
        self.alpha = alpha
         
        # get beta coefficients
        outputs2 = self.RNN2(x_in,h_0) # [b x seq x 128]
        b,seq,_ = outputs2[0].shape
        outputs2 = self.Wb(outputs2[0].contiguous().view(-1,self.hidden_size*self.bi)) # [b*seq x hid]
        self.Beta = torch.tanh(outputs2).view(b, seq, self.embed_dim) # [b x seq x 128]
        result = self.compute(x_in, self.Beta, alpha)
        return result.squeeze()

    # multiply to inputs
    def compute(self, embedded, Beta, alpha):
        b,seq,_ = embedded.size()
        outputs = (embedded*Beta)*alpha.unsqueeze(2).expand(b,seq,self.embed_dim)
        outputs = outputs.sum(1) # [b x hidden]
        return self.sigmoid(self.W_out(outputs)) # [b x num_classes]
    
    
    # interpret
    def interpret(self,u,v,i,o):
        # u: user number, v: visit number, i: input element number, o: output sickness
        a = self.alpha[u][v] # [1]
        B = self.Beta[u][v] # [h] embed dim
        W_emb = self.emb[i] # [h] embed)dim
        W = self.W_out.weight.squeeze() # [h]
        out = a*torch.dot(W,(B*W_emb))
        return out
        


