# -*- coding: utf-8 -*-

#latest codes for LR Dim2 as well as RNN from Laila 
#for 2D LR: extracting embedded matrix by default; return output, real_label tensor and embeded. 
#for 1D LR: modify embed_dim to 1 
import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

# Model 1: simple Logistic Regression(dimension = 2) 
class EHR_LR(nn.Module):
    def __init__(self,input_size, embed_dim = 2):    
        super(EHR_LR, self).__init__()
         
        #object parameters and attributes 
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(input_size,embed_dim)
        self.out = nn.Linear(self.embed_dim,1)
        self.sigmoid = nn.Sigmoid()
        #extracting initial weights, can be commentted out if not desired
        self.x_emb_m = self.embedding.weight
        
    def forward(self, input):  #lets say the input the one sample data of the merged set  
        label, ehr_seq = input[0] 
        label_tensor = Variable(torch.FloatTensor([[float(label)]]))
        if use_cuda:
            label_tensor = label_tensor.cuda()
        if use_cuda:
            result = Variable(torch.LongTensor([int(v) for v in ehr_seq])).cuda() 
        else:
            result = Variable(torch.LongTensor([int(v) for v in ehr_seq])) 
        embedded = self.embedding(result).view(-1, self.embed_dim) 
        embedded = torch.sum(embedded, dim=0).view(1,-1)
        output = self.sigmoid(self.out(embedded))
        return output, label_tensor,embedded #return output and also label tensor


# Model 2:RNN     
class EHR_RNN(nn.Module):
    def __init__(self, input_size, embed_dim , n_hidden, n_layers=1 , dropout_r=0.1 , cell_type='LSTM'):
        super(EHR_RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = n_hidden
        self.embed_dim = embed_dim
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.embedBag = nn.EmbeddingBag(input_size, self.embed_dim,mode= 'sum')
        
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError
        
        self.rnn_c = cell(self.embed_dim, self.hidden_size,num_layers=n_layers, dropout= dropout_r )
        
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
            
            zp= nn.ZeroPad2d((0,0,(lp-len(pt_visits)),0))
            xzp= zp(emb_bp)
            tb[pt]=xzp.data

        tb= tb.permute(1, 0, 2) ### as my final input need to be seq_len x batch_size x input_size
        emb_m=Variable(tb)
        label_tensor = Variable(lbt1)

        if use_cuda:
                label_tensor = label_tensor.cuda()
                emb_m = emb_m.cuda()
        #print (label_tensor)        
        return emb_m , label_tensor

    def forward(self, input):
        
        x_in , lt = self.EmbedPatient_MB(input)
        
        for i in range(self.n_layers):
                output, hidden = self.rnn_c(x_in) # input (seq_len, batch, input_size) need to check torch.nn.utils.rnn.pack_padded_sequence() 
                                                          
        output = self.sigmoid(self.out(output[-1]))
        #print (output, lt)
        return output, lt


#Model 2: DRNN
class DRNN(nn.Module):

    def __init__(self, input_size, embed_dim , n_hidden, n_layers=1 , dropout_r=0.1 , cell_type='LSTM'):
    

        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.D = embed_dim
        self.embedBag = nn.EmbeddingBag(input_size,self.D, mode= 'sum')
        self.cells = nn.ModuleList([])

        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(embed_dim, n_hidden, dropout=dropout_r)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout_r)
            self.cells.append(c)

        self.out = nn.Linear(n_hidden,1)

    

    def EmbedPatient_MB(self, seq_mini_batch): # x is a ehr_seq_tensor
        
       
        lp= len(max(seq_mini_batch, key=lambda xmb: len(xmb[1]))[1])
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
            
            #print ('embed bag Matrix : /n' , emb_bp )
            
            zp= nn.ZeroPad2d((0,0,0,(lp-len(pt_visits))))
            xzp= zp(emb_bp)

            #print ('padded embed bag Matrix : /n' , xzp )

            tb[pt]=xzp.data
        
        #print ('pts embed bag Matrix : /n' , tb )
        
        # input for RNN need to be seq_len, batch, input_size
        tb= tb.permute(1, 0, 2)  ### as my final input need to be seq_len x batch_size x input_size 
        
        #print (tb.size())

        #print ('Final input : /n' , tb )
        
        emb_m=Variable(tb)
        label_tensor = Variable(lbt1)

        if use_cuda:
                label_tensor = label_tensor.cuda()
                emb_m = emb_m.cuda()
        
        #print ('just for verificaton: /n Label tensor var: /n', label_tensor , 'input emb : /n', emb_m , 'input reformat done')
        return emb_m , label_tensor

    def forward(self, inputs, hidden=None):
        
        x , lt = self.EmbedPatient_MB(inputs)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                x = self.drnn_layer(cell, x, dilation)
            else:
                x = self.drnn_layer(cell, x, dilation, hidden[i])
            
            outputs.append(x[-dilation:])

        #x= F.sigmoid(F.max_pool1d(self.out(x)))
        #x = self.out(x).squeeze()
        #print ('x dim', x.size())
        x = F.sigmoid(F.max_pool1d(self.out(x).permute(2,1,0),x.size(0)))

        return x.squeeze(), lt.squeeze()

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
            dilated_outputs = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs
    
       
    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):

        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs = cell(dilated_inputs, hidden)[0]

        return dilated_outputs

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

    def init_hidden(self, batch_size, hidden_dim):
        c = Variable(torch.zeros(batch_size, hidden_dim))
        if use_cuda:
            c = c.cuda()
        if self.cell_type == "LSTM":
            m = Variable(torch.zeros(batch_size, hidden_dim))
            if use_cuda:
                m = m.cuda()
            return (c, m)
        else:
            return c
            
            
#Model 3: CNN_EHR            

class CNN_EHR(nn.Module):
   
    def __init__ (self, input_size, embed_dim, ch_out, kernel_size=2, dropout=0 , dilation_depth=1 , cnn_grp=1 ):
        
        # model.CNN_EHR(args.input_size, args.embed_dim, args.ch_out, args.k_size, args.dropout, args.n_layers) 
        # for Dilated CNN , n_layers represent the dilation depth use the ch_out=emb_dim
        # for depthwise CNN , use ch_out = embed_dim*kernel_size, groups= embed_dim)
        super(CNN_EHR, self).__init__()
        
        self.P = input_size ## number of predictors in our model
        self.D  = embed_dim
        self.K = kernel_size 
        self.dilations = [self.K**i for i in range(dilation_depth)] 
        self.cnn_grp = cnn_grp
        C = 1
        Ci = 1
        self.Co = ch_out

        self.embedBag = nn.EmbeddingBag(self.P , self.D,mode= 'sum') 
        self.convs = nn.ModuleList([])
        for d in self.dilations:
          if d==1:
              conv= nn.Conv1d(in_channels=self.D, out_channels=self.Co, kernel_size=self.K, dilation=d , padding = d , groups= self.cnn_grp )
          else: 
              conv= nn.Conv1d(in_channels=self.Co, out_channels=self.Co, kernel_size=self.K, dilation=d , padding = d , groups= self.cnn_grp )
              
          self.convs.append(conv)
          
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.Co, C)
        self.sigmoid = nn.Sigmoid()
   
    def EmbedPatient_MB(self, seq_mini_batch): # x is a ehr_seq_tensor
        
       
        lp= len(max(seq_mini_batch, key=lambda xmb: len(xmb[1]))[1]) 
        tb= torch.FloatTensor(len(seq_mini_batch),lp,self.D) 
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
        #x = self.EmbedPatient(input) # [seqlen*batchsize*embdim]

        x , lt = self.EmbedPatient_MB(input)
        x = x.permute(1,2,0) # [N, Co, W]
        
        for conv in self.convs:
            x = conv(x)
        x = F.relu(x)  # [(N, Co, W), ...]*len(Ks)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) 
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        y = self.sigmoid(logit)
        

        return y , lt
