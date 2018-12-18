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
from torchqrnn import QRNN
import bnlstm

use_cuda = torch.cuda.is_available()


# Model 1:RNN     
class EHR_RNN(nn.Module):
    def __init__(self, input_size,embed_dim, hidden_size, n_layers=1,dropout_r=0.1,cell_type='LSTM',bii=False ,time=False, preTrainEmb=''):
        super(EHR_RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.preTrainEmb=preTrainEmb
        self.time=time
        
        if bii: self.bi=2 
        else: self.bi=1
            
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
            if len(input_size)!=3: raise ValueError('the input list either of 1 or 3 length')
            else: 
                self.multi_emb=True
                self.diag=self.med=self.oth=1
            
            if input_size[0]> 0 : self.embed_d= nn.Embedding(input_size[0], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.diag=0
            if input_size[1]> 0 : self.embed_m= nn.Embedding(input_size[1], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.med=0
            if input_size[2]> 0 : self.embed_o= nn.Embedding(input_size[2], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.oth=0
            self.in_size=(self.diag+self.med+self.oth)*embed_dim
        
        if self.time: self.in_size= self.in_size+1 
               
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError
            
          
        self.rnn_c = cell(self.in_size, hidden_size,num_layers=n_layers, dropout= dropout_r , bidirectional=bii , batch_first=True)
        self.out = nn.Linear(self.hidden_size*self.bi,1)
        self.sigmoid = nn.Sigmoid()

        
    def EmbedPatient_MB(self, input): # x is a ehr_seq_tensor

        if use_cuda:
            flt_typ=torch.cuda.FloatTensor
            lnt_typ=torch.cuda.LongTensor
        else: 
            lnt_typ=torch.LongTensor
            flt_typ=torch.FloatTensor
        mb=[]
        mtd=[]
        lbt=[]
        seq_l=[]
        self.bsize=len(input) ## no of pts in minibatch
        lp= len(max(input, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch
        llv=0
        for x in input:
            lv= len(max(x[-1], key=lambda xmb: len(xmb[1]))[1])
            if llv < lv:
                llv=lv     # max number of codes per visit in minibatch        

        for pt in input:
            sk,label,ehr_seq_l = pt
            lpx=len(ehr_seq_l) ## no of visits in pt record
            seq_l.append(lpx) 
            lbt.append(Variable(flt_typ([[float(label)]])))### check if code issue replace back to the above
            ehr_seq_tl=[]
            time_dim=[]
            for ehr_seq in ehr_seq_l:
                pd=(0, (llv -len(ehr_seq[1])))
                result = F.pad(torch.from_numpy(np.asarray(ehr_seq[1],dtype=int)).type(lnt_typ),pd,"constant", 0)
                ehr_seq_tl.append(result)
                if self.time:                 
                    #time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                    # use log time as RETAIN
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))

            ehr_seq_t= Variable(torch.stack(ehr_seq_tl,0)) 
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits
            zp= nn.ZeroPad2d((0,0,0,lpp)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise
            ehr_seq_t= zp(ehr_seq_t) ## zero pad the visits med codes
            mb.append(ehr_seq_t)
            if self.time:
                time_dim_v= Variable(torch.stack(time_dim,0))
                time_dim_pv= zp(time_dim_v)## zero pad the visits time diff codes
                mtd.append(time_dim_pv)

            
        mb_t= Variable(torch.stack(mb,0)) 
        if use_cuda:
            mb_t.cuda()
        embedded = self.embed(mb_t)  ## Embedding for codes
        embedded = torch.sum(embedded, dim=2) 
        lbt_t= Variable(torch.stack(lbt,0))
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        return out_emb, lbt_t,seq_l #,dem_emb
    
    
    def EmbedPatient_SMB(self, input): ## splitted input

        if use_cuda:
            flt_typ=torch.cuda.FloatTensor
            lnt_typ=torch.cuda.LongTensor
        else: 
            lnt_typ=torch.LongTensor
            flt_typ=torch.FloatTensor
        mtd=[]
        lbt=[]
        seq_l=[]
        self.bsize=len(input) ## no of pts in minibatch
        lp= len(max(input, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch # this remains fine with whatever input format

        if self.diag==1: 
            mbd=[]
            llvd=0
        if self.med==1:
            mbm=[]
            llvm=0
        if self.oth==1:
            mbo=[]      
            llvo=0

        for x in input:
                if self.diag==1:
                    lvd= len(max(x[-1], key=lambda xmb: len(xmb[1][0]))[1][0])
                    if llvd < lvd:  llvd=lvd     # max number of diagnosis codes per visit in minibatch
                if self.med==1:
                    lvm= len(max(x[-1], key=lambda xmb: len(xmb[1][1]))[1][1])
                    if llvm < lvm:  llvm=lvm     # max number of medication codes per visit in minibatch 
                if self.oth==1:
                    lvo= len(max(x[-1], key=lambda xmb: len(xmb[1][2]))[1][2])
                    if llvo < lvo:  llvo=lvo     # max number of demographics and other codes per visit in minibatch                                     
        #print(llvd,llvm,llvo)

        for pt in input:
            sk,label,ehr_seq_l = pt
            lpx=len(ehr_seq_l) ## no of visits in pt record
            seq_l.append(lpx) 
            lbt.append(Variable(flt_typ([[float(label)]])))### check if code issue replace back to the above
            time_dim=[]        
            ehr_seq_tld=[]
            ehr_seq_tlm=[]
            ehr_seq_tlo=[]
          
            for ehr_seq in ehr_seq_l: 
                if self.diag==1: 
                    pdd=(0, (llvd -len(ehr_seq[1][0])))
                    if len(ehr_seq[1][0])==0: resultd = F.pad(lnt_typ([0]),(0,llvd-1),"constant", 0)    
                    else: resultd = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][0],dtype=int)).type(lnt_typ),pdd,"constant", 0)
                    ehr_seq_tld.append(resultd)
                if self.med==1: 
                    pdm=(0, (llvm -len(ehr_seq[1][1])))
                    if len(ehr_seq[1][1])==0: resultm = F.pad(lnt_typ([0]),(0,llvm-1),"constant", 0)     
                    else:resultm = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][1],dtype=int)).type(lnt_typ),pdm,"constant", 0)
                    ehr_seq_tlm.append(resultm)
                if self.oth==1: 
                    pdo=(0, (llvo -len(ehr_seq[1][2])))
                    if len(ehr_seq[1][2])==0: resulto = F.pad(lnt_typ([0]),(0,llvo-1),"constant", 0)     
                    else: resulto = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][2],dtype=int)).type(lnt_typ),pdo,"constant", 0)
                    ehr_seq_tlo.append(resulto)
                
                if self.time:                 
                    #time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                    # use log time as RETAIN
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))
                    
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits
            zp= nn.ZeroPad2d((0,0,0,lpp)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise
  
            if self.diag==1:
                ehr_seq_td= Variable(torch.stack(ehr_seq_tld,0))
                ehr_seq_td= zp(ehr_seq_td) ## zero pad the visits diag codes
                mbd.append(ehr_seq_td)
            if self.med==1: 
                ehr_seq_tm= Variable(torch.stack(ehr_seq_tlm,0)) 
                ehr_seq_tm= zp(ehr_seq_tm) ## zero pad the visits med codes
                mbm.append(ehr_seq_tm)
            if self.oth==1: 
                ehr_seq_to= Variable(torch.stack(ehr_seq_tlo,0)) 
                ehr_seq_to= zp(ehr_seq_to) ## zero pad the visits dem&other codes
                mbo.append(ehr_seq_to)
            
            if self.time:
                time_dim_v= Variable(torch.stack(time_dim,0))
                time_dim_pv= zp(time_dim_v)## zero pad the visits time diff codes
                mtd.append(time_dim_pv)

            
        #mb_t= Variable(torch.stack(mb,0)) 
        #if use_cuda:
        #    mb_t.cuda()
        #embedded = self.embed(mb_t)  ## Embedding for codes
        #embedded = torch.sum(embedded, dim=2) #### split all the above
        
        if self.diag==1:
            mb_td= Variable(torch.stack(mbd,0))
            if use_cuda: mb_td.cuda()
            embedded_d = torch.sum(self.embed_d(mb_td), dim=2)
            embedded= embedded_d 
        if self.med==1: 
            mb_tm= Variable(torch.stack(mbm,0))
            if use_cuda: mb_tm.cuda()  
            embedded_m = torch.sum(self.embed_m(mb_tm), dim=2)
            if self.diag==1: embedded=torch.cat((embedded,embedded_m),dim=2)
            else: embedded=embedded_m 
                
        if self.oth==1: 
            mb_to= Variable(torch.stack(mbo,0))
            if use_cuda: mb_to.cuda()
            embedded_o = torch.sum(self.embed_o(mb_to), dim=2)
            if self.diag + self.med > 0 : embedded=torch.cat((embedded,embedded_o),dim=2)
            else: embedded=embedded_o 

        #embedded=torch.cat((embedded_d,embedded_m,embedded_o),dim=2)## the concatination of above
        
        lbt_t= Variable(torch.stack(lbt,0))
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        return out_emb, lbt_t,seq_l #,dem_emb
 
    
    
    
    
    def init_hidden(self):
        
        h_0 = Variable(torch.rand(self.n_layers*self.bi,self.bsize, self.hidden_size))
        
        if use_cuda: h_0.cuda()

        
        if self.cell_type == "LSTM":
            result = (h_0,h_0)
        else: 
            result = h_0
            
        
        return result
    
    def forward(self, input):
        if self.multi_emb: x_in , lt ,x_lens = self.EmbedPatient_SMB(input)
        else: x_in , lt ,x_lens = self.EmbedPatient_MB(input)
        x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)   
        h_0= self.init_hidden()
        output, hidden = self.rnn_c(x_inp)#,h_0) 
        if self.cell_type == "LSTM":
            hidden=hidden[0]
        if self.bi==2:
            output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1)))
        else:
            output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze(), lt.squeeze()


#Model 2: DRNN
class DRNN(nn.Module):

    def __init__(self,input_size, embed_dim, n_hidden,  n_layers, dropout_r=0, cell_type='GRU',time=False,preTrainEmb=''):

        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.D = embed_dim
        self.time=time
        self.preTrainEmb=preTrainEmb
        if len(input_size)==1:
            self.multi_emb=False
            if len(self.preTrainEmb)>0:
                emb_t= torch.FloatTensor(np.asmatrix(self.preTrainEmb))
                self.embed= nn.Embedding.from_pretrained(emb_t)#,freeze=False) 
                self.in_size= embed_dim ### need to be updated to be automatically capyured from the input
            else:
                input_size=input_size[0]
                self.embed= nn.Embedding(input_size, self.D,padding_idx=0)#,scale_grad_by_freq=True)
                self.in_size= embed_dim
        else:
            if len(input_size)!=3: raise ValueError('the input list either of 1 or 3 length')
            else: 
                self.multi_emb=True
                self.diag=self.med=self.oth=1
            
            if input_size[0]> 0 : self.embed_d= nn.Embedding(input_size[0], self.D,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.diag=0
            if input_size[1]> 0 : self.embed_m= nn.Embedding(input_size[1], self.D,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.med=0
            if input_size[2]> 0 : self.embed_o= nn.Embedding(input_size[2], self.D,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.oth=0
            self.in_size=(self.diag+self.med+self.oth)*self.D
        
        if self.time: self.in_size= self.in_size+1 
        
        
        
        
        self.layers = nn.ModuleList([])

        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        elif self.cell_type == "QRNN":
            from torchqrnn import QRNN
            cell = QRNN
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(self.in_size, n_hidden, dropout=dropout_r)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout_r)
            self.layers.append(c)
        
        self.cells = nn.Sequential(*self.layers)
        
        self.out = nn.Linear(n_hidden,1)

    

    def EmbedPatient_MB(self, input): # x is a ehr_seq_tensor

        if use_cuda:
            flt_typ=torch.cuda.FloatTensor
            lnt_typ=torch.cuda.LongTensor
        else: 
            lnt_typ=torch.LongTensor
            flt_typ=torch.FloatTensor
        mb=[]
        mtd=[]
        lbt=[]
        seq_l=[]
        self.bsize=len(input) ## no of pts in minibatch
        lp= len(max(input, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch
        llv=0
        for x in input:
            lv= len(max(x[-1], key=lambda xmb: len(xmb[1]))[1])
            if llv < lv:
                llv=lv     # max number of codes per visit in minibatch        

        for pt in input:
            sk,label,ehr_seq_l = pt
            lpx=len(ehr_seq_l) ## no of visits in pt record
            seq_l.append(lpx) 
            lbt.append(Variable(flt_typ([[float(label)]])))### check if code issue replace back to the above
            ehr_seq_tl=[]
            time_dim=[]
            for ehr_seq in ehr_seq_l:
                pd=(0, (llv -len(ehr_seq[1])))
                result = F.pad(torch.from_numpy(np.asarray(ehr_seq[1],dtype=int)).type(lnt_typ),pd,"constant", 0)
                ehr_seq_tl.append(result)
                if self.time:                 
                    #time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                    # use log time as RETAIN
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))

            ehr_seq_t= Variable(torch.stack(ehr_seq_tl,0)) 
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits
            zp= nn.ZeroPad2d((0,0,lpp,0)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise
            ehr_seq_t= zp(ehr_seq_t) ## zero pad the visits med codes
            mb.append(ehr_seq_t)
            if self.time:
                time_dim_v= Variable(torch.stack(time_dim,0))
                time_dim_pv= zp(time_dim_v)## zero pad the visits time diff codes
                mtd.append(time_dim_pv)

            
        mb_t= Variable(torch.stack(mb,0)) 
        if use_cuda:
            mb_t.cuda()
        embedded = self.embed(mb_t)  ## Embedding for codes
        embedded = torch.sum(embedded, dim=2) 
        lbt_t= Variable(torch.stack(lbt,0))
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        return out_emb, lbt_t,seq_l #,dem_emb
    
    
    def EmbedPatient_SMB(self, input): ## splitted input

        if use_cuda:
            flt_typ=torch.cuda.FloatTensor
            lnt_typ=torch.cuda.LongTensor
        else: 
            lnt_typ=torch.LongTensor
            flt_typ=torch.FloatTensor
        mtd=[]
        lbt=[]
        seq_l=[]
        self.bsize=len(input) ## no of pts in minibatch
        lp= len(max(input, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch # this remains fine with whatever input format

        if self.diag==1: 
            mbd=[]
            llvd=0
        if self.med==1:
            mbm=[]
            llvm=0
        if self.oth==1:
            mbo=[]      
            llvo=0

        for x in input:
                if self.diag==1:
                    lvd= len(max(x[-1], key=lambda xmb: len(xmb[1][0]))[1][0])
                    if llvd < lvd:  llvd=lvd     # max number of diagnosis codes per visit in minibatch
                if self.med==1:
                    lvm= len(max(x[-1], key=lambda xmb: len(xmb[1][1]))[1][1])
                    if llvm < lvm:  llvm=lvm     # max number of medication codes per visit in minibatch 
                if self.oth==1:
                    lvo= len(max(x[-1], key=lambda xmb: len(xmb[1][2]))[1][2])
                    if llvo < lvo:  llvo=lvo     # max number of demographics and other codes per visit in minibatch                                     
        #print(llvd,llvm,llvo)

        for pt in input:
            sk,label,ehr_seq_l = pt
            lpx=len(ehr_seq_l) ## no of visits in pt record
            seq_l.append(lpx) 
            lbt.append(Variable(flt_typ([[float(label)]])))### check if code issue replace back to the above
            time_dim=[]        
            ehr_seq_tld=[]
            ehr_seq_tlm=[]
            ehr_seq_tlo=[]
          
            for ehr_seq in ehr_seq_l: 
                if self.diag==1: 
                    pdd=(0, (llvd -len(ehr_seq[1][0])))
                    if len(ehr_seq[1][0])==0: resultd = F.pad(lnt_typ([0]),(0,llvd-1),"constant", 0)    
                    else: resultd = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][0],dtype=int)).type(lnt_typ),pdd,"constant", 0)
                    ehr_seq_tld.append(resultd)
                if self.med==1: 
                    pdm=(0, (llvm -len(ehr_seq[1][1])))
                    if len(ehr_seq[1][1])==0: resultm = F.pad(lnt_typ([0]),(0,llvm-1),"constant", 0)     
                    else:resultm = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][1],dtype=int)).type(lnt_typ),pdm,"constant", 0)
                    ehr_seq_tlm.append(resultm)
                if self.oth==1: 
                    pdo=(0, (llvo -len(ehr_seq[1][2])))
                    if len(ehr_seq[1][2])==0: resulto = F.pad(lnt_typ([0]),(0,llvo-1),"constant", 0)     
                    else: resulto = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][2],dtype=int)).type(lnt_typ),pdo,"constant", 0)
                    ehr_seq_tlo.append(resulto)
                
                if self.time:                 
                    #time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                    # use log time as RETAIN
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))
                    
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits
            zp= nn.ZeroPad2d((0,0,lpp,0)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise
  
            if self.diag==1:
                ehr_seq_td= Variable(torch.stack(ehr_seq_tld,0))
                ehr_seq_td= zp(ehr_seq_td) ## zero pad the visits diag codes
                mbd.append(ehr_seq_td)
            if self.med==1: 
                ehr_seq_tm= Variable(torch.stack(ehr_seq_tlm,0)) 
                ehr_seq_tm= zp(ehr_seq_tm) ## zero pad the visits med codes
                mbm.append(ehr_seq_tm)
            if self.oth==1: 
                ehr_seq_to= Variable(torch.stack(ehr_seq_tlo,0)) 
                ehr_seq_to= zp(ehr_seq_to) ## zero pad the visits dem&other codes
                mbo.append(ehr_seq_to)
            
            if self.time:
                time_dim_v= Variable(torch.stack(time_dim,0))
                time_dim_pv= zp(time_dim_v)## zero pad the visits time diff codes
                mtd.append(time_dim_pv)

            
        #mb_t= Variable(torch.stack(mb,0)) 
        #if use_cuda:
        #    mb_t.cuda()
        #embedded = self.embed(mb_t)  ## Embedding for codes
        #embedded = torch.sum(embedded, dim=2) #### split all the above
        
        if self.diag==1:
            mb_td= Variable(torch.stack(mbd,0))
            if use_cuda: mb_td.cuda()
            embedded_d = torch.sum(self.embed_d(mb_td), dim=2)
            embedded= embedded_d 
        if self.med==1: 
            mb_tm= Variable(torch.stack(mbm,0))
            if use_cuda: mb_tm.cuda()  
            embedded_m = torch.sum(self.embed_m(mb_tm), dim=2)
            if self.diag==1: embedded=torch.cat((embedded,embedded_m),dim=2)
            else: embedded=embedded_m 
                
        if self.oth==1: 
            mb_to= Variable(torch.stack(mbo,0))
            if use_cuda: mb_to.cuda()
            embedded_o = torch.sum(self.embed_o(mb_to), dim=2)
            if self.diag + self.med > 0 : embedded=torch.cat((embedded,embedded_o),dim=2)
            else: embedded=embedded_o 

        #embedded=torch.cat((embedded_d,embedded_m,embedded_o),dim=2)## the concatination of above
        
        lbt_t= Variable(torch.stack(lbt,0))
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        return out_emb, lbt_t,seq_l #,dem_emb

    
    def forward(self, inputs, hidden=None):
        if self.multi_emb: x , lt ,_ = self.EmbedPatient_SMB(inputs)
        else: x , lt ,_ = self.EmbedPatient_MB(inputs)
 
        #x , lt , _ = self.EmbedPatient_MB(inputs)
        x=x.permute(1,0,2)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                x,_ = self.drnn_layer(cell, x, dilation)
            else:
                x,hidden[i] = self.drnn_layer(cell, x, dilation, hidden[i])
            
            #outputs.append(x[-dilation:])
        outputs=x[-dilation:]
        #x= F.sigmoid(F.max_pool1d(self.out(x)))
        #x = self.out(x).squeeze()
        #print ('x dim', x.size())
        #x = F.sigmoid(F.max_pool1d(self.out(x).permute(2,1,0),x.size(0)))
        x=F.sigmoid(self.out(torch.sum(outputs,0)))
        
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

# Model 4: QRNN
class EHR_QRNN(nn.Module):
    def __init__(self, input_size, embed_dim,hidden_size, n_layers=1,dropout_r=0.1,cell_type='QRNN',bii=False,time=False , preTrainEmb=''):
        super(EHR_QRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.preTrainEmb=preTrainEmb
        self.time=time
        if bii: self.bi=2 
        else: self.bi=1
                        
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
            if len(input_size)!=3: raise ValueError('the input list either of 1 or 3 length')
            else: 
                self.multi_emb=True
                self.diag=self.med=self.oth=1
            
            if input_size[0]> 0 : self.embed_d= nn.Embedding(input_size[0], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.diag=0
            if input_size[1]> 0 : self.embed_m= nn.Embedding(input_size[1], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.med=0
            if input_size[2]> 0 : self.embed_o= nn.Embedding(input_size[2], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.oth=0
            self.in_size=(self.diag+self.med+self.oth)*embed_dim
        
        if self.time: self.in_size= self.in_size+1 

        
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        elif self.cell_type == "BNLSTM":
            cell = bnlstm.LSTM    
        elif self.cell_type == "QRNN":
            cell = QRNN  
        else:
            raise NotImplementedError
      
        if self.cell_type == "BNLSTM":
            self.rnn_c = cell(bnlstm.BNLSTMCell, self.in_size, hidden_size,num_layers=n_layers,use_bias=False, dropout= dropout_r,max_length=30)
        elif self.cell_type == "QRNN":
            self.bi=1 ### QRNN not support Bidirectional
            self.rnn_c = cell(self.in_size, hidden_size,num_layers=n_layers, dropout= dropout_r)
        else:
            self.rnn_c = cell(self.in_size, hidden_size,num_layers=n_layers, dropout= dropout_r , bidirectional=bi  )
        
        self.out = nn.Linear(self.hidden_size*self.bi,1)
        self.sigmoid = nn.Sigmoid()
        
    def EmbedPatient_MB(self, input): # x is a ehr_seq_tensor

        if use_cuda:
            flt_typ=torch.cuda.FloatTensor
            lnt_typ=torch.cuda.LongTensor
        else: 
            lnt_typ=torch.LongTensor
            flt_typ=torch.FloatTensor
        mb=[]
        mtd=[]
        lbt=[]
        seq_l=[]
        self.bsize=len(input) ## no of pts in minibatch
        lp= len(max(input, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch
        llv=0
        for x in input:
            lv= len(max(x[-1], key=lambda xmb: len(xmb[1]))[1])
            if llv < lv:
                llv=lv     # max number of codes per visit in minibatch        

        for pt in input:
            sk,label,ehr_seq_l = pt
            lpx=len(ehr_seq_l) ## no of visits in pt record
            seq_l.append(lpx) 
            lbt.append(Variable(flt_typ([[float(label)]])))### check if code issue replace back to the above
            ehr_seq_tl=[]
            time_dim=[]
            for ehr_seq in ehr_seq_l:
                pd=(0, (llv -len(ehr_seq[1])))
                result = F.pad(torch.from_numpy(np.asarray(ehr_seq[1],dtype=int)).type(lnt_typ),pd,"constant", 0)
                ehr_seq_tl.append(result)
                if self.time:                 
                    #time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                    # use log time as RETAIN
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))

            ehr_seq_t= Variable(torch.stack(ehr_seq_tl,0)) 
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits
            zp= nn.ZeroPad2d((0,0,lpp,0)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise
            ehr_seq_t= zp(ehr_seq_t) ## zero pad the visits med codes
            mb.append(ehr_seq_t)
            if self.time:
                time_dim_v= Variable(torch.stack(time_dim,0))
                time_dim_pv= zp(time_dim_v)## zero pad the visits time diff codes
                mtd.append(time_dim_pv)

            
        mb_t= Variable(torch.stack(mb,0)) 
        if use_cuda:
            mb_t.cuda()
        embedded = self.embed(mb_t)  ## Embedding for codes
        embedded = torch.sum(embedded, dim=2) 
        lbt_t= Variable(torch.stack(lbt,0))
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        return out_emb, lbt_t,seq_l #,dem_emb
    
    
    def EmbedPatient_SMB(self, input): ## splitted input

        if use_cuda:
            flt_typ=torch.cuda.FloatTensor
            lnt_typ=torch.cuda.LongTensor
        else: 
            lnt_typ=torch.LongTensor
            flt_typ=torch.FloatTensor
        mtd=[]
        lbt=[]
        seq_l=[]
        self.bsize=len(input) ## no of pts in minibatch
        lp= len(max(input, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch # this remains fine with whatever input format

        if self.diag==1: 
            mbd=[]
            llvd=0
        if self.med==1:
            mbm=[]
            llvm=0
        if self.oth==1:
            mbo=[]      
            llvo=0

        for x in input:
                if self.diag==1:
                    lvd= len(max(x[-1], key=lambda xmb: len(xmb[1][0]))[1][0])
                    if llvd < lvd:  llvd=lvd     # max number of diagnosis codes per visit in minibatch
                if self.med==1:
                    lvm= len(max(x[-1], key=lambda xmb: len(xmb[1][1]))[1][1])
                    if llvm < lvm:  llvm=lvm     # max number of medication codes per visit in minibatch 
                if self.oth==1:
                    lvo= len(max(x[-1], key=lambda xmb: len(xmb[1][2]))[1][2])
                    if llvo < lvo:  llvo=lvo     # max number of demographics and other codes per visit in minibatch                                     
        #print(llvd,llvm,llvo)

        for pt in input:
            sk,label,ehr_seq_l = pt
            lpx=len(ehr_seq_l) ## no of visits in pt record
            seq_l.append(lpx) 
            lbt.append(Variable(flt_typ([[float(label)]])))### check if code issue replace back to the above
            time_dim=[]        
            ehr_seq_tld=[]
            ehr_seq_tlm=[]
            ehr_seq_tlo=[]
          
            for ehr_seq in ehr_seq_l: 
                if self.diag==1: 
                    pdd=(0, (llvd -len(ehr_seq[1][0])))
                    if len(ehr_seq[1][0])==0: resultd = F.pad(lnt_typ([0]),(0,llvd-1),"constant", 0)    
                    else: resultd = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][0],dtype=int)).type(lnt_typ),pdd,"constant", 0)
                    ehr_seq_tld.append(resultd)
                if self.med==1: 
                    pdm=(0, (llvm -len(ehr_seq[1][1])))
                    if len(ehr_seq[1][1])==0: resultm = F.pad(lnt_typ([0]),(0,llvm-1),"constant", 0)     
                    else:resultm = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][1],dtype=int)).type(lnt_typ),pdm,"constant", 0)
                    ehr_seq_tlm.append(resultm)
                if self.oth==1: 
                    pdo=(0, (llvo -len(ehr_seq[1][2])))
                    if len(ehr_seq[1][2])==0: resulto = F.pad(lnt_typ([0]),(0,llvo-1),"constant", 0)     
                    else: resulto = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][2],dtype=int)).type(lnt_typ),pdo,"constant", 0)
                    ehr_seq_tlo.append(resulto)
                
                if self.time:                 
                    #time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                    # use log time as RETAIN
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))
                    
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits
            zp= nn.ZeroPad2d((0,0,lpp,0)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise
  
            if self.diag==1:
                ehr_seq_td= Variable(torch.stack(ehr_seq_tld,0))
                ehr_seq_td= zp(ehr_seq_td) ## zero pad the visits diag codes
                mbd.append(ehr_seq_td)
            if self.med==1: 
                ehr_seq_tm= Variable(torch.stack(ehr_seq_tlm,0)) 
                ehr_seq_tm= zp(ehr_seq_tm) ## zero pad the visits med codes
                mbm.append(ehr_seq_tm)
            if self.oth==1: 
                ehr_seq_to= Variable(torch.stack(ehr_seq_tlo,0)) 
                ehr_seq_to= zp(ehr_seq_to) ## zero pad the visits dem&other codes
                mbo.append(ehr_seq_to)
            
            if self.time:
                time_dim_v= Variable(torch.stack(time_dim,0))
                time_dim_pv= zp(time_dim_v)## zero pad the visits time diff codes
                mtd.append(time_dim_pv)

            
        #mb_t= Variable(torch.stack(mb,0)) 
        #if use_cuda:
        #    mb_t.cuda()
        #embedded = self.embed(mb_t)  ## Embedding for codes
        #embedded = torch.sum(embedded, dim=2) #### split all the above
        
        if self.diag==1:
            mb_td= Variable(torch.stack(mbd,0))
            if use_cuda: mb_td.cuda()
            embedded_d = torch.sum(self.embed_d(mb_td), dim=2)
            embedded= embedded_d 
        if self.med==1: 
            mb_tm= Variable(torch.stack(mbm,0))
            if use_cuda: mb_tm.cuda()  
            embedded_m = torch.sum(self.embed_m(mb_tm), dim=2)
            if self.diag==1: embedded=torch.cat((embedded,embedded_m),dim=2)
            else: embedded=embedded_m 
                
        if self.oth==1: 
            mb_to= Variable(torch.stack(mbo,0))
            if use_cuda: mb_to.cuda()
            embedded_o = torch.sum(self.embed_o(mb_to), dim=2)
            if self.diag + self.med > 0 : embedded=torch.cat((embedded,embedded_o),dim=2)
            else: embedded=embedded_o 

        #embedded=torch.cat((embedded_d,embedded_m,embedded_o),dim=2)## the concatination of above
        
        lbt_t= Variable(torch.stack(lbt,0))
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        return out_emb, lbt_t,seq_l #,dem_emb


    
    def forward(self, input):
        
        if self.multi_emb: x_in , lt ,x_lens = self.EmbedPatient_SMB(input)
        else: x_in , lt ,x_lens = self.EmbedPatient_MB(input)

        x_in = x_in.permute(1,0,2) ## QRNN not support batch first
        output, hidden = self.rnn_c(x_in)#,h_0) 
        if self.cell_type == "LSTM":
            hidden=hidden[0]
        if self.bi==2:
            output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1)))
        else:
            output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze(), lt.squeeze()

##### T-lstm
import tplstm 
from tplstm import TPLSTM

class EHR_TLSTM(nn.Module):
    def __init__(self, input_size,embed_dim, hidden_size, n_layers=1,dropout_r=0.1,cell_type='TLSTM'):#,bi=False , preTrainEmb=''):
        super(EHR_TLSTM,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.preTrainEmb=preTrainEmb=''
        bi=False
        if bi: self.bi=2 
        else: self.bi=1
        self.time= False
        
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
            if len(input_size)!=3: raise ValueError('the input list either of 1 or 3 length')
            else: 
                self.multi_emb=True
                self.diag=self.med=self.oth=1
            
            if input_size[0]> 0 : self.embed_d= nn.Embedding(input_size[0], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.diag=0
            if input_size[1]> 0 : self.embed_m= nn.Embedding(input_size[1], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.med=0
            if input_size[2]> 0 : self.embed_o= nn.Embedding(input_size[2], self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
            else: self.oth=0
            self.in_size=(self.diag+self.med+self.oth)*embed_dim
        
        if self.time: self.in_size= self.in_size+1 

        
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        elif self.cell_type == "BNLSTM":
            cell = bnlstm.LSTM    
        elif self.cell_type == "TLSTM":
            cell = TPLSTM 
        else:
            raise NotImplementedError
      
        if self.cell_type == "BNLSTM":
            self.rnn_c = cell(bnlstm.BNLSTMCell, self.in_size, hidden_size,num_layers=n_layers,use_bias=False, dropout= dropout_r,max_length=30)
        elif self.cell_type == "TLSTM":
            self.bi=1 
            #self.rnn_c = cell(self.embed_dim, 1, hidden_size, hidden_size/2)
            self.rnn_c = cell(self.in_size, hidden_size)

        else:
            self.rnn_c = cell(self.in_size, hidden_size,num_layers=n_layers, dropout= dropout_r , bidirectional=bi  )
        
        self.out = nn.Linear(self.hidden_size*self.bi,1)
        self.sigmoid = nn.Sigmoid()

        
    def EmbedPatient_MB(self, input): # x is a ehr_seq_tensor
        
        mb=[]
        mtd=[]
        lbt=[]
        seq_l=[]
        self.bsize=len(input)
        lp= len(max(input, key=lambda xmb: len(xmb[-1]))[-1])
        #print (max(input, key=lambda xmb: len(xmb[-1])),lp) #verified
        llv=0
        for x in input:
            lv= len(max(x[-1], key=lambda xmb: len(xmb[1]))[1])
            #print(max(x[-1], key=lambda xmb: len(xmb[1:])),lv) #verified  
            if llv < lv:
                llv=lv             
        #print (llv)
        for pt in input:
            sk,label,ehr_seq_l = pt
            lpx=len(ehr_seq_l)
            seq_l.append(lpx)
            label_tensor = Variable(torch.FloatTensor([[float(label)]]))
            if use_cuda:
                label_tensor = label_tensor.cuda()
            lbt.append(label_tensor)
            if use_cuda:
                    flt_typ=torch.cuda.FloatTensor
                    lnt_typ=torch.cuda.LongTensor
            else: 
                lnt_typ=torch.LongTensor
                flt_typ=torch.FloatTensor
            ml=(len(max(ehr_seq_l, key=len)))
            ehr_seq_tl=[]
            time_dim=[]
            for ehr_seq in ehr_seq_l:
                #print (ehr_seq,ehr_seq[1])#verified
                #print(n_ehr_seq)
                pd=(0, (llv -len(ehr_seq[1])))
                time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                result = F.pad(torch.from_numpy(np.asarray(ehr_seq[1],dtype=int)).type(lnt_typ),pd,"constant", 0)
                ehr_seq_tl.append(result)
            ehr_seq_t= Variable(torch.stack(ehr_seq_tl,0)) 
            time_dim_v= Variable(torch.stack(time_dim,0))
            lpp= lp-lpx
            zp= nn.ZeroPad2d((0,0,0,lpp))
            ehr_seq_t= zp(ehr_seq_t)
            time_dim_pv= zp(time_dim_v)
            mb.append(ehr_seq_t)
            mtd.append(time_dim_pv)
            #print('ehr_seq_t',ehr_seq_t) #verified
            
        mb_t= Variable(torch.stack(mb,0)) 
        mtd_t= Variable(torch.stack(mtd,0))
        if use_cuda:
            mb_t.cuda()
            mtd_t.cuda()
        embedded = self.embed(mb_t)
        #print(mb_t,embedded) #verified
        embedded = torch.sum(embedded, dim=2) 
        lbt_t= Variable(torch.stack(lbt,0))
        #dem_t= Variable(torch.stack(demt,0))
        #if use_cuda: dem_t.cuda()
        #dem_emb=self.embed(dem_t)
        #dem_emb = torch.sum(dem_emb, dim=1) 
        #print ('embedded',embedded.shape,embedded,'time_dim_pv',mtd_t.shape,mtd_t)
        out_emb= torch.cat((embedded,mtd_t),dim=2)
        #print ('out_emb with time',out_emb.shape,out_emb)
        return out_emb, lbt_t,seq_l #,dem_emb
    
    
    def EmbedPatient_SMB(self, input): ## splitted input ###### This one need to be revised against the one above

        if use_cuda:
            flt_typ=torch.cuda.FloatTensor
            lnt_typ=torch.cuda.LongTensor
        else: 
            lnt_typ=torch.LongTensor
            flt_typ=torch.FloatTensor
        mtd=[]
        lbt=[]
        seq_l=[]
        self.bsize=len(input) ## no of pts in minibatch
        lp= len(max(input, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch # this remains fine with whatever input format

        if self.diag==1: 
            mbd=[]
            llvd=0
        if self.med==1:
            mbm=[]
            llvm=0
        if self.oth==1:
            mbo=[]      
            llvo=0

        for x in input:
                if self.diag==1:
                    lvd= len(max(x[-1], key=lambda xmb: len(xmb[1][0]))[1][0])
                    if llvd < lvd:  llvd=lvd     # max number of diagnosis codes per visit in minibatch
                if self.med==1:
                    lvm= len(max(x[-1], key=lambda xmb: len(xmb[1][1]))[1][1])
                    if llvm < lvm:  llvm=lvm     # max number of medication codes per visit in minibatch 
                if self.oth==1:
                    lvo= len(max(x[-1], key=lambda xmb: len(xmb[1][2]))[1][2])
                    if llvo < lvo:  llvo=lvo     # max number of demographics and other codes per visit in minibatch                                     
        #print(llvd,llvm,llvo)

        for pt in input:
            sk,label,ehr_seq_l = pt
            lpx=len(ehr_seq_l) ## no of visits in pt record
            seq_l.append(lpx) 
            lbt.append(Variable(flt_typ([[float(label)]])))### check if code issue replace back to the above
            time_dim=[]        
            ehr_seq_tld=[]
            ehr_seq_tlm=[]
            ehr_seq_tlo=[]
          
            for ehr_seq in ehr_seq_l: 
                if self.diag==1: 
                    pdd=(0, (llvd -len(ehr_seq[1][0])))
                    if len(ehr_seq[1][0])==0: resultd = F.pad(lnt_typ([0]),(0,llvd-1),"constant", 0)    
                    else: resultd = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][0],dtype=int)).type(lnt_typ),pdd,"constant", 0)
                    ehr_seq_tld.append(resultd)
                if self.med==1: 
                    pdm=(0, (llvm -len(ehr_seq[1][1])))
                    if len(ehr_seq[1][1])==0: resultm = F.pad(lnt_typ([0]),(0,llvm-1),"constant", 0)     
                    else:resultm = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][1],dtype=int)).type(lnt_typ),pdm,"constant", 0)
                    ehr_seq_tlm.append(resultm)
                if self.oth==1: 
                    pdo=(0, (llvo -len(ehr_seq[1][2])))
                    if len(ehr_seq[1][2])==0: resulto = F.pad(lnt_typ([0]),(0,llvo-1),"constant", 0)     
                    else: resulto = F.pad(torch.from_numpy(np.asarray(ehr_seq[1][2],dtype=int)).type(lnt_typ),pdo,"constant", 0)
                    ehr_seq_tlo.append(resulto)
                
                if self.time:                 
                    #time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                    # use log time as RETAIN
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))
                    
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits
            zp= nn.ZeroPad2d((0,0,lpp,0)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise
  
            if self.diag==1:
                ehr_seq_td= Variable(torch.stack(ehr_seq_tld,0))
                ehr_seq_td= zp(ehr_seq_td) ## zero pad the visits diag codes
                mbd.append(ehr_seq_td)
            if self.med==1: 
                ehr_seq_tm= Variable(torch.stack(ehr_seq_tlm,0)) 
                ehr_seq_tm= zp(ehr_seq_tm) ## zero pad the visits med codes
                mbm.append(ehr_seq_tm)
            if self.oth==1: 
                ehr_seq_to= Variable(torch.stack(ehr_seq_tlo,0)) 
                ehr_seq_to= zp(ehr_seq_to) ## zero pad the visits dem&other codes
                mbo.append(ehr_seq_to)
            
            if self.time:
                time_dim_v= Variable(torch.stack(time_dim,0))
                time_dim_pv= zp(time_dim_v)## zero pad the visits time diff codes
                mtd.append(time_dim_pv)

            
        #mb_t= Variable(torch.stack(mb,0)) 
        #if use_cuda:
        #    mb_t.cuda()
        #embedded = self.embed(mb_t)  ## Embedding for codes
        #embedded = torch.sum(embedded, dim=2) #### split all the above
        
        if self.diag==1:
            mb_td= Variable(torch.stack(mbd,0))
            if use_cuda: mb_td.cuda()
            embedded_d = torch.sum(self.embed_d(mb_td), dim=2)
            embedded= embedded_d 
        if self.med==1: 
            mb_tm= Variable(torch.stack(mbm,0))
            if use_cuda: mb_tm.cuda()  
            embedded_m = torch.sum(self.embed_m(mb_tm), dim=2)
            if self.diag==1: embedded=torch.cat((embedded,embedded_m),dim=2)
            else: embedded=embedded_m 
                
        if self.oth==1: 
            mb_to= Variable(torch.stack(mbo,0))
            if use_cuda: mb_to.cuda()
            embedded_o = torch.sum(self.embed_o(mb_to), dim=2)
            if self.diag + self.med > 0 : embedded=torch.cat((embedded,embedded_o),dim=2)
            else: embedded=embedded_o 

        #embedded=torch.cat((embedded_d,embedded_m,embedded_o),dim=2)## the concatination of above
        
        lbt_t= Variable(torch.stack(lbt,0))
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            if use_cuda: mtd_t.cuda()
            out_emb= torch.cat((embedded,mtd_t),dim=2)
        else:
            out_emb= embedded
        return out_emb, lbt_t,seq_l #,dem_emb
    
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

        x_in = x_in.permute(1,0,2) ## QRNN not support batch first
        #x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)
        h_0 = self.init_hidden()
        output, hidden,_ = self.rnn_c(x_in,h_0) 
        if self.cell_type == "LSTM" or self.cell_type == "TLSTM":
            hidden=hidden[0]
        if self.bi==2:
            output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1)))
        #elif self.cell_type == "TLSTM":
            #output = hidden
        else:
            output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze(), lt.squeeze()




