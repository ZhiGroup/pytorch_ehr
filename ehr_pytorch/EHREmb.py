"""
This Class is mainly for the creation of the EHR patients' visits embedding
which is the key input for all the deep learning models in this Repo

@authors: Lrasmy , Jzhu @ DeguiZhi Lab - UTHealth SBMI
Last revised Mar 25 2019

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
    def __init__(self, input_size, embed_dim ,hidden_size, n_layers=1,dropout_r=0.1,cell_type='LSTM', bii=False, time=False , preTrainEmb='', packPadMode = True):
        super(EHREmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.time=time
        self.preTrainEmb=preTrainEmb
        self.packPadMode = packPadMode
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
            self.in_size=(self.diag+self.med+self.oth)*self.embed_dim
       
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
            self.rnn_c = self.cell(self.in_size, self.hidden_size, num_layers=self.n_layers, dropout= self.dropout_r, bidirectional=self.bi)
         
        self.out = nn.Linear(self.hidden_size*self.bi,1)
        self.sigmoid = nn.Sigmoid()
      
                            
    #let's define this class method
    def EmbedPatients_MB(self,input): #let's define this
    
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
                if self.cell_type == 'TLSTM': #Ginny the correct implementation for TLSTM time
                    time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                elif self.time:                 
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))
            
            
            ehr_seq_t= Variable(torch.stack(ehr_seq_tl,0)) 
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits PLEASE MODIFY 
            if self.packPadMode:
                zp= nn.ZeroPad2d((0,0,0,lpp)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise. Ginny Done!
            else: 
                zp= nn.ZeroPad2d((0,0,lpp,0))
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
        return out_emb, lbt_t,seq_l #Always should return these3
  
    def EmbedPatients_SMB(self,input): ## splitted input
    
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
                
                if self.cell_type == 'TLSTM': #Ginny: correct implementation for TLSTM time
                    time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))
                elif self.time:                 
                    time_dim.append(Variable(torch.div(flt_typ([1.0]), torch.log(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ) + flt_typ([2.7183])))))
            
                    
            lpp= lp-lpx ## diff be max seq in minibatch and cnt of pt visits
        
            if self.packPadMode:
                zp= nn.ZeroPad2d((0,0,0,lpp)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise.
            else: 
                zp= nn.ZeroPad2d((0,0,lpp,0))
      
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
        return out_emb, lbt_t,seq_l 
   

