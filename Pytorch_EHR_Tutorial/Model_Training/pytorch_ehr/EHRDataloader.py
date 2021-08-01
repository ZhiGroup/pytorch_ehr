# coding: utf-8
"""
Created on Mon Oct 29 12:57:40 2018

@authors Jzhu, Lrasmy , Xin128 @ DeguiZhi Lab - UTHealth SBMI

Last updated Feb 20 2020
"""

#general utilities
from __future__ import print_function, division
from tabulate import tabulate
import numpy as np
import random
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle
import warnings
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn
warnings.filterwarnings("ignore")
plt.ion()

#torch libraries 
import torch
from torch.utils.data import Dataset, DataLoader

use_cuda = torch.cuda.is_available()
#use_cuda=False

# Dataset class loaded from pickles
class EHRdataFromPickles(Dataset):
    def __init__(self, root_dir, file = None, transform=None, sort = True, model='RNN', test_ratio = 0, valid_ratio = 0):
        """
        Args:
            1) root_dir (string): Path to pickled file(s).
                               The directory contains the directory to file(s): specify 'file' 
                               please create separate instances from this object if your data is split into train, validation and test files.               
            2) data should have the format: pickled, 4 layer of lists, a single patient's history should look at this (use .__getitem__(someindex, seeDescription = True))
                [310062,
                 0,
                 [[[0],[7, 364, 8, 30, 10, 240, 20, 212, 209, 5, 167, 153, 15, 3027, 11, 596]],
                  [[66], [590, 596, 153, 8, 30, 11, 10, 240, 20, 175, 190, 15, 7, 5, 183, 62]],
                  [[455],[120, 30, 364, 153, 370, 797, 8, 11, 5, 169, 167, 7, 240, 190, 172, 205, 124, 15]]]]
                 where 310062: patient id, 
                       0: no heart failure
                      [0]: visit time indicator (first one), [7, 364, 8, 30, 10, 240, 20, 212, 209, 5, 167, 153, 15, 3027, 11, 596]: visit codes.
                      
            3)transform (optional): Optional transform to be applied on a sample. Data augmentation related. 
            4)test_ratio,  valid_ratio: ratios for splitting the data if needed.
        """
        self.file = None
        if file != None:
            self.file = file
            self.data = pickle.load(open(root_dir + file, 'rb'), encoding='bytes') 
            if sort: 
                self.data.sort(key=lambda pt:len(pt[2]),reverse=True) 
            self.test_ratio = test_ratio 
            self.valid_ratio = valid_ratio       
        else:
            print('No file specified')
        self.root_dir = root_dir  
        self.transform = transform 
        
    def __splitdata__(self, sort = True):
        
        random.seed(3)
        random.shuffle(self.data)
        dataSize = len(self.data)
        nTest = int(self.test_ratio * dataSize)
        nValid = int(self.valid_ratio * dataSize) 
        test= self.data[:nTest]
        valid = self.data[nTest:nTest+nValid]
        train = self.data[nTest+nValid:]
        if sort: 
            #sort train, validation and test again
            test.sort(key=lambda pt:len(pt[2]),reverse=True) 
            valid.sort(key=lambda pt:len(pt[2]),reverse=True) 
            train.sort(key=lambda pt:len(pt[2]),reverse=True) 
        return train, test, valid
        
                                     
    def __getitem__(self, idx, seeDescription = False):
        '''
        Return the patient data of index: idx of a 4-layer list 
        patient_id (pt_sk); 
        label: 0 for no, 1 for yes; 
        visit_time: int indicator of the time elapsed from the previous visit, so first visit_time for each patient is always [0];
        visit_codes: codes for each visit.
        '''
        if self.file != None: 
            sample = self.data[idx]
        else:
            print('No file specified')
        if self.transform:
            sample = self.transform(sample)
        
        vistc = np.asarray(sample[2])
        desc = {'patient_id': sample[0], 'label': sample[1], 'visit_time': vistc[:,0],'visit_codes':vistc[:,1]}     
        if seeDescription: 
            '''
            if this is True:
            You will get a descriptipn of what each part of data stands for
            '''
            print(tabulate([['patient_id', desc['patient_id']], ['label', desc['label']], 
                            ['visit_time', desc['visit_time']], ['visit_codes', desc['visit_codes']]], 
                           headers=['data_description', 'data'], tablefmt='orgtbl'))
        #print('\n Raw sample of index :', str(idx))     
        return sample

    def __len__(self):
        ''' 
        just the length of data
        '''
        if self.file != None:
            return len(self.data)
        else: 
            print('No file specified')



# Dataset class from already  loaded pickled lists
class EHRdataFromLoadedPickles(Dataset):
    def __init__(self, loaded_list, transform=None, sort = True, model='RNN'):
        """
        Args:
            1) loaded_list from pickled file
            2) data should have the format: pickled, 4 layer of lists, a single patient's history should look at this (use .__getitem__(someindex, seeDescription = True))
                [310062,
                 0,
                 [[[0],[7, 364, 8, 30, 10, 240, 20, 212, 209, 5, 167, 153, 15, 3027, 11, 596]],
                  [[66], [590, 596, 153, 8, 30, 11, 10, 240, 20, 175, 190, 15, 7, 5, 183, 62]],
                  [[455],[120, 30, 364, 153, 370, 797, 8, 11, 5, 169, 167, 7, 240, 190, 172, 205, 124, 15]]]]
                 where 310062: patient id, 
                       0: no heart failure
                      [0]: visit time indicator (first one), [7, 364, 8, 30, 10, 240, 20, 212, 209, 5, 167, 153, 15, 3027, 11, 596]: visit codes.                      
            3)transform (optional): Optional transform to be applied on a sample. Data augmentation related. 
            4)test_ratio,  valid_ratio: ratios for splitting the data if needed.
        """
        self.data = loaded_list 
        if sort: 
                self.data.sort(key=lambda pt:len(pt[2]),reverse=True) 
        self.transform = transform 
              
                                     
    def __getitem__(self, idx, seeDescription = False):
        '''
        Return the patient data of index: idx of a 4-layer list 
        patient_id (pt_sk); 
        label: 0 for no, 1 for yes; 
        visit_time: int indicator of the time elapsed from the previous visit, so first visit_time for each patient is always [0];
        visit_codes: codes for each visit.
        '''
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        
        vistc = np.asarray(sample[2])
        desc = {'patient_id': sample[0], 'label': sample[1], 'visit_time': vistc[:,0],'visit_codes':vistc[:,1]}     
        if seeDescription: 
            '''
            if this is True:
            You will get a descriptipn of what each part of data stands for
            '''
            print(tabulate([['patient_id', desc['patient_id']], ['label', desc['label']], 
                            ['visit_time', desc['visit_time']], ['visit_codes', desc['visit_codes']]], 
                           headers=['data_description', 'data'], tablefmt='orgtbl'))
        #print('\n Raw sample of index :', str(idx))     
        return sample

    def __len__(self):
        return len(self.data)
            
def preprocess(batch,pack_pad,surv,half): ### LR Sep 30 20 added surv_m
    # Check cuda availability
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
    bsize=len(batch) ## number of patients in minibatch
    lp= len(max(batch, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch
    llv=0
    for x in batch:
        lv= len(max(x[-1], key=lambda xmb: len(xmb[1]))[1])
        if llv < lv:
            llv=lv     # max number of codes per visit in minibatch        
    for pt in batch:
        sk,label,ehr_seq_l = pt
        lpx=len(ehr_seq_l) ## no of visits in pt record
        seq_l.append(lpx)
        if surv: lbt.append(Variable(flt_typ([label])))### LR Sep 30 20 added surv_m
        else: lbt.append(Variable(flt_typ([[float(label)]])))
        ehr_seq_tl=[]
        time_dim=[]
        for ehr_seq in ehr_seq_l:
            pd=(0, (llv -len(ehr_seq[1])))
            result = F.pad(torch.from_numpy(np.asarray(ehr_seq[1],dtype=int)).type(lnt_typ),pd,"constant", 0)
            ehr_seq_tl.append(result)
            time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))

        ehr_seq_t= Variable(torch.stack(ehr_seq_tl,0)) 
        lpp= lp-lpx ## diffence between max seq in minibatch and cnt of patient visits 
        if pack_pad:
            zp= nn.ZeroPad2d((0,0,0,lpp)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise. 
        else: 
            zp= nn.ZeroPad2d((0,0,lpp,0))
        ehr_seq_t= zp(ehr_seq_t) ## zero pad the visits med codes
        mb.append(ehr_seq_t)
        time_dim_v= Variable(torch.stack(time_dim,0))
        time_dim_pv= zp(time_dim_v) ## zero pad the visits time diff codes
        mtd.append(time_dim_pv)
    lbt_t= Variable(torch.stack(lbt,0))
    mb_t= Variable(torch.stack(mb,0)) 
    if use_cuda:
        mb_t.cuda()
        lbt_t.cuda()
    if half: 
        mb_t.half()
        mtd.half()
    return mb_t, lbt_t,seq_l, mtd 
            
def preprocess_multilabel(batch,pack_pad,half): ### LR Feb 18 21 for multi-label 
    # Check cuda availability
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
    bsize=len(batch) ## number of patients in minibatch
    lp= len(max(batch, key=lambda xmb: len(xmb[-1]))[-1]) ## maximum number of visits per patients in minibatch
    llv=0
    for x in batch:
        lv= len(max(x[-1], key=lambda xmb: len(xmb[1]))[1])
        if llv < lv:
            llv=lv     # max number of codes per visit in minibatch        
    for pt in batch:
        sk,label,ehr_seq_l = pt
        lpx=len(ehr_seq_l) ## no of visits in pt record
        seq_l.append(lpx)
        lbt.append(Variable(flt_typ([label])))### LR Sep 30 20 added surv_m
        ehr_seq_tl=[]
        time_dim=[]
        for ehr_seq in ehr_seq_l:
            pd=(0, (llv -len(ehr_seq[1])))
            result = F.pad(torch.from_numpy(np.asarray(ehr_seq[1],dtype=int)).type(lnt_typ),pd,"constant", 0)
            ehr_seq_tl.append(result)
            time_dim.append(Variable(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ)))

        ehr_seq_t= Variable(torch.stack(ehr_seq_tl,0)) 
        lpp= lp-lpx ## diffence between max seq in minibatch and cnt of patient visits 
        if pack_pad:
            zp= nn.ZeroPad2d((0,0,0,lpp)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise. 
        else: 
            zp= nn.ZeroPad2d((0,0,lpp,0))
        ehr_seq_t= zp(ehr_seq_t) ## zero pad the visits med codes
        mb.append(ehr_seq_t)
        time_dim_v= Variable(torch.stack(time_dim,0))
        time_dim_pv= zp(time_dim_v) ## zero pad the visits time diff codes
        mtd.append(time_dim_pv)
    lbt_t= Variable(torch.stack(lbt,0))
    mb_t= Variable(torch.stack(mb,0)) 
    if use_cuda:
        mb_t.cuda()
        lbt_t.cuda()
    if half: 
        mb_t.half()
        #print('mb_t should be FB16',mb_t)
    return mb_t, lbt_t,seq_l, mtd 

         
#customized parts for EHRdataloader
def my_collate(batch):
    if multilabel_m : mb_t, lbt_t,seq_l, mtd =preprocess_multilabel(batch,pack_pad,half_m) ### LR Sep 30 20 added surv_m
    else: mb_t, lbt_t,seq_l, mtd = preprocess(batch,pack_pad,surv_m,half_m) ### LR Sep 30 20 added surv_m
    
    return [mb_t, lbt_t,seq_l, mtd]
            

def iter_batch2(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    for _ in range(samplesize):
        results.append(iterator.__next__())
    random.shuffle(results)  
    return results

class EHRdataloader(DataLoader):
    def __init__(self, dataset, batch_size=128, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, packPadMode = False , surv=False,multilbl=False,half=False): ### LR Sep 30 20 added surv
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None)
        
        self.collate_fn = collate_fn
        global pack_pad
        global surv_m ### LR Sep 30 20 added surv_m
        global multilabel_m
        global half_m
        pack_pad = packPadMode
        surv_m=surv ### LR Sep 30 20 added surv_m
        multilabel_m=multilbl
        half_m=half
        if multilabel_m : print('multilabel data processing')
        if half_m: print ('FP16 applied')

 
########END of main contents of EHRDataloader############
