# coding: utf-8
"""
Created on Mon Oct 29 12:57:40 2018

@author: ginnyzhu
"""

#general utilities
from __future__ import print_function, division
#import os
#from os import walk
from tabulate import tabulate
#import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle
import warnings
warnings.filterwarnings("ignore")
plt.ion()

#torch libraries 
import torch
from torch.utils.data import Dataset, DataLoader


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


#customized parts for EHRdataloader
def my_collate(batch):
    return list(batch)          
            

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
                 timeout=0, worker_init_fn=None):
        DataLoader.__init__(self, dataset, batch_size=128, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=my_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None)
        self.collate_fn = collate_fn
 
########END of main contents of EHRDataloader############
