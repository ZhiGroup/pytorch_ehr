# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:53:25 2018

@author: jzhu8
"""
import numpy as np
import random

"""
merged_set takes in the format of labeled_ehr_seq_list,
for each patient visit is either a list for LG
or list of list for NN models
To do: 1. future load_data might take in train and test data from seperate files (done in main file)
       2. future load_data might take in time sequence component
"""
def load_data(merged_set, test_ratio = 0.2, validation_ratio = 0.1):  
  
    
    dataSize = len(merged_set)
    nTest = int(test_ratio * dataSize)
    nValid = int(validation_ratio * dataSize) 
    
    random.seed(3) 
    random.shuffle(merged_set)

    test_set = merged_set[:nTest]
    valid_set = merged_set[nTest:nTest+nValid]
    train_set = merged_set[nTest+nValid:]

    return train_set, valid_set, test_set    
