import os
import pandas as pd
import numpy as np
from datetime import datetime 
try:
    import cPickle as pickle
except:
    import pickle
import time
from tqdm import tqdm

import string
import re
import sklearn.metrics as m
from sklearn.metrics import roc_auc_score
from termcolor import colored, cprint
import random

## ML and Stats 
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
import sklearn.linear_model  as lm
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
 
# Similarly LGBMRegressor can also be imported for a regression model.
from lightgbm import LGBMClassifier

## Visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.graph_objs import *
from IPython.display import HTML

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import logging
import optuna

###GPU enabling and device allocation
use_cuda = torch.cuda.is_available()
if use_cuda: torch.cuda.set_device(5)
import concurrent.futures
from sklearn.preprocessing import StandardScaler
import sklearn
from importlib import reload
import functools
from concurrent.futures import ProcessPoolExecutor
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from collections import defaultdict
import ast

def iter_batch2(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    for _ in range(samplesize):
        results.append(iterator.__next__())
    random.shuffle(results)  
    return results

# Sample data (to be replaced with actual data)
# train_sl = [
#     ['patient1', [0], [[1, 'feature1'], [2, 'feature2']]],
#     ['patient2', [1], [[1, 'feature3'], [2, 'feature4']]]
# ]

def process_chunk(chunk):
    final_chunk = pd.DataFrame(columns=['patient', 'day', 'features'])
    final_outcome_chunk = pd.DataFrame(columns=['patient', 'outcome'])
    c = 0
    for i in range(len(chunk)):
        day = 0
        for f in range(len(chunk[i][2])):
            final_chunk.loc[c, 'patient'] = chunk[i][0]
            final_outcome_chunk.loc[c, 'patient'] = chunk[i][0]
            final_outcome_chunk.loc[c, 'outcome'] = chunk[i][1][0]
            final_chunk.loc[c, 'day'] = day + chunk[i][2][f][0][0]
            day = day + chunk[i][2][f][0][0]
            final_chunk.loc[c, 'features'] = chunk[i][2][f][1]
            c += 1
    return (final_chunk, final_outcome_chunk)

def process_data_multiprocessing(train_sl):
    # Number of CPUs and chunk size calculation
    num_cpus = 42
    chunk_size = len(train_sl) // num_cpus
    chunks = [train_sl[i:i + chunk_size] for i in range(0, len(train_sl), chunk_size)]
    
    # Process data in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(process_chunk, chunks))
        
    # Merge results from all processes
    final = pd.concat([res[0] for res in results], ignore_index=True)
    final_outcome = pd.concat([res[1] for res in results], ignore_index=True)
    final_outcome = final_outcome.drop_duplicates()
    
    return final, final_outcome


def process_data_multiprocessing_time_elapse(test_input):
    expanded_features = test_input.explode('features').drop_duplicates(subset=['patient', 'features'])
    expanded_features2 = test_input.explode('features').drop_duplicates()
    # Number of CPUs and chunk size calculation
    num_cpus = 40
    chunk_size = expanded_features['patient'].nunique() // num_cpus
    patient_id_temp = [expanded_features['patient'].unique()[i:i + chunk_size] for i in range(0, expanded_features['patient'].nunique(), chunk_size)]
    
    # Process data in parallel
    results = []
    results2 = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(functools.partial(process_chunk_time_elapse, expanded_features= expanded_features), patient_id_temp))
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results2 = list(executor.map(functools.partial(process_chunk_count, expanded_features2=expanded_features2), patient_id_temp))
        
    # Merge results from all processes
    feature_time_elapsed = pd.concat([res.reset_index() for res in results], ignore_index=True)
    feature_counts = pd.concat([res for res in results2], ignore_index=True)
    feature_counts.drop_duplicates(inplace=True)
       
    feature_time_elapsed = feature_time_elapsed.drop_duplicates()
    feature_time_elapsed.columns = [f'feature_time_elapsed_{col}' for col in feature_time_elapsed.columns]
    feature_time_elapsed.fillna(9999, inplace=True)
    
    feature_counts.columns = [f'feature_{col}' for col in feature_counts.columns]
    
    feature_time_elapsed = feature_time_elapsed.rename(columns={'feature_time_elapsed_patient': 'patient'})
    feature_counts = feature_counts.rename(columns={'feature_patient': 'patient'})
    
    feature_time_elapsed = feature_time_elapsed.set_index('patient')
    feature_counts = feature_counts.set_index('patient')
    
    final_table = feature_counts.join(feature_time_elapsed, how = 'outer')
    
    return  final_table


def process_data(data):
    test_input, test_output = process_data_multiprocessing(data)
    final_table = process_data_multiprocessing_time_elapse(test_input)
    master_columns = final_table[:0]
    feature_cols = [col for col in master_columns.columns if 'feature_feature' in col]
    time_cols = [col for col in master_columns.columns if 'feature_time_elapsed' in col]
    
    final_table2 = master_columns.append(final_table)
    final_table2[time_cols] = final_table2[time_cols].fillna(9999)
    final_table2[feature_cols] = final_table2[feature_cols].fillna(0)
    
    final_table_output = test_output.set_index('patient').join(final_table2, how='outer')
    
    return final_table_output

def final_table_elapse(test_input, test_outcome):
    # Expand the features lists into separate rows
    expanded_features = test_input.explode('features')

    # Create a pivot table to count occurrences of each feature for each patient
    feature_counts = expanded_features.groupby(['patient', 'features']).size().unstack(fill_value=0)
    feature_counts.columns = [f'feature_{col}' for col in feature_counts.columns]

    # Reset the index
    feature_counts = feature_counts.reset_index()

    feature_counts.head()

    # Calculate the days since the most recent appearance of each feature for each patient
    most_recent_feature_date = expanded_features.groupby(['patient', 'features'])['day'].max().unstack(fill_value=None)
    feature_time_elapsed = most_recent_feature_date.sub(0).fillna(0).astype(int)
    feature_time_elapsed.columns = [f'feature_time_elapsed_{col}' for col in feature_time_elapsed.columns]

    # Reset the index
    feature_time_elapsed = feature_time_elapsed.reset_index()

    feature_time_elapsed.head()

    # Merge the feature counts and feature time elapsed dataframes
    merged_features = pd.merge(feature_counts, feature_time_elapsed, on='patient', how='outer')

    # Merge the merged_features dataframe with the test_outcome dataframe
    final_table = pd.merge(merged_features, test_outcome, on='patient', how='left')

    return final_table


def final_LR_table(train_sl):
    final, final_outcome = process_data_multiprocessing(train_sl)
    final_table = final_table_elapse(test_input, test_outcome)
    return final_table

def process_chunk_time_elapse(patient_id_temp, expanded_features):
    final_chunk = pd.pivot_table(expanded_features[expanded_features['patient'].isin(patient_id_temp)], columns=['features'], index=['patient'], values = 'day', aggfunc=np.min)
    return (final_chunk)

def process_chunk_count(patient_id_temp, expanded_features2):
    chunk_data = expanded_features2[expanded_features2['patient'].isin(patient_id_temp)]
    feature_counts = chunk_data.groupby(['patient', 'features']).size().unstack(fill_value=0)
    feature_counts.columns = [f'feature_{col}' for col in feature_counts.columns]

    # Reset the index
    feature_counts = feature_counts.reset_index()

    return feature_counts

def process_chunk(chunk):
    temp = chunk.iloc[:, 2:].values
    temp2 = chunk.iloc[:, 1:2].values
    temp3 = chunk.iloc[:, :1].values
    return temp, temp2, temp3

def np_array_parallel(file_name, num_cpus=30):
    chunks = pd.read_csv(file_name, chunksize=100000)

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(process_chunk, chunks))

    final_array = np.vstack([result[0] for result in results])
    final_outcome = np.vstack([result[1] for result in results])
    final_id = np.vstack([result[2] for result in results])

    return final_id, final_array, final_outcome