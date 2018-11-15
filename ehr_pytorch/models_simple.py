#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:47:31 2018

@author: ginnyzhu
"""

import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import export_graphviz
import sys, random
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import math
from sklearn.metrics import roc_auc_score
import sklearn.metrics as m
import sklearn.linear_model as lm


# Load data set and target values
train_sl= pickle.load(open('/data/projects/py_ehr_2/Data/readm_new_h143_90_tp.train', 'rb'), encoding='bytes')
test_sl= pickle.load(open('/data/projects/py_ehr_2/Data/readm_new_h143_90_tp.test', 'rb'), encoding='bytes')
valid_sl= pickle.load(open('/data/projects/py_ehr_2/Data/readm_new_h143_90_tp.valid', 'rb'), encoding='bytes')


# Data Preparation
#simple data processing
def sepalists(ehrlists):
    pts = []
    labels = []
    features = []
    for pt in ehrlists:
        pts.append(pt[0])
        labels.append(pt[1])
        x = []
        for i in range(len(pt[2])):
            x.extend(pt[2][i][1])
        features.append(x)
    return pts, labels, features  
    
    
pts_tr, labels_tr, features_tr = sepalists(train_sl)
pts_v, labels_v, features_v = sepalists(valid_sl)
pts_t, labels_t, features_t = sepalists(test_sl)

# OneHot encoding
# wrong 
mlb = MultiLabelBinarizer(classes=range(1,15816)) #this or previous need to be modified
nfeatures_tr = mlb.fit_transform(features_tr)
nfeatures_v = mlb.transform(features_v)
nfeatures_t = mlb.transform(features_t)


# Output into a logfile 
def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()

logFile='HF_models_simple.log'
header = 'Model|vAUC|tAUC'
print2file(header, logFile)


#model 1: Random Forest 
EHR_RF = RandomForestClassifier(bootstrap=True, 
                                class_weight={0: 9, 1: 1},
                                criterion='gini', 
                                max_depth=80, 
                                max_features='auto',
                                max_leaf_nodes=None, 
                                min_impurity_decrease=0.0,
                                min_impurity_split=None, 
                                min_samples_leaf=1,
                                min_samples_split=2, 
                                min_weight_fraction_leaf=0.0,
                                n_estimators=1000, 
                                n_jobs=-1, 
                                oob_score=True,
                                random_state=None,
                                verbose=0, 
                                warm_start=True)
   
EHR_RF.fit(nfeatures_tr, labels_tr)
  
v_predictions = EHR_RF.predict(nfeatures_v )
v_auc=roc_auc_score(labels_v,v_predictions)

t_predictions = EHR_RF.predict(nfeatures_t )
t_auc=roc_auc_score(labels_t,t_predictions)

pFile= str(EHR_RF)+'|'+str(v_auc)+'|'+str(t_auc)

print2file(pFile, logFile)



##Model 2: Logistic Regression
EHR_LR = LogistiRegression()
EHR_LR.fit(nfeatures_tr, labels_tr)
  
v_predictions = EHR_LR.predict(nfeatures_v )
v_auc=roc_auc_score(labels_v,v_predictions)

t_predictions = EHR_LR.predict(nfeatures_t )
t_auc=roc_auc_score(labels_t,t_predictions)

pFile= str(EHR_LR)+'|'+str(v_auc)+'|'+str(t_auc)

print2file(pFile, logFile)
  

def evaluate(model, test_features, test_labels):
   predictions = model.predict(test_features)
   pred_prob=model.predict_proba(test_features)
   print(pred_prob[:,1])
   errors = abs(predictions - test_labels)
   mape = 100 * np.mean(errors / test_labels)
   #accuracy = 100 - mape
   accuracy = m.accuracy_score(test_labels,predictions)
   auc=roc_auc_score(test_labels,predictions)
   auc_p=roc_auc_score(test_labels,pred_prob[:,1])
   F1=m.f1_score(test_labels,predictions)
   #print(predictions,errors,test_labels)
   print('Model Performance')
   print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
   print('Accuracy = {:0.2f}%.'.format(accuracy*100))
   print('AUC_p = {:0.2f}%.'.format(auc_p*100))
   print('F1 score = {:0.2f}.'.format(F1))

   return test_labels,predictions

#For random forest
evaluate(EHR_RF, nfeatures_t, labels_t)
#for logistic regression  
#should be the same
evaluate(EHR_LR, nfeatures_t, labels_t)
