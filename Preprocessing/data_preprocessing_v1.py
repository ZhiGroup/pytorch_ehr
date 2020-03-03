'''
# This script processes originally extracted data for example from Cerner HealthFacts Dataset
# and builds pickled lists including a full list that includes all information for case and controls
# The output data include splitted cPickled lists suitable for training Doctor AI or RETAIN using similar logic of process_mimic Written by Edward Choi
# Additionally it outputs pickled list of the following shape
#[[pt1_id,label,[
#                  [[delta_time 0],[list of Medical codes in Visit0]],
#                  [[delta_time between V0 and V1],[list of Medical codes in Visit2]],
#                   ......]],
# [pt2_id,label,[[[delta_time 0],[list of Medical codes in Visit0 ]],[[delta_time between V0 and V1],[list of Medical codes in Visit2]],......]]]
#
# Usage: feed this script with Case file and Control files each is just a three columns like pt_id | medical_code | visit_date and execute like:
#
# python data_preprocessing_v1.py <Case File> <Control File> <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> 
# you can optionally activate <case_samplesize> <control_samplesize> based on your cohort definition
# This file will later split the data to training , validation and Test sets of ratio
# Output files for Doctor AI or RETAIN
# <output file>.pts: List of unique Cerner Patient SKs. Created for validation and comparison purposes
# <output file>.labels: List of binary values indicating the label of each patient (either case(1) or control(0)) 
# <output file>.days: List of List of integers representing number of days between consequitive vists. The outer List is for each patient. The inner List is for each visit made by each# patient
# <output file>.visits: List of List of List of integer diagnosis codes. The outer List is for each patient. The middle List contains visits made by each patient. The inner List
# contains the integer diagnosis codes that occurred in each visit
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.
# Main output files for the baseline RNN models are <output file>.combined
'''

import sys
from optparse import OptionParser
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import random
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
#import timeit ( for time tracking if required)

if __name__ == '__main__':
    
   caseFile= sys.argv[1]
   controlFile= sys.argv[2]
   typeFile= sys.argv[3]
   outFile = sys.argv[4]
   #samplesize_case = int(sys.argv[5])
   #samplesize_ctrl = int(sys.argv[6])
   parser = OptionParser()
   (options, args) = parser.parse_args()
 
   
   #_start = timeit.timeit()
  
   debug=False
   #np.random.seed(1)
   time_list = []
   dates_list =[]
   label_list = []
   pt_list = []

   
   print (" Loading cases and controls" ) 
  
   ## loading Case
   print('loading cases')
   data_case = pd.read_table(caseFile)
   data_case.columns = ["Pt_id", "ICD", "Time"]
   data_case['Label'] = 1
   #data_case=data_case[~data_case["ICD"].str.startswith('P')] ### use if you need to exclude certain type of codes
    

   ## loading Control
   print('loading ctrls')
   data_control = pd.read_table(controlFile)
   data_control.columns = ["Pt_id", "ICD", "Time"]
   data_control['Label'] = 0
   #data_control=data_control[~data_control["ICD"].str.startswith('P')] ### use if you need to exclude certain type of codes
   
  
   ### An example of sampling code: Control Sampling
   #print('ctrls sampling')       
   #ctr_sk=data_control["Pt_id"]
   #ctr_sk=ctr_sk.drop_duplicates()
   #ctr_sk_samp=ctr_sk.sample(n=samplesize_ctrl)
   #data_control=data_control[data_control["Pt_id"].isin(ctr_sk_samp.values.tolist())]

   


   data_l= pd.concat([data_case,data_control])   
   
   ## loading the types
  
   if typeFile=='NA': 
       types={}
   else:
      with open(typeFile, 'rb') as t2:
             types=pickle.load(t2)
        
   #end_time = timeit.timeit()
   #print ("consumed time for data loading",(_start -end_time)/1000.0 )
    
   full_list=[]
   index_date = {}
   time_list = []
   dates_list =[]
   label_list = []
   pt_list = []
   dur_list=[]
   newVisit_list = []
   count=0
   
   for Pt, group in data_l.groupby('Pt_id'):
            data_i_c = []
            data_dt_c = []
            for Time, subgroup in group.sort_values(['Time'], ascending=False).groupby('Time', sort=False): ### ascending=True normal order ascending=False reveresed order
                        data_i_c.append(np.array(subgroup['ICD']).tolist())             
                        data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))
            if len(data_i_c) > 0:
                 # creating the duration in days between visits list, first visit marked with 0        
                  v_dur_c=[]
            if len(data_dt_c)<=1:
                     v_dur_c=[0]
            else:
                     for jx in range (len(data_dt_c)):
                         if jx==0:
                             v_dur_c.append(jx)
                         else:
                             #xx = ((dt.strptime(data_dt_c[jx-1], '%d-%b-%y'))-(dt.strptime(data_dt_c[jx], '%d-%b-%y'))).days ## use if original data have time information or different date format
                             #xx = (data_dt_c[jx]- data_dt_c[jx-1]).days ### normal order
                             xx = (data_dt_c[jx-1] - data_dt_c[jx]).days ## reversed order                            
                             v_dur_c.append(xx)
            
            ### Diagnosis recoding
            newPatient_c = []
            for visit in data_i_c:
                      newVisit_c = []
                      for code in visit:
                    				if code in types: newVisit_c.append(types[code])
                    				else:                             
                    					  types[code] = len(types)+1
                    					  newVisit_c.append(types[code])
                      newPatient_c.append(newVisit_c)
                                                            
            if len(data_i_c) > 0: ## only save non-empty entries
                  label_list.append(group.iloc[0]['Label'])
                  pt_list.append(Pt)
                  newVisit_list.append(newPatient_c)
                  dur_list.append(v_dur_c)
 
            count=count+1
            if count % 1000 == 0: print ('processed %d pts' % count)

   
    ### Creating the full pickled lists ### uncomment if you need to dump the all data before splitting

   #pickle.dump(label_list, open(outFile+'.labels', 'wb'), -1)
   #pickle.dump(newVisit_list, open(outFile+'.visits', 'wb'), -1)
   pickle.dump(types, open(outFile+'.types', 'wb'), -1)
   #pickle.dump(pt_list, open(outFile+'.pts', 'wb'), -1)
   #pickle.dump(dur_list, open(outFile+'.days', 'wb'), -1)

  
    ### Random split to train ,test and validation sets
   print ("Splitting")
   dataSize = len(pt_list)
   #np.random.seed(0)
   ind = np.random.permutation(dataSize)
   nTest = int(0.2 * dataSize)
   nValid = int(0.1 * dataSize)
   test_indices = ind[:nTest]
   valid_indices = ind[nTest:nTest+nValid]
   train_indices = ind[nTest+nValid:]
    
   for subset in ['train','valid','test']:
       if subset =='train':
            indices = train_indices
       elif subset =='valid':
            indices = valid_indices
       elif subset =='test':
            indices = test_indices
       else: 
            print ('error')
            break
       subset_x = [newVisit_list[i] for i in indices]
       subset_y = [label_list[i] for i in indices]
       subset_t = [dur_list[i] for i in indices]
       subset_p = [pt_list[i] for i in indices]
       nseqfile = outFile +'.visits.'+subset
       nlabfile = outFile +'.labels.'+subset
       ntimefile = outFile +'.days.'+subset
       nptfile = outFile +'.pts.'+subset
       pickle.dump(subset_x, open(nseqfile, 'wb'),protocol=2)
       pickle.dump(subset_y, open(nlabfile, 'wb'),protocol=2)
       pickle.dump(subset_t, open(ntimefile, 'wb'),protocol=2)
       pickle.dump(subset_p, open(nptfile, 'wb'),protocol=2)    
        
    ### Create the combined list for the Pytorch RNN
   fset=[]
   print ('Reparsing')
   for pt_idx in range(len(pt_list)):
                pt_sk= pt_list[pt_idx]
                pt_lbl= label_list[pt_idx]
                pt_vis= newVisit_list[pt_idx]
                pt_td= dur_list[pt_idx]
                d_gr=[]
                n_seq=[]
                d_a_v=[]
                for v in range(len(pt_vis)):
                        nv=[]
                        nv.append([pt_td[v]])
                        nv.append(pt_vis[v])                   
                        n_seq.append(nv)
                n_pt= [pt_sk,pt_lbl,n_seq]
                fset.append(n_pt)              
    
   ### split the full combined set to the same as individual files

   train_set_full = [fset[i] for i in train_indices]
   test_set_full = [fset[i] for i in test_indices]
   valid_set_full = [fset[i] for i in valid_indices]
   ctrfilename=outFile+'.combined.train'
   ctstfilename=outFile+'.combined.test'
   cvalfilename=outFile+'.combined.valid'    
   pickle.dump(train_set_full, open(ctrfilename, 'wb'), -1)
   pickle.dump(test_set_full, open(ctstfilename, 'wb'), -1)
   pickle.dump(valid_set_full, open(cvalfilename, 'wb'), -1)
  

