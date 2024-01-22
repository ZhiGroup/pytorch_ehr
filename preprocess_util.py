import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
import pickle

import Levenshtein as lv
import difflib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from mimic4_preprocess_util import *

#util functions for mapping
def contains(include, lower ,data, column ,words):
    if include == 'y':
        if lower == 'y':
            temp = data[data[column].str.lower().str.contains('|'.join(words), na = False)]
        else:
            temp = data[data[column].str.contains('|'.join(words), na = False)]
    else:
        if lower == 'y':
            temp = data[~data[column].str.lower().str.contains('|'.join(words), na = False)]
        else:
            temp = data[~data[column].str.contains('|'.join(words), na = False)]
            
    return temp


### Each data files may contain similar terms. It may be one to many or many to one mactching. 
### To avoid going through all terms of I used "fuzzywuzzy" to find similar terms close to each. 
### This will give you the candidates of the words something up to certain "threshold"

# Before apply this, its better to spell out all abbreviation, such as "HSV" to get closer matches. 
def fuzzy_mapping(MHH_antibiotics_list, prescription_id, iteration, thresh):

    A_cleaned = MHH_antibiotics_list['v1'].unique()
    B_cleaned = prescription_id['fuzzy match'].unique().tolist()
    i = iteration
    df = pd.DataFrame()
    for c in range(0,i):
        if c < i - 1:

            #B_cleaned.extend(dict_labitems['fuzzy match'].unique().tolist())
            tuples_list = [max([(fuzz.token_sort_ratio(i,j), j) for j in B_cleaned])for i in A_cleaned]
            #tuples_list = [x for x in tuples_list if x[0] > 99]
            # Unpack list of tuples into two lists
            similarity_score, fuzzy_match = map(list,zip(*tuples_list))
            temp = pd.DataFrame({"v1":A_cleaned[:len(fuzzy_match)], "fuzzy match": fuzzy_match, "similarity score":similarity_score})
            temp = temp[temp['similarity score'] >= thresh]
            fuzzy_match = temp['fuzzy match'].unique()
            df = df.append(temp)
            display(temp)
            B_cleaned = [fruit for fruit in B_cleaned if fruit not in fuzzy_match]


        else:
            #B_cleaned.extend(dict_labitems['fuzzy match'].unique().tolist())
            tuples_list = [max([(fuzz.token_sort_ratio(i,j), j) for j in B_cleaned])for i in A_cleaned]
            # Unpack list of tuples into two lists
            similarity_score, fuzzy_match = map(list,zip(*tuples_list))
            temp = pd.DataFrame({"v1":A_cleaned, "fuzzy match": fuzzy_match, "similarity score":similarity_score})
            df = df.append(temp)

    # Create pandas DataFrame
    df.drop_duplicates(inplace=True)
    df = df.sort_values(by = ['similarity score', 'v1'], ascending= False)
    return df

def datechange(data):
    data['chartdate'] = data['chartdate'].astype('datetime64')
    data['date'] = data['chartdate'].dt.date
    data = data.sort_values(by = ['chartdate'])
    return data

# Remove cultures within X days from the index cultures
def remove_culture_within(final_label, days):

    final_label2 = pd.DataFrame()
    length = len(final_label)
    c =0
    while length != 0:
        c +=1
        d = 0
        for f in final_label['subject_id'].unique():
            temp = final_label[final_label['subject_id'] == f]
            temp = temp[~temp['charttime'].isnull()]  # Some culture result does not have any dates
            if len(temp) != 0:
                              
                datetime_object = datetime.strptime(temp.iloc[0, 1], '%Y-%m-%d %H:%M:%S')
                temp = temp[temp['chartdate'] <= datetime_object + pd.Timedelta(days, 'D')]
                temp_index = temp.index
                temp = temp.drop_duplicates(subset = ['subject_id'])
                temp['new_subject_id'] = f + '_' + str(c)
                final_label = final_label.drop(temp_index, axis=0)
                final_label2 = final_label2.append(temp)
            else:
                final_label = final_label[final_label['subject_id'] != f] # remove the culture results which does not have dates
                #print(f)
            d += 1
            if d%1000 == 0:
                print('patient', round(d/final_label['subject_id'].nunique()*100, 1), '%')
        length = len(final_label)
        print(c)
    return final_label2

def extract_diag_mimic(data, case):
    d = 1
    data = datechange(data)
    final_df = pd.DataFrame()
    for f in case['new_subject_id'].unique():
        d += 1
        subject_id = case[case['new_subject_id'] == f]['subject_id'].iloc[0]
        index_date = case[case['new_subject_id'] == f]['admittime'].iloc[0] 
        temp = data[data['subject_id'] == subject_id]
        temp = temp[temp['chartdate'] < pd.to_datetime(index_date)]
        temp['new_subject_id'] = f
        final_df = final_df.append(temp)
        if d%1000 == 0:
            print(d)
    return final_df

def extract_diag(data, case):
    d = 1
    data = datechange(data)
    final_df = pd.DataFrame()
    for f in case['new_subject_id'].unique():
        d += 1
        subject_id = case[case['new_subject_id'] == f]['subject_id'].iloc[0]
        index_date = case[case['new_subject_id'] == f]['chartdate'].iloc[0]
        temp = data[data['subject_id'] == subject_id]
        temp = temp[temp['chartdate'] <= pd.to_datetime(index_date)]
        temp['new_subject_id'] = f
        final_df = final_df.append(temp)
        if d%1000 == 0:
            print(d)
    return final_df

def extract_demo(data, case):
    data = datechange(data)
    data2 = data[data['event_code'].str.contains('LOS_')]
    data = data[~data['event_code'].str.contains('LOS_')]
    
    data2['event_code2'] = data2['event_code'].str[len('LOS_'):]
    data2['event_code2'] = data2['event_code2'].str[:-5]
    data2 = data2[data2['event_code2'] != '']
    data2['event_code2'] = data2['event_code2'].astype(int)
    data2['date'] = data2['date'].astype('datetime64')
    data2['end_date'] = data2['date'] + pd.to_timedelta(data2['event_code2'], unit= 'D')
    data2 = data2[data2['event_code2']>=0]
    data2 = data2[data2['event_code2']<=1000]
    
    case = case[['new_subject_id', 'subject_id', 'chartdate']]
    case.columns =['new_subject_id', 'subject_id', 'index_date']
    data = case.merge(data, on = 'subject_id', how = 'left')
    data = data[~data['event_code'].isnull()]
    data = data[data['chartdate'] < data['index_date']]
    data = data[['subject_id', 'chartdate', 'event_code', 'date', 'new_subject_id']]
    
    data2 = case.merge(data2, on = 'subject_id', how = 'left')
    data2 = data2[~data2['event_code'].isnull()]
    data2 = data2[data2['end_date'] < data2['index_date']]
    data2 = data2[['subject_id', 'chartdate', 'event_code', 'date', 'new_subject_id']]
    
    data = data.append(data2)
    return data

def medication(data):
    extract = ['medication', 'route', 'Med']
    extract_dict = {'medication': 'MED_','Institution': 'INT_', 'route': 'ROUTE_', 'Med': 'MED_'}
    final_df = pd.DataFrame()

    final_demo = pd.DataFrame()

    for f in extract:
        temp = data[['subject_id','charttime', f]]
        temp.columns = ['subject_id','chartdate', 'event_code']
        temp['event_code'] = extract_dict[f] + temp['event_code']
        final_demo = final_demo.append(temp)

    return final_demo

def round_half_up(n, decimals=-1):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def demographics(data):
    data['loc'] = data['admission_location']
    extract = ['loc', 'age', 'ethnicity', #'admission_type','insurance',
              'language', 'gender', 'race']
    final_df = pd.DataFrame()
    admit = data[['subject_id', 'admittime']]
    admit['event_code'] = 'admit'
    admit.columns = ['subject_id', 'chartdate', 'event_code']
    final_demo = pd.DataFrame()

    for f in extract:
        temp = data[['subject_id','admittime', f]]
        temp = temp[temp[f]!='']
        temp = temp[~temp[f].isnull()]
        temp.columns = ['subject_id','chartdate', 'event_code']
        temp['event_code'] = temp['event_code'].astype(str)
        temp['event_code'] = f.upper() +'_' +temp['event_code'] 
        final_demo = final_demo.append(temp)
    
    f = 'los'
    temp = data[['subject_id','admittime', f, 'dischtime']]
    temp = temp[temp[f]!='']
    temp = temp[~temp[f].isnull()]
    temp.columns = ['subject_id','chartdate', 'event_code',  'dischtime']
    temp['event_code'] = temp['event_code'].astype(str)
    temp['event_code'] = f.upper() +'_' +temp['event_code'] 
    final_demo = final_demo.append(temp)
       
    final_demo = final_demo.append(admit)
    final_demo = final_demo[~final_demo['event_code'].isnull()]
    final_demo.drop_duplicates(inplace=True)

    return final_demo


def culture_order(data):
    data = data.fillna('0')
    extract = ['Collect Location','Collect Institution', 'Isolation','Admit Diagnosis (related)',  
               'Specimen Category']
    extract_dict = {'Collect Location': 'LOC_','Order': 'ORDER_', 'Collect Institution': 'INT_',
                    'Organism': 'ORG_', 'Isolation': 'ISOL_', 'Admit Diagnosis (related)': 'ADM_DX_',
                   'Specimen Category': 'CxGrp_'}
    result = ['Result Type']  #, 'Result']
    order = ['Order']
 
    final_demo = pd.DataFrame()
    final_order = pd.DataFrame()

    for f in extract:
        temp = data[['subject_id','chartdate', f, 'M_Result_date']]
        temp.columns = ['subject_id','chartdate', 'event_code', 'M_Result_date']
        temp['event_code'] = extract_dict[f] + temp['event_code']
        final_demo = final_demo.append(temp)
        
    for f in order:
        temp = data[['subject_id','chartdate', f, 'M_Result_date']]
        temp.columns = ['subject_id','chartdate', 'event_code', 'M_Result_date']
        temp['event_code'] = extract_dict[f] + temp['event_code']
        final_order = final_order.append(temp)
        
    final_demo2 = pd.DataFrame()
    temp = data[['subject_id','chartdate','Order','Specimen Category','Organism', 'M_Result_date']]
    temp['event_code'] = 'ORG_' + temp['Order'] +temp['Specimen Category'] + '$' + temp['Organism']
    final_demo2 = final_demo2.append(temp)        
        
    
    for f in result:
        temp = data[['subject_id','chartdate','Order','Specimen Category','Organism', f, 'M_Result_date']]
        temp['event_code'] = 'TEST_' + temp['Order'] +'_' + temp['Specimen Category'] + '_' + temp['Organism'] + '_'+ f + '$' + temp[f]
        final_demo2 = final_demo2.append(temp)

    return final_demo, final_order, final_demo2

def extract_sensitivity(data, case, days):
    c = 0
    data = datechange(data)
    case['chartdate'] =case['chartdate'].astype('datetime64')
    data['chartdate'] =data['chartdate'].astype('datetime64')
    final_df = pd.DataFrame()
    for f in case['new_subject_id'].unique():
        subject_id = case[case['new_subject_id'] == f]['subject_id'].iloc[0]
        index_date = case[case['new_subject_id'] == f]['chartdate'].iloc[0]
        temp = data[data['subject_id'] == subject_id]
        temp = temp[temp['chartdate'] <= pd.to_datetime(index_date) - pd.Timedelta(days, 'Day')]
        temp['new_subject_id'] = f
        final_df = final_df.append(temp)
        c +=1
        if c%1000 == 0:
            print(c, round(c/case['new_subject_id'].nunique()*100, 1))
    return final_df

def extract_test_results(data, case):
    data = datechange(data)
    final_df = pd.DataFrame()
    for f in case['new_subject_id'].unique():
        subject_id = case[case['new_subject_id'] == f]['subject_id'].iloc[0]
        index_date = case[case['new_subject_id'] == f]['chartdate'].iloc[0]
        temp = data[data['subject_id'] == subject_id]
        temp = temp[temp['M_Result_date'] <= pd.to_datetime(index_date)]
        temp['new_subject_id'] = f
        final_df = final_df.append(temp)
    return final_df

def extract_diag_mimic2(data, case):
    data = datechange(data)
    case = datechange(case)
    final_df = pd.DataFrame()

    case = case[['new_subject_id', 'subject_id', 'admittime']]
    data = case.merge(data, on = 'subject_id', how = 'outer')
    data = data[~data['icd_code'].isnull()]
    data = data[data['chartdate'] < data['admittime']]
    data = data[['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version',
       'chartdate', 'date', 'new_subject_id', 'admittime']]
    return data

def extract_proc_mimic2(data, case):
    data = datechange(data)
    case = datechange(case)
    final_df = pd.DataFrame()

    case = case[['new_subject_id', 'subject_id', 'admittime']]
    data = case.merge(data, on = 'subject_id', how = 'outer')
    data = data[~data['icd_code'].isnull()]
    data = data[data['chartdate'] < data['admittime']]
    data = data[['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version',
       'chartdate', 'date', 'new_subject_id', 'admittime', 'event_code']]
    return data

def extract_diag2(data, case):
    data = datechange(data)
    case = datechange(case)
    final_df = pd.DataFrame()

    case = case[['new_subject_id', 'subject_id', 'chartdate']]
    case.columns =['new_subject_id', 'subject_id', 'index_date']
    data = case.merge(data, on = 'subject_id', how = 'outer')
    data = data[~data['event_code'].isnull()]
    data = data[data['chartdate'] < data['index_date']]
    data = data[['subject_id', 'chartdate', 'event_code', 'date', 'new_subject_id']]
    return data

# To avoid include diagnostic code in the same encounter. 

def encounter_date(case, stay):
    case = datechange(case)
    case = case.drop(columns = 'admittime', axis= 1)
    d = 1
    final_df = pd.DataFrame()
    for f in case['new_subject_id'].unique():
        d += 1
        subject_id = case[case['new_subject_id'] == f]['subject_id'].iloc[0]
        index_date = case[case['new_subject_id'] == f]['chartdate'].iloc[0]
        temp = stay[stay['subject_id'] == subject_id]
        temp = temp[temp['admittime'] <= pd.to_datetime(index_date)]
        temp['new_subject_id'] = f
        temp = temp.sort_values(by = 'admittime', ascending = False)
        temp = temp.drop_duplicates(subset = ['new_subject_id'])
        temp = temp[['new_subject_id', 'subject_id', 'hadm_id', 'admittime']]
        final_df = final_df.append(temp)
        if d%2000 == 0:
            print(d)
    final_df = case.merge(final_df, on = ['new_subject_id', 'subject_id'], how = 'left')
    temp = final_df[final_df['admittime'].isnull()]
    temp['admittime'] = temp['chartdate']
    temp2 = final_df[~final_df['admittime'].isnull()]
    
    final_df = temp.append(temp2)
    return final_df

def Age_cheker(case_demo):
    temp = case_demo[case_demo['event_code'].str.contains('AGE_', na=False)]
    return temp['event_code'].unique()


def sensitivity_mimic(data):
    
    extract_dict = {'Location': 'LOC_','Institution': 'INT_','Organism': 'ORG_', 'Specimen Group': 'CxGrp_', 'Specimen Source': 'CxSrc_'}
    sensitivity = ['AMIKACIN', 'AMPICILLIN', 'AMPICILLIN/SULBACTAM', 'CEFAZOLIN', 'CEFEPIME', 'CEFTAZIDIME', 'CEFTRIAXONE', 'CEFUROXIME', 'CIPROFLOXACIN',
       'CLINDAMYCIN', 'DAPTOMYCIN', 'ERYTHROMYCIN', 'GENTAMICIN', 'IMIPENEM','LEVOFLOXACIN', 'LINEZOLID', 'MEROPENEM', 'NITROFURANTOIN', 'OXACILLIN',
       'PENICILLIN G', 'PIPERACILLIN', 'PIPERACILLIN/TAZO', 'RIFAMPIN', 'TETRACYCLINE', 'TOBRAMYCIN', 'TRIMETHOPRIM/SULFA', 'VANCOMYCIN']

    sensitivity = ['amikacin_interprit', 'amoxicillin_interprit',
       'amoxicillin-clavulanate_interprit', 'ampicillin_interprit',
       'ceFAZolin_interprit', 'ampicillin-sulbactam_interprit',
       'aztreonam_interprit', 'cefTAZidime_interprit', 'cefTRIAXone_interprit',
       'cefazolin_interprit', 'DAPTOmycin_interprit', 'cefotetan_interprit',
       'ceftaroline_interprit', 'ceftazidime_interprit',
       'ceftizoxime_interprit', 'cefoxitin_interprit', 'cefotaxime_interprit',
       'cefuroxime_interprit', 'chloramphenicol_interprit',
       'ciprofloxacin_interprit', 'cefepime_interprit',
       'clindamycin_interprit', 'colistin_interprit', 'ceftriaxone_interprit',
       'dalfopristin-quinupristin_interprit', 'amphotericin B_interprit',
       'cefpodoxime_interprit']
    
    final_demo = pd.DataFrame()

    temp = data[['subject_id','storetime','test_name','spec_type_desc', 'org_name', 'chartdate']]
    temp.columns = ['subject_id','chartdate','Specimen Group','spec_type_desc','Organism', 'drawntime']
    temp['event_code'] = 'ORG_' + temp['Specimen Group']   + '$' + temp['Organism'] #'_' + temp['spec_type_desc']
    temp = temp[['subject_id','chartdate', 'event_code', 'drawntime']]
    final_demo = final_demo.append(temp)
    for f in sensitivity:
        temp = data[['subject_id','storetime','chartdate','test_name','spec_type_desc','org_name', f]]
        temp.columns = ['subject_id','chartdate','drawntime','Specimen Group','spec_type_desc','Organism', f]
        temp[f] = temp[f].fillna("0")
        temp['event_code'] = 'SENSI_' + temp['Specimen Group'] + '_' + temp['Organism'] + '_'+ f + '$' + temp[f]
        temp = temp[['subject_id','chartdate','drawntime' ,'event_code']] # Chart data = Store Time
        final_demo = final_demo.append(temp)

    return final_demo

def micro_order_mimic(data):
    data = data[~data['test_name'].str.contains('MRSA', na=True)]
    data = data[~data['test_name'].str.lower().str.contains('aureus')]
    data = data[~data['test_name'].str.lower().str.contains('r/o')]
    final_demo = pd.DataFrame()
    extract_dict = {'Location': 'LOC_','Institution': 'INT_','Organism': 'ORG_', 'test_name': 'CxGrp_', 'Specimen Source': 'CxSrc_'}
    extract = ['test_name']
    for f in extract:
        temp = data[['subject_id','charttime', f, 'storetime']]
        temp.columns = ['subject_id','chartdate', 'event_code', 'storetime']
        temp['event_code'] = extract_dict[f] + temp['event_code']
        final_demo = final_demo.append(temp)
    return final_demo

def lab_order_mimic(data):
    final_demo = pd.DataFrame()
    extract = ['label'] #loinc_code
    for f in extract:
        temp = data[['subject_id','charttime', f, 'storetime']]
        temp.columns = ['subject_id','chartdate', 'event_code', 'storetime']
        temp['event_code'] = "ORDER_" + temp['event_code']
        final_demo = final_demo.append(temp)
    return final_demo

def lab_result_mimic(data):
    final_demo = pd.DataFrame()
    extract_dict = {'Location': 'LOC_','Institution': 'INT_','Organism': 'ORG_', 'test_name': 'CxGrp_', 'Specimen Source': 'CxSrc_'}
    extract = ['label'] #'loinc_code',
    
    temp = data.copy()
    temp['int'] = ''
    temp['int'][temp['comments'].str.lower().str.contains('positive', na=False)]= 'Positive'
    temp['int'][temp['comments'].str.lower().str.contains('pos.', na=False)]= 'Positive'
    temp['int'][temp['comments'].str.lower().str.contains('pos*.', na=False)]= 'Positive'
    temp['int'][temp['comments'].str.lower().str.contains('negative', na=False)]= 'Negative'
    temp['int'][temp['comments'].str.lower().str.contains('neg.', na=False)]= 'Negative'   
    temp = temp[~temp['int'].isnull()]

    for f in extract:
        temp2 = temp[['subject_id','storetime', f, 'int', 'charttime', 'fluid']]
        temp2.columns = ['subject_id','chartdate', 'event_code', 'int', 'drawntime', 'fluid']
        temp2['event_code'] = "TEST_" + temp2['event_code']+ '_'+temp2['fluid'] +'_0_Result Type'  +'$' + temp2['int']
        final_demo = final_demo.append(temp2)
        print('first done')
    return final_demo        

def culture_result_mimic2(data):
    data = data.fillna('0')
    
    final_demo2 = pd.DataFrame()
    temp = data[['subject_id','chartdate','Order','spec_type_desc','mhh_org', 'charttime', 'storetime']]
    temp['event_code'] = 'ORG_' + temp['Order'] +temp['spec_type_desc'] + '$' + temp['mhh_org']
    final_demo2 = final_demo2.append(temp)        
    temp = data[['subject_id','chartdate','Order','spec_type_desc','mhh_org', 'charttime', 'storetime', 'result']]
    temp['event_code'] = 'TEST_' + temp['Order'] +'_' + temp['spec_type_desc'] + '_' + temp['mhh_org'] + '_Result Type' + '$' + temp['result']
    final_demo2 = final_demo2.append(temp)
    
    return final_demo2

def extract_order_result2(data, case):
    data = datechange(data)
    case = datechange(case)
    final_df = pd.DataFrame()

    case = case[['new_subject_id', 'subject_id', 'chartdate']]
    case.columns =['new_subject_id', 'subject_id', 'index_date']
    data = case.merge(data, on = 'subject_id', how = 'outer')
    data = data[~data['event_code'].isnull()]
    data = data[data['chartdate'] < data['index_date']]
    data = data[['subject_id', 'drawntime', 'event_code', 'date', 'new_subject_id']]
    data.columns = ['subject_id', 'chartdate', 'event_code', 'date', 'new_subject_id']
    return data

def extract_order(data, case, hr):
    data = datechange(data)
    final_df = pd.DataFrame()
    
    case = case[['new_subject_id', 'subject_id', 'chartdate']]
    case.columns =['new_subject_id', 'subject_id', 'index_date']
    data = case.merge(data, on = 'subject_id', how = 'left')
    data = data[~data['event_code'].isnull()]
    data = data[data['chartdate'] <= data['index_date']+  pd.Timedelta(hr, 'Hour')]
    data = data[['subject_id', 'chartdate', 'event_code', 'date', 'new_subject_id']]
    
    return data

def extract_order_result_micro(data, case):
    data = datechange(data)
    case = datechange(case)
    data['storetime'] = data['storetime'].astype('datetime64')
    final_df = pd.DataFrame()

    case = case[['new_subject_id', 'subject_id', 'chartdate']]
    case.columns =['new_subject_id', 'subject_id', 'index_date']
    data = case.merge(data, on = 'subject_id', how = 'outer')
    data = data[~data['event_code'].isnull()]
    data = data[data['storetime'] < data['index_date']]
    data = data[['subject_id', 'charttime', 'event_code', 'date', 'new_subject_id']]
    data.columns = ['subject_id', 'chartdate', 'event_code', 'date', 'new_subject_id']
    return data

def add_demo(case):
    case_age = case.merge(patients, on = 'subject_id', how = 'left')
    case_age['anchor_year'] = case_age['anchor_year'].astype(int)
    case_age['anchor_age'] = case_age['anchor_age'].astype(int)
    case_age['year_diff'] = case_age['chartdate'].dt.year -case_age['anchor_year']
    case_age['age'] = case_age['anchor_age'] + case_age['year_diff']
    case_age['age'] = case_age['age'].apply(round_half_up)
    case_age['age'] = case_age['age'].astype(int)
    case_age['event_code'] = 'AGE_' + case_age['age'].astype(str)
    case_gen = case_age.copy()
    case_gen['event_code'] = 'GENDER_' + case_gen['gender']
    case_age = case_age.append(case_gen)
    
    return case_age

def culture_type(case):
    case2 = pd.merge(case, mimic_micro_map, how = 'left', left_on='Order', right_on='label')
    case2 = case2.drop('new_subject_id', axis=1)
    case2['Order'] = case2['Order_y']
    case2['event_code'] = 'ORDER_' + case2['Order']
    case2_order = extract_diag2(case2, case)
    case2_order.drop_duplicates(inplace=True)
    return case2_order

def check_terms(data, term):
    return data[data['event_code'].isin(term)]

def ICD_restructure(data, dict_df):
    data2 = check_terms(data, dict_df[0].unique())
    icd10 = data[data['event_code'].str.contains('ICD10')]
    icd9 = data[data['event_code'].str.contains('ICD9')]
    icd10['event_code'] = icd10['event_code'].str[:11] + '.' + icd10['event_code'].str[11:]
    icd9['event_code'] = icd9['event_code'].str[:10] + '.' + icd9['event_code'].str[10:]
    icd9 = icd9.append(icd10)
    icd9 = check_terms(icd9, dict_df[0].unique())
    icd9 = icd9.append(data2)
    icd9.drop_duplicates(inplace=True)
    
    return icd9

def Proc_restructure(data, dict_df):
    data2 = check_terms(data, dict_df[0].unique())
    icd9 = data.copy()
    icd10 = data.copy()
    icd9['event_code'] = icd9['event_code'].str[:5] + '.' + icd9['event_code'].str[5:]
    icd10['event_code'] = icd10['event_code'].str[:4] + '.' + icd10['event_code'].str[4:]
    icd0 = icd9.append(icd10)
    icd9 = check_terms(icd9, dict_df[0].unique())
    icd9 = icd9.append(data2)
    icd9.drop_duplicates(inplace=True)
    
    return icd9

# Create random split: 70:20:30 for case and control data
# We do not want to include same patient with different events in the same sets
# This step has to be here.
def split(total_list):

    dataSize = len(total_list['subject_id'].unique())
    ind = np.random.permutation(dataSize)
    nTest = int(0.2 * dataSize)
    nValid = int(0.1 * dataSize)
    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest+nValid]
    train_indices = ind[nTest+nValid:]
    c = 0
    dict_patient = {}
    for f in total_list['subject_id'].unique():
        dict_patient[c] = f
        c+=1
    patient_df = pd.DataFrame(dict_patient.items())
    test_mrns = patient_df[patient_df[0].isin(test_indices)]
    valid_mrns = patient_df[patient_df[0].isin(valid_indices)]
    train_mrns = patient_df[patient_df[0].isin(train_indices)]
    
    return train_mrns, valid_mrns, test_mrns


