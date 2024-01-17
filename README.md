```diff
@@ ## This Branch is dedicated to Clinicians -- please feel free to report an issue to help making this Repo more user friendly ## @@
```

## Steps to run:

### Environment Preparation:

   1. login to https://colab.research.google.com
   
   2. Select the GitHub option and enter this GitHub repo link (https://github.com/ZhiGroup/pytorch_ehr) and select the "Clinician" Branch 
   ![image](https://github.com/ZhiGroup/pytorch_ehr/assets/25290490/94281334-565c-4aa2-b0c7-8b00c7ce155e)

   
   3. click the arrow to open the Prepare_env notebook
   
   4. Run the file. It will display some messages; please select  "Run anyway" and follow the instructions to add the authorization code.
   
   5. As it completes successfully, you can see the pytorch_ehr drive created under your MyDrive:
   
      ![image](https://user-images.githubusercontent.com/25290490/127777065-79c66fd4-a488-4b80-844d-0f0e29f93f72.png)
  

      Now you are ready to enjoy the tutorial :)

### Data Preparation:

This year, we have 2 options:

1. Use the N3C SynthPuf data (only Demo, for code sharing on the N3C enclave Kindly email the presenters)
2. We will use SQLite to prepare our cohorts extracted from 100K Covid-19 patients synthetic data (  Synthea(TM) COVID-19 data:     Walonoski J, Klaus S, Granger E, Hall D, Gregorowicz A, Neyarapally G, Watson A, Eastman J. Synthea™ Novel coronavirus (COVID-19) model and synthetic data set. Intelligence-Based Medicine. 2020 Nov;1:100007. https://doi.org/10.1016/j.ibmed.2020.100007 )

   a. Download SQLite browser from https://sqlitebrowser.org/dl/
   
   b. Under the Data_Prep folder you can Download the dataprep.sql which describes the cohort definition and data extraction from the covid100k.db adapted from  https://synthea.mitre.org/ covid-19 100K data set.
    
   c. You will also find DataPrep.ipynb notebook which will guide you through the data preprocessing.


### Model Training:
   1. Go to https://drive.google.com/, navigate to Model_Training folder. You will find Model_Training.ipynb notebook which will guide you through the RNN model training 
   2. Model_explanation.ipynb will be used for the model explanation demo. 



## Predictive Modeling on Electronic Health Records(EHR) using Pytorch
***************** 

**Overview**

Although there are plenty of repos on vision and NLP models, there are very limited repos on EHR using deep learning that we can find. Here we open source our repo, implementing data preprocessing, data loading, and a zoo of common RNN models. The main goal is to lower the bar of entering this field for researchers. We are not claiming any state-of-the-art performance, though our models are quite competitive.  

Based on existing works (e.g., Dr. AI and RETAIN), we represent electronic health records (EHRs) using the pickled list of list of list, which contain histories of patients' diagnoses, medications, and other various events. We integrated all relevant information of a patient's history, allowing easy subsetting.

Currently, this repo includes the following predictive models: Vanilla RNN, GRU, LSTM, Bidirectional RNN, Bidirectional GRU, Bidirectional LSTM, Dilated RNN, Dilated GRU, Dilated LSTM, QRNN,and T-LSTM to analyze and predict clinical performaces. Additionally we have tutorials comparing perfomance to plain LR, Random Forest. 

**Pipeline**

![pipeline](https://github.com/ZhiGroup/pytorch_ehr/blob/master/tutorials/Pipeline%20for%20data%20flow.png)



**Data Structure**

*  We followed the data structure used in the RETAIN. Encounters may include pharmacy, clinical and microbiology laboratory, admission, and billing information from affiliated patient care locations. All admissions, medication orders and dispensing, laboratory orders, and specimens are date and time-stamped, providing a temporal relationship between treatment patterns and clinical information. These clinical data are mapped to the most common standards, for example, diagnoses and procedures are mapped to the International Classification of Diseases (ICD) codes, and laboratory tests are linked to their LOINIC codes.


*  Our processed pickle data: multi-level lists. From most outmost to gradually inside (assume we have loaded them as X)
    * Outmost level: patients level, e.g. X[0] is the records for patient indexed 0
    * 2nd level: patient information indicated in X[0][0], X[0][1], X[0][2] are patient id, outcome label or disease status (1: yes, 0: no disease), in case of survival it will be [disease status , time_to_disease], and visits records
    * 3rd level: a list of length of total visits. Each element will be an element of two lists (as indicated in 4)
    * 4th level: for each row in the 3rd-level list. 
        *  1st element, e.g. X[0][2][0][0] is list of visit_time (since last time)
        *  2nd element, e.g. X[0][2][0][1] is a list of codes corresponding to a single visit
    * 5th level: either a visit_time, or a single code
*  An illustration of the data structure is shown below: 

![data structure](https://github.com/ZhiGroup/pytorch_ehr/blob/master/tutorials/Data%20structure%20with%20explanation.png)

In the implementation, the medical codes are tokenized with a unified dictionary for all patients.
![data example](https://github.com/ZhiGroup/pytorch_ehr/blob/MasterUpdateJun2019/tutorials/data.png)
* Notes: as long as you have multi-level list you can use our EHRdataloader to generate batch data and feed them to your model

* How it works
![image](https://user-images.githubusercontent.com/25290490/127748409-f2e20a7f-16d9-4c46-856f-9aec7da8b737.png)

**Paper Reference**

Since we started our pytorch_ehr project a number of papers are published, for latest version kindly cite:
Rasmy L, Nigo M, Kannadath BS, Xie Z, Mao B, Patel K, Zhou Y, Zhang W, Ross A, Xu H, Zhi D. Recurrent neural network models (CovRNN) for predicting outcomes of patients with COVID-19 on admission to hospital: model development and validation using electronic health record data. The Lancet Digital Health. 2022 Jun 1;4(6):e415-25

**Versions**
This is towards Version 0.3, more details will be in the [release notes](https://github.com/ZhiGroup/pytorch_ehr_internal/releases/tag/v0.2-Feb20)

**Dependencies**
* [Pytorch 0.4.0] (http://pytorch.org) All models except the QRNN and T-LSTM are compatble with the latest pytorch version (verified)
* [Torchqrnn] (https://github.com/salesforce/pytorch-qrnn)
* Pynvrtc
* sklearn
* Matplotlib (for visualizations)
* tqdm
* Python: 3.6+

 

**License**

* This repo is for research purpose. Using it at your own risk. 
* This repo is under GPL-v3 license. 

**Acknowledgements**
Hat-tip to:
* [DRNN github](https://github.com/zalandoresearch/pt-dilate-rnn)
* [QRNN github](https://github.com/salesforce/pytorch-qrnn)
* [T-LSTM paper](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf)


