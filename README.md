# Predictive Modeling on Electronic Health Records(EHR) using Pytorch
***************** 

**Overview**

Although there are plenty of repos on vision and NLP models, there are very limited repos on EHR using deep learning that we can find. Here we open source our repo, implementing data preprocessing, data loading, and a zoo of common RNN models. The main goal is to lower the bar of entering this field for researchers. We are not claiming any state-of-the-art performance, though our models are quite competitive (a paper describing our work will be available soon).  

Based on existing works (e.g., Dr. AI and RETAIN), we represent electronic health records (EHRs) using the pickled list of list of list, which contain histories of patients' diagnoses, medications, and other various events. We integrated all relevant information of a patient's history, allowing easy subsetting.

Currently, this repo includes the following predictive models: Vanilla RNN, GRU, LSTM, Bidirectional RNN, Bidirectional GRU, Bidirectional LSTM, Dilated RNN, Dilated GRU, Dilated LSTM, QRNN,and T-LSTM to analyze and predict clinical performaces. Additionally we have tutorials comparing perfomance to plain LR, Random Forest. 

**Pipeline**

![pipeline](https://github.com/ZhiGroup/pytorch_ehr/blob/MasterUpdateJun2019/tutorials/Pipeline%20for%20data%20flow.png)


**Paper Reference**
* The [paper](RNNisAllyouNeedv01.docx) upon which this repo was built. (to-do: include paper link)


**Folder Organization**
* [ehr_pytorch](https://github.com/ZhiGroup/pytorch_ehr/blob/MasterUpdateJun2019/ehr_pytorch): main folder with modularized components:
    * EHREmb.py: EHR embeddings
    * EHRDataloader.py: a separate module to allow for creating batch preprocessed data with multiple functionalities including sorting on visit length and shuffle batches before feeding.
    * Models.py: multiple different models
    * Utils.py
    * main.py: main execution file
    * tplstm.py: tplstm package file
* Data
    * toy.train: pickle file of  toy data with the same structure (multi-level lists) of our processed Cerner data, can be directly utilized for our models for demonstration purpose;
* Preprocessing
    * [data_preprocessing_v1.py](https://github.com/ZhiGroup/pytorch_ehr/blob/MasterUpdateJun2019/Preprocessing/data_preprocessing_v1.py): preprocess the data from dataset to build the required multi-level input structure
      (clear description of how to run this file is in its document header)
* [Tutorials](https://github.com/ZhiGroup/pytorch_ehr/tree/MasterUpdateJun2019/tutorials)
    * RNN_tutorials_toy.ipynb: jupyter notebooks with examples on how to run our models with visuals and/or utilize our dataloader as a standalone;
    * HF prediction for Diabetic Patients Pre and Post Diabetes.ipynb
    * Early Readmission v2.ipynb
* trained_models examples:
    * hf.trainEHRmodel.log: examples of the output of the model
    * hf.trainEHRmodel.pth: actual trained model
    * hf.trainEHRmodel.st: state dictionary

**Data Structure**

*  We followed the data structure used in the RETAIN. Encounters may include pharmacy, clinical and microbiology laboratory, admission, and billing information from affiliated patient care locations. All admissions, medication orders and dispensing, laboratory orders, and specimens are date and time stamped, providing a temporal relationship between treatment patterns and clinical information.These clinical data are mapped to the most common standards, for example, diagnoses and procedures are mapped to the International Classification of Diseases (ICD) codes, medimultications information include the national drug codes (NDCs), and laboratory tests are linked to their LOINIC codes.


*  Our processed pickle data: multi-level lists. From most outmost to gradually inside (assume we have loaded them as X)
    * Outmost level: patients level, e.g. X[0] is the records for patient indexed 0
    * 2nd level: patient information indicated in X[0][0], X[0][1], X[0][2] are patient id, disease status (1: yes, 0: no disease), and records
    * 3rd level: a list of length of total visits. Each element will be an element of two lists (as indicated in 4)
    * 4th level: for each row in the 3rd-level list. 
        *  1st element, e.g. X[0][2][0][0] is list of visit_time (since last time)
        *  2nd element, e.g. X[0][2][0][1] is a list of codes corresponding to a single visit
    * 5th level: either a visit_time, or a single code
*  An illustration of the data structure is shown below: 

![data structure](https://github.com/ZhiGroup/pytorch_ehr/blob/MasterUpdateJun2019/tutorials/Data%20structure%20with%20explanation.png)

In the implementation, the medical codes are tokenized with a unified dictionary for all patients.
![data example](https://github.com/ZhiGroup/pytorch_ehr/blob/MasterUpdateJun2019/tutorials/data.png)
* Notes: as long as you have multi-level list you can use our EHRdataloader to generate batch data and feed them to your model

**Paper Reference**

The [paper](https://github.com/ZhiGroup/pytorch_ehr/blob/MasterUpdateJun2019/Medinfo2019_PA_SimpleRNNisAllweNeed.pdf) upon which this repo was built. 

**Dependencies**
* [Pytorch 0.4.0] (http://pytorch.org)
* [Torchqrnn] (https://github.com/salesforce/pytorch-qrnn)
* Pynvrtc
* sklearn
* Matplotlib (for visualizations)
* Python: 3.6+

**Usage**
* For preprocessing
 python data_preprocessing.py <Case File> <Control File> <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> 
The above case and control files each is just a three columns table like pt_id | medical_code | visit/event_date  

* To run our models, directly use (you don't need to separately run dataloader, everything can be specified in args here):
<pre>
python3 main.py -root_dir<'your folder that contains data file(s)'> -files<['filename(train)' 'filename(valid)' 'filename(test)']> -which_model<'RNN'> -optimizer<'adam'> ....(feed as many args as you please)
</pre>
* Example:

<pre>
python3.7 main.py -root_dir /.../Data/ -files sample.train sample.valid sample.test -input_size 15800 -batch_size 100 -which_model LR -lr 0.01 -eps 1e-06 -L2 1e-04
</pre>


* To singly use our dataloader for generating data batches, use:
<pre>
data = EHRdataFromPickles(root_dir = '../data/', 
                          file = ['toy.train'])
loader =  EHRdataLoader(data, batch_size = 128)
</pre>  
  #Note: If you want to split data, you must specify the ratios in EHRdataFromPickles()
         otherwise, call separate loaders for your seperate data files 
         If you want to shuffle batches before using them, add this line 
 <pre>
loader = iter_batch2(loader = loader, len(loader))
</pre>
otherwise, directly call 

<pre>
for i, batch in enumerate(loader): 
    #feed the batch to do things
</pre>

Check out this [notebook](https://github.com/ZhiGroup/pytorch_ehr/blob/master/tutorials/RNN_tutorials_toy.ipynb) with a step by step guide of how to utilize our package. 

**Warning**

* This repo is for research purpose. Using it at your own risk. 
* This repo is under GPL-v3 license. 

**Acknowledgements**
Hat-tip to:
* [DRNN github](https://github.com/zalandoresearch/pt-dilate-rnn)
* [QRNN github](https://github.com/salesforce/pytorch-qrnn)
* [T-LSTM paper](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf)


