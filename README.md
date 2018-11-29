# Pytorch_ehr
***************** 

**Overview**
* Predictive analytics of risk onset on Cerner Electronic Health Records(EHR) using Pytorch library;
* Cerner EHR: derived from > 600 Cerner implementation throughout the United States; contains clinical information for over 50 million unique patients with > 10 years of records. In total there are more than 110 million patient visits (encounters), 15815 unique medical codes. Detailed info see  **Data Description** section below.
* Models built: Vanilla RNN, GRU, LSTM, Bidirectional RNN, Bidirectional GRU, Bidirectional LSTM, Dilated RNN, Dilated GRU, Dilated LSTM, QRNN, T-LSTM, GRU-Logistic Regression(GRU-LR), LR with embedding, plain LR, Random Forest;


**Folder Organization**
* ehr_pytorch: main folder with modularized components for EHR embeddings, EHR dataloader(created from pytorch dataloader), models, utils, and training and evaluation of models, main file for excution;
  * EHRDataloader: a separate module to allow for creating batch preprocessed data with multiple functionalities including sorting on visit length and shuffle batches before feeding. If you don't want to use our models, you can use it as a standalone to process data specified in **Data Description**: basically multi-level list data in pickles 
* data: sample processed (pickled) data from Cerner, can be directly utilized for our models for demonstration purpose
* tutorials: jupyter notebooks with examples on how to utilize our dataloader and run our models with visuals
* test: coming up soon. Shell commands to quickly test on our package functionalities
* Sphinx build documentations: coming up soon


**Data Description**
* Cerner: derived from > 600 Cerner implementation throughout the United States; contains clinical information for over 50 million unique patients with > 10 years of records. In total there are more than 110 million patient visits (encounters), 15815 unique medical codes. Encounters may include pharmacy, clinical and microbiology laboratory, admission, and billing information from affiliated patient care locations. All admissions, medication orders and dispensing, laboratory orders, and specimens are date and time stamped, providing a temporal relationship between treatment patterns and clinical information.These clinical data are mapped to the most common standards, for example, diagnoses and procedures are mapped to the International Classification of Diseases (ICD) codes, medications information include the national drug codes (NDCs), and laboratory tests are linked to their LOINIC codes. 
* Processed pickle format: multil-level lists. From most outmost to gradually inside (assume we have loaded them as X)
  * Outmost level: patients level, e.g. X[0] is records for patient indexed 0
  * 2nd level: patient information indicated in X[0][0], X[0][1], X[0][2] are patient id, binary indicator of disease (1: yes, 0: no disease), and records
  * 3rd level: a list of length of total visits. Each element will be an element of two lists (as indicated in *4*) 
  * 4th level: for each row in the 3rd-level list, 1st element, e.g. X[0][2][0][0] is list of visit_time (since last time), 2nd, eg.e.g. X[0][2][1][1] is a list of codes corresponding to the visits
  * 5th level: either time, or the single code. 
    ![data structure visual](https://github.com/ZhiGroup/pytorch_ehr/blob/master/tutorials/Dataformat.png)
   * notes: as long as you have multi-level list you can use our EHRdataloader to generate batch data and feed them to your model


**Paper Reference**
* The [paper]() upon which this repo was built. (to-do: include paper link)


## Prerequisites

* Pytorch library, <http://pytorch.org/> 


## Usage

* To run our models, directly use (you don't need to separately use dataloader, everything can be specified in args here):
<pre>
python3 main.py -root_dir<'your folder that contains data file'> -file<'filename'> -which_model<'RNN'> -optimizer<'adam'> ....(feed as many args as you please)
</pre>

* To **singly** use our dataloader for generating data batches purpose, use:
<pre>
data = EHRdataFromPickles(root_dir = '../data/', 
                          file = 'hf.train')
loader =  EHRdataLoader(data)
#Note: if you want to split data, you must specify the ratios in EHRdataFromPickles()
      #otherwise, call separate loaders for your seperate data files

#if you want to shuffle batches before using them, add this line 
loader = iter_batch2(loader = loader, len(loader))

#otherwise, directly call 
for i, batch in enumerate(loader): 
    #feed the batch to do things
</pre>

- Check out this
[notebook](https://github.com/ZhiGroup/pytorch_ehr/edit/master/README.md) MODIFY with a step by step guide of how to utilize our package. 

## Authors

* See the list of [contributors]( https://github.com/ZhiGroup/pytorch_ehr/graphs/contributors)
* For development related requests [Contact](https://github.com/chocolocked)

## Acknowledgements

Hat-tip to:
* [DRNN github](https://github.com/zalandoresearch/pt-dilate-rnn)
* [QRNN github](https://github.com/salesforce/pytorch-qrnn)
* [T-LSTM paper](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf)



