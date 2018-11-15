# Pytorch_ehr
***************** 

**Overview**
* Predictive analytics of risk onset on Cerner Electronic Health Records(EHR) using Pytorch library;
* Cerner EHR: 15815 unique medical codes. Full cohort with >1,000,000 records. Detailed info see  **Data Description** section below.
* Models built: Vanilla RNN, GRU, LSTM, Bidirectional RNN, Bidirectional GRU, Bidirectional LSTM, Dilated RNN, Dilated GRU, Dilated LSTM, QRNN, T-LSTM, GRU-Logistic Regression(GRU-LR), LR with embedding, plain LR, Random Forest;


**Folder Organization**
* ehr_pytorch: main folder with modularized components for all models, data loading and processing, and training, validation and test of models, main file for parsing arguments, and a EHRDataloader;
* 1. EHRDataloader: a separate function to allow for utilizing pytorch dataloader child object to create preprocessed data batch for testing our models;
* data: sample processed (pickled) data from Cerner, can be directly utilized for dataloader, and then models
* Tutorials: jupyter notebooks with examples on how to utilize our dataloader and run our models with visuals
* Test: coming up soon. Shell commands to quickly test on our package functionalities
* Sphinx build documentations
* Sample results:(? keep or discard? prob discard) 


**Data Description**
* Cerner, with 15815 unique medical codes. Full cohort with >1,000,000 records 
format: pickled multil-level lists. From most outmost to gradually inside:  Outmost level: 2nd level:  3rd level: 4th level:  5th level 
code types & what they stand for: ?
some visuals of the what the data looks like: my print out tables) 
in the format of xx xx and xx 
format: 
code types & what they stand for: 
some visuals of the what the data looks like

**Paper Reference**
* The [paper]() upon which this repo was built. (include paper link)


## Prerequisites

* Pytorch library, <http://pytorch.org/> 


## Tests


* To try our dataloader, use:
<pre>
data = EHRdataFromPickles(root_dir = '../data/', 
                                      file = 'hf.train')
loader =  EHRdataLoader(data)
#if you want to shuffle batches before using them 
loader = iter_batch2(loader = loader, len(loader))
for i, batch in enumerate(loader): 
    #feed the batch to do things 
#otherwise, directly call 
for i, batch in enumerate(loader): 
    #feed the batch to do things
</pre>

* To run our models, use:
<pre>
python main.py  --root_dire<your folder that contains data file>  --file <filename> --which_model <'RNN'>  --optimizer<'adam'>....
</pre>


## Authors

* See the list of [contributors]( https://github.com/ZhiGroup/pytorch_ehr/graphs/contributors)
* For development related requests [Contact chocolocked](https://github.com/chocolocked)

## Acknowledgements

Hat-tip to:
* [DRNN github](https://github.com/zalandoresearch/pt-dilate-rnn)
* [QRNN github](https://github.com/salesforce/pytorch-qrnn)
* [T-LSTM paper](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf)



