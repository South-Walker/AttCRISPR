# AttCRISPR
### OS requirement
The code was tested on Linux. 
Although without testing, it can also run on Windows (or other OS) in theory.
> #### Note:
> - We use tensorflow as the backend of Keras for testing.
> - ViennaRNA is a C code library for prediction of RNA secondary structure, 
needs to be downloaded before installation. 

The required software/packages are:
* python=3.6.11
* numpy=1.19.1
* scipy=1.5.2
* tensorflow=1.14.0
* keras=2.2.4
* scikit-learn=0.23.2
* biopython=1.71 
* viennarna=2.4.5
* pandas=1.1.0
* hyperopt

### Demo
If you want to train the standard AttCRISPR from scratch one way is to run as follow
```bash
python AttCRISPRDemo.py WT
```
where WT can be replaced by ESP and SP, which means using the corresponding data set (wild-type SpCas9, enhanced SpCas9 and Cas9-High Fidelity, respectively). 
This work will take a lot of time, and that is why we suggest you customise your own script and run it in parallel. 

### Main files and directories description
* [Train/TrainCNN.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/TrainCNN.py) and 
[Train/TrainRNN.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/TrainRNN.py), 
contain the code of the model's architecture in spatial and temporal domain respectively.
* [Train/Ensemble.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/Ensemble.py) 
contain the code of the ensemble model of spatial and temporal (with/without hand-crafted biological features).
* [Train/Pipeline.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/Pipeline.py) 
contain the code of the complete pipeline of training the standard AttCRISPR.  
* [Train/WTConst.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/WTConst.py), 
[Train/ESPConst.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/ESPConst.py) and 
[Train/SPConst.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/SPConst.py), 
the detailed parameters of the corresponding dataset.
* [Validation.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/Validation.py) contain the code of validation.
* [Util](https://github.com/South-Walker/AttCRISPR/tree/master/Util) Tools to separate source data into different pkl.