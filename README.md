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

### Files description
* [Train/TrainCNN.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/TrainCNN.py) and 
[Train/TrainRNN.py](https://github.com/South-Walker/AttCRISPR/blob/master/Train/TrainRNN.py), 
contain the code of the model's architecture in spatial and temporal domain respectively.