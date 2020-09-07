import math
import numpy as np
import pandas as pd

from WTConst import Params as params

from sklearn.model_selection import train_test_split
def SplitData(onehot,biofeat,label):    
    random_state=40
    test_size = 0.15
    validate_size = 0.1
    x_train_validate_onehot, x_test_onehot, y_train_validate, y_test = train_test_split(onehot, label, test_size=test_size, random_state=random_state)
    x_train_validate_biofeat, x_test_biofeat, _, _ = train_test_split(biofeat, label, test_size=test_size, random_state=random_state)
    
    x_train_onehot, x_validate_onehot, y_train, y_validate = train_test_split(x_train_validate_onehot, y_train_validate, test_size=validate_size, random_state=random_state)
    x_train_biofeat, x_validate_biofeat, _, _ = train_test_split(x_train_validate_biofeat, y_train_validate, test_size=validate_size, random_state=random_state)
    data = {'input':
            {'train':{'onehot':x_train_onehot,'biofeat':x_train_biofeat},
             'validate':{'onehot':x_validate_onehot,'biofeat':x_validate_biofeat},
             'test':{'onehot':x_test_onehot,'biofeat':x_test_biofeat}
             },
            'label':
            {'train':y_train,
             'validate':y_validate,
             'test':y_test
             }
            }
    return data

import pickle
def Read_Data():
    pkl = open(params['data_file'],'rb')
    x_onehot = pickle.load(pkl)
    x_biofeat = pickle.load(pkl)
    label = pickle.load(pkl)
    x_seq = pickle.load(pkl)
    return SplitData(x_onehot,x_biofeat,label)
