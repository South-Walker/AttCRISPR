import math
import numpy as np
import pandas as pd

from ESPConst import Params as params

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.model_selection import train_test_split
def ZScore(biofeat):
    for i in range(len(biofeat[0])):
        sum = 0
        for j in range(len(biofeat)):
            sum += biofeat[j][i]
        avg = sum / len(biofeat)
        sumx_2 = 0
        for j in range(len(biofeat)):
            sumx_2 += (biofeat[j][i]-avg)*(biofeat[j][i]-avg)
        sigma = math.sqrt(sumx_2 / len(biofeat))
        for j in range(len(biofeat)):
            biofeat[j][i] = (biofeat[j][i]-avg) / sigma

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
    ZScore(x_biofeat)
    label = pickle.load(pkl)
    x_seq = pickle.load(pkl)
    return SplitData(x_onehot,x_biofeat,label)

def AddNoise(onehot,biofeat,label,rate=50,intensity=0.30):
    from random import randint
    import copy
    t_onehot = onehot.tolist()
    t_biofeat = biofeat.tolist()
    t_label = label.tolist()
    x=0
    for i in range(len(onehot)):
        if randint(0,100) >= rate:
            continue
        j = randint(4,16)
        for k in range(4):
            if onehot[i][j][k][0] == 1:
               x = k
        offset = randint(1,4)
        noise = intensity/3
        newx = copy.deepcopy(onehot[i])
        for k in range(4):
            newx[j][k][0] = (1-intensity) if x==k else noise
        t_onehot.append(newx)
        t_biofeat.append(copy.deepcopy(t_biofeat[i]))
        t_label.append(copy.deepcopy(t_label[i]))


    index=np.arange(len(t_onehot))
    np.random.shuffle(index)
    
    t_onehot = np.array(t_onehot).reshape(-1,21,4,1)
    t_biofeat = np.array(t_biofeat).reshape(-1,11)
    t_label = np.array(t_label).reshape(-1)

    onehot=t_onehot[index,:,:,:]
    biofeat=t_biofeat[index,:] 
    label=t_label[index]

    return onehot,biofeat,label