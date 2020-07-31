from keras.models import load_model
import math
import pickle
import numpy as np
from keras.models import *
import scipy as sp
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pkl = open('alldata.pkl','rb')
x_onehot =  pickle.load(pkl)
x_biofeat = pickle.load(pkl)
y = pickle.load(pkl)
x_seq = pickle.load(pkl)

random_state=40
test_size = 0.15
x_train_onehot, x_test_onehot, y_train, y_test = train_test_split(x_onehot, y, test_size=test_size, random_state=random_state)
x_train_biofeat, x_test_biofeat, y_train, y_test = train_test_split(x_biofeat, y, test_size=test_size, random_state=random_state)
x_train_seq, x_test_seq, y_train, y_test = train_test_split(x_seq, y, test_size=test_size, random_state=random_state)


def getscore(model,y):
    y_pre = model.predict([x_test_onehot,x_test_biofeat,x_test_seq]).reshape(-1)

    return float(sp.stats.spearmanr(y, y_pre)[0])

from keras.callbacks import Callback
from keras.callbacks import LambdaCallback
x_attention_onehot = x_train_onehot
x_attention_biofeat = x_train_biofeat
x_attention_seq = x_train_seq
y_attention = y_train
def get_spatial_attention(model,usebiofeat=True):
    spatial_layer_model = Model(
        inputs=load_model.input, outputs=model.get_layer('rnn_embedding').output)
    if usebiofeat:
        input = [x_attention_onehot,x_attention_biofeat,x_attention_seq]
    else:
        input = [x_attention_onehot,x_attention_seq]
    spatial_value = spatial_layer_model.predict(input)
    weights = [0 for x in range(0,21)]
    for w in range(len(spatial_value)):
        for i in range(21):
            weights[i] += spatial_value[w][i]*y_attention[w]
    total = 0
    for i in range(21):
        total += weights[i]
    for i in range(21):
        weights[i] = weights[i] / total
    return weights;
def get_temporal_attention(model,usebiofeat=True):
    weight = get_spatial_attention(model,usebiofeat);
    base_code_dict = {'T': 1, 'A': 2, 'C': 3, 'G': 4,'START': 0} 
    code_base_dict = {1: 'T', 2: 'A', 3: 'C', 4: 'G'}
    at_pos_temporal_weights_of = {'A':[0 for x in range(0,21)],'C':[0 for x in range(0,21)],
                                  'G':[0 for x in range(0,21)],'T':[0 for x in range(0,21)]}
    at_pos_count_of = {'A':[0 for x in range(0,21)],'C':[0 for x in range(0,21)],
                       'G':[0 for x in range(0,21)],'T':[0 for x in range(0,21)]}
    temporal_layer_model = Model(
        inputs=load_model.input, outputs=model.get_layer('cnn_embedding').output)
    if usebiofeat:
        input = [x_attention_onehot,x_attention_biofeat,x_attention_seq]
    else:
        input = [x_attention_onehot,x_attention_seq]
    temporal_value = temporal_layer_model.predict(input)
    for i in range(len(x_attention_seq)):
        for basepos in range(21):
            base = code_base_dict[x_attention_seq[i][basepos]]
            at_pos_count_of[base][basepos]+=1
            at_pos_temporal_weights_of[base][basepos]+=temporal_value[i][basepos]*y_attention[i]*weight[basepos]
    for k in at_pos_temporal_weights_of.keys():
        for basepos in range(21):
            at_pos_temporal_weights_of[k][basepos]/=max(at_pos_count_of[k][basepos],1)
    return at_pos_temporal_weights_of;
def map_temporal_attention_to_sum0(temporal_attention):
    for pos in range(21):
        total = 0
        nonzero = 0
        for k in temporal_attention.keys():
            total += temporal_attention[k][pos]
            if temporal_attention[k][pos] != 0:
                nonzero += 1
        offset = (total)/max(nonzero,1)
        for k in temporal_attention.keys():
            if temporal_attention[k][pos] != 0:
                temporal_attention[k][pos] -= offset

load_model = load_model('./new_500.h5')
batch_end_print_callback = LambdaCallback(
    on_epoch_end=lambda batch,logs: print(getscore(load_model,y_test)))

isusebiofeat = True
spatial_weights = get_spatial_attention(load_model,isusebiofeat)
temporal_weights = get_temporal_attention(load_model,isusebiofeat)
map_temporal_attention_to_sum0(temporal_weights)
print(spatial_weights)
print()
for k in temporal_weights.keys():
    print(temporal_weights[k])
print('end')