from keras.models import load_model
import math
import pickle
import numpy as np
from keras.models import *
import scipy as sp
from sklearn.model_selection import train_test_split

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

load_model = load_model('./sislowbatman_200.h5')
batch_end_print_callback = LambdaCallback(
    on_epoch_end=lambda batch,logs: print(getscore(load_model,y_test)))


intermediate_layer_model = Model(
        inputs=load_model.input, outputs=load_model.get_layer('cnn_output').output)
y = intermediate_layer_model.predict([x_train_onehot,x_train_biofeat,x_train_seq])
weights = [0 for x in range(0,21)]
for w in y:
    for i in range(21):
        weights[i] += w[i]
total = 0
for i in range(21):
    total += weights[i]
for i in range(21):
    weights[i] = weights[i] / total

print(weights)