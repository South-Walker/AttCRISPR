import math
import pickle
import numpy as np
######
pkl = open('alldata.pkl','rb')
x_onehot =  pickle.load(pkl)
x_biofeat = pickle.load(pkl)
y = pickle.load(pkl)
x_seq = pickle.load(pkl)

best = -1
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
random_state=40
test_size = 0.15
validate_size = 0.1
x_train_validate_onehot, x_test_onehot, y_train_validate, y_test = train_test_split(x_onehot, y, test_size=test_size, random_state=random_state)
x_train_validate_biofeat, x_test_biofeat, y_train_validate, y_test = train_test_split(x_biofeat, y, test_size=test_size, random_state=random_state)
x_train_onehot, x_validate_onehot,y_train, y_validate = train_test_split(x_train_validate_onehot, y_train_validate, test_size=validate_size, random_state=random_state)
x_train_biofeat, x_validate_biofeat,y_train, y_validate = train_test_split(x_train_validate_biofeat, y_train_validate, test_size=validate_size, random_state=random_state)
def get_spearman(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat])
    return sp.stats.spearmanr(y_test, y_test_pred)[0]  
def get_score_at_test(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat])
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]    
    r2 = r2_score(y_test, y_test_pred)
    global best
    if best<spearmanr:
        best = spearmanr
        model.save('./bestCNN.h5')
        print('save')
    return 'MES:' + str(mse),'Spearman:' + str(spearmanr) , 'r2:' + str(r2), 'best:' + str(best)




from random import randint
import copy
l_x_train_onehot = x_train_onehot.tolist()
l_x_train_biofeat = x_train_biofeat.tolist()
l_y_train = y_train.tolist()

x=0
for i in range(len(l_x_train_onehot)):
    if randint(0,100) < 32:
        j = randint(4,16)
        for k in range(4):
            if l_x_train_onehot[i][j][k][0] == 1:
                x = k
        offset = randint(1,4)
        noise_x = (x+offset)%4
        newx = copy.deepcopy(l_x_train_onehot[i])
        newx[j][x][0] = 0.9
        newx[j][noise_x][0] = 0.1
        l_x_train_onehot.append(newx)
        l_x_train_biofeat.append(copy.deepcopy(l_x_train_biofeat[i]))
        l_y_train.append(copy.deepcopy(l_y_train[i]))
t_x_train_onehot = np.array(l_x_train_onehot).reshape(-1,21,4,1)
t_x_train_biofeat = np.array(l_x_train_biofeat).reshape(-1,11)
t_y_train = np.array(l_y_train).reshape(-1)

index=np.arange(len(l_x_train_onehot))
np.random.shuffle(index)
x_train_onehot=t_x_train_onehot[index,:,:,:]
x_train_biofeat=t_x_train_biofeat[index,:] 
y_train=t_y_train[index]
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
params = {
    'train_batch_size':16,
    'train_epochs_num':150,
    'train_base_learning_rate':0.00002,
    'cnn_fc_hidden_layer_num':2,
    'cnn_fc_hidden_layer_units_num':421,
    'cnn_fc_dropout':0.5,
    'cnn_filters_num':50,
    'cnn_conv_num':3,
    'rnn_embedding_output':75,
    'rnn_embedding_dropout':0.5,
    'rnn_unit_num':64,
    'rnn_dropout':0.5,
    'rnn_recurrent_dropout':0.5,
    'rnn_fc_hidden_layer_num':2,
    'rnn_fc_hidden_layer_units_num':135,
    'rnn_fc_dropout':0.5,
    'bio_fc_hidden_layer_num':1,
    'bio_fc_hidden_layer_units_num':40,
    'bio_fc_dropout':0.5
    }
params_range = {
    'cnn_fc_hidden_layer_units_num':[188,488],
    'cnn_filters_num':[5,45],
    'rnn_embedding_output':[20,100],
    'rnn_unit_num':[4,184],
    'rnn_fc_hidden_layer_units_num':[19,259],
    'bio_fc_hidden_layer_units_num':[16,116]
}
import keras
from keras.preprocessing import text,sequence
from keras.layers import Softmax,Input, Dense, Conv2D, Flatten, BatchNormalization,Multiply,Cropping1D,dot,merge, Embedding, Bidirectional,RepeatVector
from keras.layers.core import *
from keras.models import *
from keras.layers.recurrent import LSTM,GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback,LambdaCallback,LearningRateScheduler
from keras.optimizers import *
import keras.backend as K
import scipy as sp
import pandas as pd


initializer_dict = {'1':'lecun_uniform','2':'normal', '3':'he_normal', '0':'he_uniform'}
optimizer_dict = {'1':SGD,'2':RMSprop, '3':Adagrad, '4':Adadelta,'5':Adam,'6':Adamax,'0':Nadam}

optimizer = optimizer_dict['5']


def mlp(inputs,output_layer_activation,output_dim,output_use_bias,
        hidden_layer_num,hidden_layer_units_num,hidden_layer_activation,dropout,
        name=None,output_regularizer=None):
    if output_layer_activation == 'sigmoid' or output_layer_activation == 'tanh':
        hidden_layer_num-=1
    x = inputs
    for l in range(hidden_layer_num):
        x = Dense(hidden_layer_units_num, activation=hidden_layer_activation)(inputs)
        x = Dropout(dropout)(x)
    if output_layer_activation == 'sigmoid' or output_layer_activation == 'tanh':
        x = Dense(hidden_layer_units_num)(x)
        
        x = keras.layers.concatenate([x,inputs])
        x = Activation(hidden_layer_activation)(x)
        x = Dense(output_dim,use_bias=output_use_bias,
                  kernel_regularizer='l2',activity_regularizer=output_regularizer)(x)
        x = Activation(output_layer_activation,name=name)(x)
        return x
    x = Dense(output_dim,activation=output_layer_activation,
              kernel_regularizer='l2',activity_regularizer=output_regularizer,
              use_bias=output_use_bias,name=name)(x)
    return x

def cnn(inputs):
    convs = []
    for i in range(params['cnn_conv_num']):
        convs.append(
            Conv2D(params['cnn_filters_num'],(i+2,4),strides=(1,4),padding='same',activation='relu')(inputs)
            )
    conv_output = keras.layers.concatenate(convs,name='conv_output')
    pooling_output = conv_output
    cnn_output = Flatten()(pooling_output)
    return cnn_output
def model():
    onehot_input = Input(name = 'onehot_input', shape = (21,4, 1,))
    biological_input = Input(name = 'bio_input', shape = (x_train_biofeat.shape[1],))
    ######CNN######
    cnn_output = cnn(onehot_input)
    onehot_embedded = mlp(cnn_output,output_layer_activation='tanh',output_dim=21,output_use_bias=False,
                          hidden_layer_num=params['cnn_fc_hidden_layer_num'],hidden_layer_units_num=params['cnn_fc_hidden_layer_units_num'],
                          hidden_layer_activation='relu',dropout=params['cnn_fc_dropout'],
                          name='cnn_embedding')
    x_cnn = mlp(onehot_embedded,
            output_layer_activation='linear',output_dim=1,output_use_bias=False,
            hidden_layer_num=0,hidden_layer_units_num=0,
            hidden_layer_activation='relu',dropout=0)
    model = Model(inputs=[onehot_input, biological_input],
                 outputs=[x_cnn],name='cnn')
    return model

def train():
    m = model()
    np.random.seed(1337)
    batch_size = params['train_batch_size']
    learningrate = params['train_base_learning_rate']
    epochs = params['train_epochs_num']
    batch_end_callback = LambdaCallback(
        on_epoch_end=lambda batch,logs: print(get_score_at_test(m))
        )
    learningrate_scheduler = LearningRateScheduler(
        schedule=lambda epoch : 
        learningrate if epoch<epochs*3/5 else (learningrate*0.5 if epoch < epochs*4/5 else learningrate*0.25)
        )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    m.compile(loss='mse', optimizer=optimizer(lr=learningrate))
    m.fit([x_train_onehot,x_train_biofeat], 
                 y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_data=([x_validate_onehot,x_validate_biofeat],y_validate),
                 callbacks=[batch_end_callback])

    m.save('./conv_'+str(epochs)+'.h5')
    sp = get_spearman(m)
    return {'loss': -1*sp, 'status': STATUS_OK}

def train_with(hyperparameters):
    name2params = lambda name: int(
        hyperparameters[name]*(params_range[name][1]-params_range[name][0])
        +params_range[name][0]
        )
    params['bio_fc_hidden_layer_units_num'] = name2params('bio_fc_hidden_layer_units_num')
    return train()
train()