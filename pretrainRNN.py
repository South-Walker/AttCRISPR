import math
import pickle
import numpy as np
######
pkl = open('alldata.pkl','rb')
x_onehot =  pickle.load(pkl)
x_biofeat = pickle.load(pkl)
y = pickle.load(pkl)
x_seq = pickle.load(pkl)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

random_state=40
test_size = 0.15

x_train_onehot, x_test_onehot, y_train, y_test = train_test_split(x_onehot, y, test_size=test_size, random_state=random_state)
x_train_biofeat, x_test_biofeat, y_train, y_test = train_test_split(x_biofeat, y, test_size=test_size, random_state=random_state)
def get_spearman(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat])
    return sp.stats.spearmanr(y_test, y_test_pred)[0]  
def get_score_at_test(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat])
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]    
    r2 = r2_score(y_test, y_test_pred)
    return 'MES:' + str(mse),'Spearman:' + str(spearmanr) , 'r2:' + str(r2)



from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
params = {
    'train_batch_size':1024,
    'train_epochs_num':200,
    'train_base_learning_rate':0.001,
    'rnn_fc_dropout':0.5868,
    #'rnn_embedding_dropout':0.4872,
    #'rnn_dropout':0.5608,
    'bio_fc_dropout':0.6433,
    #'rnn_recurrent_dropout':0.4310,
    
    'embedding_output':48,
    'rnn_head_unit_num':94,
    'rnn_tail_unit_num':400,

    'rnn_fc_hidden_layer_num':2,
    'rnn_fc_hidden_layer_units_num':300,
    'bio_fc_hidden_layer_num':1,
    'bio_fc_hidden_layer_units_num':67
    }
params_range = {
    'embedding_output':[20,500],
    'rnn_head_unit_num':[20,500],
    'rnn_tail_unit_num':[100,1000],
}

import keras
from keras.preprocessing import text,sequence
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization,Multiply,Cropping1D,dot,merge, Embedding, Bidirectional,RepeatVector
from keras.layers.core import *
from keras.models import *
from keras.layers.recurrent import LSTM,GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback,LambdaCallback,LearningRateScheduler
from keras.optimizers import *
import keras.backend as K
import scipy as sp
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

def rnn(inputs):
    embedded = Conv2D(params['embedding_output'], (1, 4),strides=(1,4), padding='Valid', activation=None)(inputs)
    embedded = Reshape((21,params['embedding_output'],))(embedded)
    #(?,21,units)
    gru_layer = GRU(params['rnn_head_unit_num'],return_sequences=True,return_state=False)
    gru_layer = Bidirectional(gru_layer,merge_mode='sum',name='bgru_output')
    rnn_output = gru_layer(embedded)
    rnn_output = keras.layers.concatenate([rnn_output,embedded])
    last_gru_layer = GRU(params['rnn_tail_unit_num'],dropout=0.25,recurrent_dropout=0.25,
                     kernel_regularizer='l2',recurrent_regularizer='l2',
                     return_sequences=True,return_state=False,name='last_gru_output')
    last_gru_layer = Bidirectional(last_gru_layer,merge_mode='sum',name='last_bgru_output')
    rnn_output = last_gru_layer(rnn_output)
    return rnn_output    

def model():
    onehot_input = Input(name = 'onehot_input', shape = (21,4, 1,))
    biological_input = Input(name = 'bio_input', shape = (x_train_biofeat.shape[1],))
    ######RNN######
    rnn_output = rnn(onehot_input)
    ######Attention######
    time_rnn_embeddedat = []
    for i in range(21):
        time_rnn_embeddedat.append(
            Flatten(name='rnn_flatten_'+str(i))(
                Cropping1D(cropping=(i,21-1-i))(rnn_output)
                )
            )
        time_rnn_embeddedat[i] = mlp(time_rnn_embeddedat[i],
                                     output_layer_activation='tanh',output_dim=1,output_use_bias=False,
                                     hidden_layer_num=params['rnn_fc_hidden_layer_num'],hidden_layer_units_num=params['rnn_fc_hidden_layer_units_num'],
                                     hidden_layer_activation='relu',dropout=params['rnn_fc_dropout'],
                                     name='rnn_output_at_'+str(i))
    rnn_embedded = keras.layers.concatenate(time_rnn_embeddedat,name='rnn_embedding')
    x = mlp(rnn_embedded,
                output_layer_activation=None,output_dim=1,output_use_bias=True,
                hidden_layer_num=0,hidden_layer_units_num=0,
                hidden_layer_activation=None,dropout=0)

    ######Biofeat######
    x_bio = mlp(biological_input,
                output_layer_activation='sigmoid',output_dim=1,output_use_bias=True,
                hidden_layer_num=params['bio_fc_hidden_layer_num'],hidden_layer_units_num=params['bio_fc_hidden_layer_units_num'],
                hidden_layer_activation='relu',dropout=params['bio_fc_dropout'],
                name='biofeat_embedding')
    output = keras.layers.concatenate([x,x_bio])
    output = mlp(output,
                output_layer_activation='linear',output_dim=1,output_use_bias=True,
                hidden_layer_num=0,hidden_layer_units_num=0,
                hidden_layer_activation='relu',dropout=0)
    #output=x
    model = Model(inputs=[onehot_input, biological_input],
                 outputs=[output])
    output_model = Model(inputs=[onehot_input, biological_input],
                 outputs=[rnn_output])
    return model,output_model

def train():
    m,output_model = model()
    np.random.seed(1337)
    batch_size = params['train_batch_size']
    learningrate = params['train_base_learning_rate']
    epochs = params['train_epochs_num']
    batch_end_callback = LambdaCallback(
        on_epoch_end=lambda batch,logs: print(get_score_at_test(m))
        )
    learningrate_scheduler = LearningRateScheduler(
        schedule=lambda epoch : 
        learningrate if epoch<epochs*2/5 else (learningrate*0.5 if epoch < epochs*4/5 else learningrate*0.25)
        )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    m.compile(loss='mse', optimizer=optimizer(lr=learningrate))
    m.fit([x_train_onehot,x_train_biofeat], 
                 y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_split=0.1,
                 callbacks=[batch_end_callback,learningrate_scheduler])

    output_model.save('./pretrain_'+str(epochs)+'.h5')
    sp = get_spearman(m)
    return {'loss': -1*sp, 'status': STATUS_OK}

def train_with(hyperparameters):
    name2params = lambda name: int(
        hyperparameters[name]*(params_range[name][1]-params_range[name][0])
        +params_range[name][0]
        )
    params['embedding_output'] = name2params('embedding_output')
    params['rnn_head_unit_num'] = name2params('rnn_head_unit_num')
    params['rnn_tail_unit_num'] = name2params('rnn_tail_unit_num')

    return train()
if __name__=='__main__':
    train()

