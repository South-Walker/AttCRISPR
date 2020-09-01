import math
import pickle
import numpy as np
######
pkl = open('alldata.pkl','rb')
x_onehot =  pickle.load(pkl)
x_biofeat = pickle.load(pkl)
y = pickle.load(pkl)
x_seq = pickle.load(pkl)


from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
random_state=40
test_size = 0.15

x_train_onehot, x_test_onehot, y_train, y_test = train_test_split(x_onehot, y, test_size=test_size, random_state=random_state)
x_train_biofeat, x_test_biofeat, y_train, y_test = train_test_split(x_biofeat, y, test_size=test_size, random_state=random_state)

best = -1

def get_spearman(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat])
    return sp.stats.spearmanr(y_test, y_test_pred)[0]  
def get_score_at_test(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat])
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]    
    r2 = r2_score(y_test, y_test_pred)
    global best
    best = spearmanr if spearmanr > best else best
    return 'MES:' + str(mse),'Spearman:' + str(spearmanr) , 'r2:' + str(r2), 'best:' + str(best)



from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
params = {
    'train_batch_size':44,
    'train_epochs_num':3,
    'train_base_learning_rate':0.0001,
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
def rnn(inputs):
    embedded = Conv2D(params['rnn_embedding_output'], (1, 4),strides=(1,4), padding='Valid', activation=None)(inputs)
    embedded = Reshape((21,params['rnn_embedding_output'],))(embedded)
    #(?,21,units)
    encoder = LSTM(params['rnn_unit_num'],return_sequences=True,return_state=False,unroll=True)
    encoder = Bidirectional(encoder,merge_mode='sum',name='encoder_output')
    encoder_output = encoder(embedded)
    
    encoderat = []
    for i in range(21):
        encoderat.append(
            Flatten()(
                Cropping1D(cropping=(i,21-1-i))(encoder_output)
                )
            )
    decoder_input = keras.layers.concatenate([embedded,encoder_output])
    decoder = LSTM(params['rnn_unit_num'],dropout=0.25,recurrent_dropout=0.25,
                     return_sequences=True,return_state=False,name='last_gru_output',unroll=True)
    decoder = Bidirectional(decoder,merge_mode='sum',name='last_bgru_output')
    decoder_output = decoder(decoder_input)
    
    decoderat = []
    for i in range(21):
        decoderat.append(
            Flatten()(
                Cropping1D(cropping=(i,21-1-i))(decoder_output)
                )
            )

    for i in range(21):
        decoderat[i] = Dense(params['rnn_unit_num'], activation=None,use_bias=False)(decoderat[i])

    aat = []
    for i in range(21):
        atat = []
        for j in range(21):
            atat.append(
                Softmax(axis=-1)(dot([encoderat[j],decoderat[i]],axes=-1))
                )
        aat.append(
            Reshape((21,1,))(keras.layers.concatenate(atat))
            )
        
    ctat = []
    for i in range(21):
        weightavg = Lambda(lambda inp: inp[0]*inp[1])([encoder_output ,aat[i]])
        weightavg = Lambda(lambda inp: K.sum(inp,axis=-2,keepdims=False))(weightavg)
        ctat.append(
            Reshape((1,params['rnn_unit_num'],))(weightavg)
            )
    rnn_output = keras.layers.concatenate(ctat,axis=-2)
    rnn_output = keras.layers.concatenate([rnn_output,decoder_output])
    return rnn_output    

def model():
    onehot_input = Input(name = 'onehot_input', shape = (21,4, 1,))
    biological_input = Input(name = 'bio_input', shape = (x_train_biofeat.shape[1],))
    ######CNN######
    cnn_output = cnn(onehot_input)
    onehot_embedded = mlp(cnn_output,output_layer_activation='sigmoid',output_dim=21,output_use_bias=False,
                          hidden_layer_num=params['cnn_fc_hidden_layer_num'],hidden_layer_units_num=params['cnn_fc_hidden_layer_units_num'],
                          hidden_layer_activation='relu',dropout=params['cnn_fc_dropout'],
                          name='cnn_embedding')
    ######RNN######
    from keras.models import load_model
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
    x = dot([rnn_embedded,onehot_embedded],axes=-1,name='position_score')
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
    global best
    best = -1
    m.fit([x_train_onehot,x_train_biofeat], 
                 y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_split=0.1,
                 callbacks=[batch_end_callback])

    m.save('./conv_'+str(epochs)+'.h5')
    sp = best
    print('best:'+str(best))
    return {'loss': -1*sp, 'status': STATUS_OK}

def train_with(hyperparameters):
    name2params = lambda name: int(
        hyperparameters[name]*(params_range[name][1]-params_range[name][0])
        +params_range[name][0]
        )
    params['bio_fc_hidden_layer_units_num'] = name2params('bio_fc_hidden_layer_units_num')
    params['cnn_fc_hidden_layer_units_num'] = name2params('cnn_fc_hidden_layer_units_num')
    params['cnn_filters_num'] = name2params('cnn_filters_num')
    params['rnn_fc_hidden_layer_units_num'] = name2params('rnn_fc_hidden_layer_units_num')
    params['rnn_unit_num'] = name2params('rnn_unit_num')
    params['rnn_embedding_output'] = name2params('rnn_embedding_output')
    return train()
train()