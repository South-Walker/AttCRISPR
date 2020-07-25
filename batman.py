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

random_state=40
test_size = 0.15

x_train_onehot, x_test_onehot, y_train, y_test = train_test_split(x_onehot, y, test_size=test_size, random_state=random_state)
x_train_biofeat, x_test_biofeat, y_train, y_test = train_test_split(x_biofeat, y, test_size=test_size, random_state=random_state)
x_train_seq, x_test_seq, y_train, y_test = train_test_split(x_seq, y, test_size=test_size, random_state=random_state)

def get_score_at_test(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat,x_test_seq])
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]    
    r2 = r2_score(y_test, y_test_pred)
    return 'MES:' + str(mse),'Spearman:' + str(spearmanr) , 'r2:' + str(r2)



from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import keras
from keras.preprocessing import text,sequence
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization,Multiply,Cropping1D,dot,merge, Embedding, Bidirectional,RepeatVector
from keras.layers.core import *
from keras.models import *
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback,LambdaCallback
from keras.optimizers import *
import keras.backend as K
import scipy as sp
import pandas as pd


fc_activation_dict = {'1':'relu','2':'tanh', '3':'sigmoid', '4':'hard_sigmoid', '0':'elu'}
initializer_dict = {'1':'lecun_uniform','2':'normal', '3':'he_normal', '0':'he_uniform'}
optimizer_dict = {'1':SGD,'2':RMSprop, '3':Adagrad, '4':Adadelta,'5':Adam,'6':Adamax,'0':Nadam}

fc_activation = fc_activation_dict['3']
optimizer = optimizer_dict['6']


def mlp(inputs,output_layer_activation,output_dim,
        hidden_layer_num,hidden_layer_units_num,hidden_layer_activation,dropout,
        name=''):
    for l in range(hidden_layer_num):
        x = Dense(hidden_layer_units_num, activation=hidden_layer_activation)(inputs)
        x = Dropout(dropout)(x)
    d = Dense(output_dim,activation=output_layer_activation
              ,kernel_regularizer='l2',use_bias=False)
    if name != '':
        d.name = name
    x = d(x)
    if output_layer_activation == 'sigmoid' or output_layer_activation == 'tanh':
        x = BatchNormalization()(x)
    return x

def cnn(inputs):
    conv_1 = Conv2D(10, (1, 4), padding='same', activation='relu')(inputs)
    conv_2 = Conv2D(10, (2, 4), padding='same', activation='relu')(inputs)
    conv_3 = Conv2D(10, (3, 4), padding='same', activation='relu')(inputs)
    conv_4 = Conv2D(10, (4, 4), padding='same', activation='relu')(inputs)
    conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])
    bn_output = BatchNormalization()(conv_output)
    pooling_output = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,4), padding='valid')(bn_output)
    cnn_output = Flatten()(pooling_output)
    return cnn_output
def lstm(inputs):
    embedding_layer = Embedding(7,62,input_length=21)
    embedded = embedding_layer(inputs)
    embedded = SpatialDropout1D(0.4006)(embedded)
    #(?,21,units)
    lstm_output = LSTM(78, dropout=0.2501,recurrent_dropout=0.3719,
                kernel_regularizer='l2',recurrent_regularizer='l2',
                return_sequences=True,return_state=False)(embedded)
    return lstm_output
def lstm_model():
    onehot_embedded = RepeatVector(62)(onehot_embedded)
    onehot_embedded = Lambda(lambda inp: K.permute_dimensions(inp,(0,2,1)),name='onehot')(onehot_embedded)
    x = Multiply(name='pospoint')([embedded,onehot_embedded])

    x = Flatten()(x)

    

def model(batch_size=90, epochs=400,learning_rate=0.001):
    onehot_input = Input(name = 'onehot_input', shape = (21, 4, 1,))
    biological_input = Input(name = 'bio_input', shape = (x_train_biofeat.shape[1],))
    sequence_input = Input(name = 'seq_input', shape = (21,))
    ######CNN######
    cnn_output = cnn(onehot_input)
    onehot_embedded = mlp(cnn_output,output_layer_activation='tanh',output_dim=21,
            hidden_layer_num=1,hidden_layer_units_num=241,
            hidden_layer_activation='relu',dropout=0.2753,
            name='cnn_output')
    ######LSTM######
    lstm_output = lstm(sequence_input)
    ######Attention######
    time_lstm_embeddedat = []
    for i in range(21):
        time_lstm_embeddedat.append(
            Flatten()(
                Cropping1D(cropping=(i,21-1-i))(lstm_output)
                )
            )
        time_lstm_embeddedat[i] = mlp(time_lstm_embeddedat[i],
                                      output_layer_activation='tanh',output_dim=21,
                                      hidden_layer_num=1,hidden_layer_units_num=100,
                                      hidden_layer_activation='relu',dropout=0.25,
                                      name='lstm_output_'+str(i))
        time_lstm_embeddedat[i] = dot([time_lstm_embeddedat[i],onehot_embedded],axes=-1)

    x = keras.layers.concatenate(time_lstm_embeddedat)
    ######Biofeat######
    #可能分层输入更好？
    x = keras.layers.concatenate([x, biological_input])
    
    output = mlp(x,output_layer_activation='sigmoid',output_dim=1,
            hidden_layer_num=2,hidden_layer_units_num=174,
            hidden_layer_activation=fc_activation,dropout=0.3988)
    model = Model(inputs=[onehot_input, biological_input,sequence_input],
                 outputs=[output])
    return model

def train(batch_size=90,epochs=200,learning_rate=0.0001):
    m = model()
    np.random.seed(1337)

    batch_end_callback = LambdaCallback(
        on_epoch_end=lambda batch,logs: print(get_score_at_test(m))
        )

    #batch_end_print_callback2 = LambdaCallback(on_epoch_end=lambda batch,logs: print(list(intermediate_layer_model.predict([x_test_onehot,x_test_biofeat,x_test_seq]))))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    m.compile(loss='mse', optimizer=optimizer(lr=learning_rate))
    m.fit([x_train_onehot,x_train_biofeat,x_train_seq], 
                 y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_split=0.1,
                 callbacks=[batch_end_callback])    
    m.save('./batman_200.h5')
    sp = get_score_at_test(m)
    return {'loss': -sp, 'status': STATUS_OK}


train()

