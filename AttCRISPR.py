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
def get_spearman(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat,x_test_seq])
    return sp.stats.spearmanr(y_test, y_test_pred)[0]  
def get_score_at_test(model):
    y_test_pred = model.predict([x_test_onehot,x_test_biofeat,x_test_seq])
    mse = mean_squared_error(y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]    
    r2 = r2_score(y_test, y_test_pred)
    return 'MES:' + str(mse),'Spearman:' + str(spearmanr) , 'r2:' + str(r2)



from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
params = {
    'train_batch_size':70,
    'cnn_fc_hidden_layer_num':1,
    'cnn_fc_hidden_layer_units_num':241,
    'cnn_fc_dropout':0.2753,
    'lstm_embedding_output':62,
    'lstm_embedding_dropout':0.4006,
    'lstm_unit_num':78,
    'lstm_dropout':0.2501,
    'lstm_recurrent_dropout':0.3719,
    'lstm_fc_hidden_layer_num':1,
    'lstm_fc_hidden_layer_units_num':100,
    'lstm_fc_dropout':0.25,
    'bio_fc_hidden_layer_num':1,
    'bio_fc_hidden_layer_units_num':100,
    'bio_fc_dropout':0.25
    }
params_range = {
    'train_batch_size_min':40,
    'train_batch_size_max':100,
    'cnn_fc_hidden_layer_num_min':1,
    'cnn_fc_hidden_layer_num_max':5,
    'cnn_fc_hidden_layer_units_num_min':50,
    'cnn_fc_hidden_layer_units_num_max':300,
    'lstm_embedding_output_min':40,
    'lstm_embedding_output_max':80,
    'lstm_unit_num_min':50,
    'lstm_unit_num_max':100,
    'lstm_fc_hidden_layer_num_min':1,
    'lstm_fc_hidden_layer_num_max':5,
    'lstm_fc_hidden_layer_units_num_min':50,
    'lstm_fc_hidden_layer_units_num_max':300,
    'bio_fc_hidden_layer_num_min':0,
    'bio_fc_hidden_layer_num_max':4,
    'bio_fc_hidden_layer_units_num_min':50,
    'bio_fc_hidden_layer_units_num_max':300
    }
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


def mlp(inputs,output_layer_activation,output_dim,output_use_bias,
        hidden_layer_num,hidden_layer_units_num,hidden_layer_activation,dropout,
        name=''):
    for l in range(hidden_layer_num):
        x = Dense(hidden_layer_units_num, activation=hidden_layer_activation)(inputs)
        x = Dropout(dropout)(x)
    d = Dense(output_dim,activation=output_layer_activation
              ,kernel_regularizer='l2',use_bias=output_use_bias)
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
    embedding_layer = Embedding(7,params['lstm_embedding_output'],input_length=21)
    embedded = embedding_layer(inputs)
    embedded = SpatialDropout1D(params['lstm_embedding_dropout'])(embedded)
    #(?,21,units)
    lstm_output = LSTM(params['lstm_unit_num'], dropout=params['lstm_dropout'],recurrent_dropout=params['lstm_recurrent_dropout'],
                kernel_regularizer='l2',recurrent_regularizer='l2',
                return_sequences=True,return_state=False)(embedded)
    return lstm_output    

def model():
    onehot_input = Input(name = 'onehot_input', shape = (21,4, 1,))
    biological_input = Input(name = 'bio_input', shape = (x_train_biofeat.shape[1],))
    sequence_input = Input(name = 'seq_input', shape = (21,))
    ######CNN######
    cnn_output = cnn(onehot_input)
    onehot_embedded = mlp(cnn_output,output_layer_activation='sigmoid',output_dim=21,output_use_bias=False,
            hidden_layer_num=params['cnn_fc_hidden_layer_num'],hidden_layer_units_num=params['cnn_fc_hidden_layer_units_num'],
            hidden_layer_activation='relu',dropout=params['cnn_fc_dropout'],
            name='cnn_embedding')
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
                                      output_layer_activation='tanh',output_dim=1,output_use_bias=True,
                                      hidden_layer_num=params['lstm_fc_hidden_layer_num'],hidden_layer_units_num=params['lstm_fc_hidden_layer_units_num'],
                                      hidden_layer_activation='relu',dropout=params['lstm_fc_dropout'],
                                      name='lstm_output_at_'+str(i))
    lstm_embedded = keras.layers.concatenate(time_lstm_embeddedat,name='lstm_embedding')
    x = dot([lstm_embedded,onehot_embedded],axes=-1,name='position_score')
    ######Biofeat######
    x_bio = mlp(biological_input,
                output_layer_activation='sigmoid',output_dim=1,output_use_bias=True,
                hidden_layer_num=params['bio_fc_hidden_layer_num'],hidden_layer_units_num=params['bio_fc_hidden_layer_units_num'],
                hidden_layer_activation='relu',dropout=params['bio_fc_dropout'],
                name='biofeat_embedding')

    output = dot([x,x_bio],axes=-1,name='score')
    model = Model(inputs=[onehot_input, biological_input,sequence_input],
                 outputs=[output])
    return model

def train(epochs=25,learning_rate=0.001):
    m = model()
    np.random.seed(1337)
    batch_size = params['train_batch_size']
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
    #m.save('./hellbat_50.h5')
    sp = get_spearman(m)
    return {'loss': -1*sp, 'status': STATUS_OK}

def train_with(hyperparameters):
    uniform2int = lambda t,min,max: int(min+(max-min)*t)
    params['train_batch_size'] = uniform2int(hyperparameters['train_batch_size'],
                                             params_range['train_batch_size_min'],
                                             params_range['train_batch_size_max'])
    params['cnn_fc_hidden_layer_num'] = uniform2int(hyperparameters['cnn_fc_hidden_layer_num'],
                                                    params_range['cnn_fc_hidden_layer_num_min'],
                                                    params_range['cnn_fc_hidden_layer_num_max'])
    params['cnn_fc_hidden_layer_units_num'] = uniform2int(hyperparameters['cnn_fc_hidden_layer_units_num'],
                                                          params_range['cnn_fc_hidden_layer_units_num_min'],
                                                          params_range['cnn_fc_hidden_layer_units_num_max'])
    params['cnn_fc_dropout'] = hyperparameters['cnn_fc_dropout']
    params['lstm_embedding_output'] = uniform2int(hyperparameters['lstm_embedding_output'],
                                                  params_range['lstm_embedding_output_min'],
                                                  params_range['lstm_embedding_output_max'])
    params['lstm_embedding_dropout'] = hyperparameters['lstm_embedding_dropout']
    params['lstm_unit_num'] = uniform2int(hyperparameters['lstm_unit_num'],
                                          params_range['lstm_unit_num_min'],
                                          params_range['lstm_unit_num_max'])
    params['lstm_dropout'] = hyperparameters['lstm_dropout']
    params['lstm_recurrent_dropout'] = hyperparameters['lstm_recurrent_dropout']
    params['lstm_fc_hidden_layer_num'] = uniform2int(hyperparameters['lstm_fc_hidden_layer_num'],
                                                     params_range['lstm_fc_hidden_layer_num_min'],
                                                     params_range['lstm_fc_hidden_layer_num_max'])
    params['lstm_fc_hidden_layer_units_num'] = uniform2int(hyperparameters['lstm_fc_hidden_layer_units_num'],
                                                           params_range['lstm_fc_hidden_layer_units_num_min'],
                                                           params_range['lstm_fc_hidden_layer_units_num_max'])
    
    params['lstm_fc_dropout'] = hyperparameters['lstm_fc_dropout']
    params['bio_fc_hidden_layer_num'] = uniform2int(hyperparameters['bio_fc_hidden_layer_num'],
                                                    params_range['bio_fc_hidden_layer_num_min'],
                                                    params_range['bio_fc_hidden_layer_num_max'])
    params['bio_fc_hidden_layer_units_num'] = uniform2int(hyperparameters['bio_fc_hidden_layer_units_num'],
                                                          params_range['bio_fc_hidden_layer_units_num_min'],
                                                          params_range['bio_fc_hidden_layer_units_num_max'])
    params['lstm_fc_dropout'] = hyperparameters['lstm_fc_dropout']
    train()
if __name__=='__main__':
    train()
