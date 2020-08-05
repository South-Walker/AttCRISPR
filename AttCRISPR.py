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
    'train_batch_size':44,
    'train_epochs_num':50,
    'train_base_learning_rate':0.0001,
    'cnn_fc_hidden_layer_num':3,
    'cnn_fc_hidden_layer_units_num':140,
    'cnn_fc_dropout':0.2839,
    'cnn_filters_num':20,
    'rnn_embedding_output':60,
    'rnn_embedding_dropout':0.4872,
    'rnn_unit_num':80,
    'rnn_dropout':0.5608,
    'rnn_recurrent_dropout':0.4310,
    'rnn_fc_hidden_layer_num':2,
    'rnn_fc_hidden_layer_units_num':118,
    'rnn_fc_dropout':0.5868,
    'bio_fc_hidden_layer_num':0,
    'bio_fc_hidden_layer_units_num':70,
    'bio_fc_dropout':0.6433
    }
params_range = {
    'train_batch_size_min':40,
    'train_batch_size_max':100,
    'cnn_fc_hidden_layer_num_min':1,
    'cnn_fc_hidden_layer_num_max':5,
    'cnn_fc_hidden_layer_units_num_min':50,
    'cnn_fc_hidden_layer_units_num_max':300,
    'rnn_embedding_output_min':40,
    'rnn_embedding_output_max':120,
    'rnn_unit_num_min':50,
    'rnn_unit_num_max':120,
    'rnn_fc_hidden_layer_num_min':1,
    'rnn_fc_hidden_layer_num_max':5,
    'rnn_fc_hidden_layer_units_num_min':50,
    'rnn_fc_hidden_layer_units_num_max':300,
    'bio_fc_hidden_layer_num_min':0,
    'bio_fc_hidden_layer_num_max':4,
    'bio_fc_hidden_layer_units_num_min':20,
    'bio_fc_hidden_layer_units_num_max':120
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


initializer_dict = {'1':'lecun_uniform','2':'normal', '3':'he_normal', '0':'he_uniform'}
optimizer_dict = {'1':SGD,'2':RMSprop, '3':Adagrad, '4':Adadelta,'5':Adam,'6':Adamax,'0':Nadam}

optimizer = optimizer_dict['5']


def mlp(inputs,output_layer_activation,output_dim,output_use_bias,
        hidden_layer_num,hidden_layer_units_num,hidden_layer_activation,dropout,
        name=None,output_regularizer=None):
    x = inputs
    for l in range(hidden_layer_num):
        x = Dense(hidden_layer_units_num, activation=hidden_layer_activation)(inputs)
        x = Dropout(dropout)(x)
    if output_layer_activation == 'sigmoid' or output_layer_activation == 'tanh':
        x = Dense(output_dim,use_bias=output_use_bias,
                  kernel_regularizer='l2',activity_regularizer=output_regularizer)(x)
        x = Activation(output_layer_activation,name=name)(BatchNormalization()(x))
        return x
    x = Dense(output_dim,activation=output_layer_activation,
              kernel_regularizer='l2',activity_regularizer=output_regularizer,
              use_bias=output_use_bias,name=name)(x)
    return x

def cnn(inputs):
    conv_1 = Conv2D(params['cnn_filters_num'], (2, 4), padding='same', activation='relu')(inputs)
    conv_2 = Conv2D(params['cnn_filters_num'], (3, 4), padding='same', activation='relu')(inputs)
    conv_3 = Conv2D(params['cnn_filters_num'], (4, 4), padding='same', activation='relu')(inputs)
    conv_output = keras.layers.concatenate([ conv_1, conv_2, conv_3],name='conv_output')
    conv_output = BatchNormalization(name='cnn_batchnormal')(conv_output)
    maxpooling_output = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,4), padding='valid')(conv_output)
    avgpooling_output = keras.layers.AvgPool2D(pool_size=(2, 2), strides=(1,4), padding='valid')(conv_output)
    pooling_output = keras.layers.concatenate([maxpooling_output,avgpooling_output],name='pooling_output')
    cnn_output = Flatten()(pooling_output)
    return cnn_output
def rnn(inputs):
    embedding_layer = Embedding(21,params['rnn_embedding_output'],input_length=21)
    embedded = embedding_layer(inputs)
    embedded = SpatialDropout1D(params['rnn_embedding_dropout'])(embedded)
    #(?,21,units)
    rnn_output = GRU(params['rnn_unit_num'],dropout=params['rnn_dropout'],recurrent_dropout=params['rnn_recurrent_dropout'],
                     kernel_regularizer='l2',recurrent_regularizer='l2',
                     return_sequences=True,return_state=False,name='rnn_output')(embedded)
    return rnn_output    

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
    ######RNN######
    rnn_output = rnn(sequence_input)
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

    output = dot([x,x_bio],axes=-1,name='score')
    #output=x
    model = Model(inputs=[onehot_input, biological_input,sequence_input],
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
        learningrate if epoch<epochs*2/5 else (learningrate*0.1 if epoch < epochs*4/5 else learningrate*0.01)
        )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    m.compile(loss='mse', optimizer=optimizer(lr=learningrate))
    m.fit([x_train_onehot,x_train_biofeat,x_train_seq], 
                 y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_split=0.1,
                 callbacks=[batch_end_callback,early_stopping,learningrate_scheduler])

    m.save('./conv_'+str(epochs)+'.h5')
    sp = get_spearman(m)
    return {'loss': -1*sp, 'status': STATUS_OK}

def train_with(hyperparameters):
    uniform2int = lambda t,min,max: int(min+(max-min)*t)
    params['cnn_filters_num'] = hyperparameters['cnn_filters_num']
    params['cnn_fc_hidden_layer_num'] = hyperparameters['cnn_fc_hidden_layer_num']
    params['cnn_fc_hidden_layer_units_num'] = uniform2int(hyperparameters['cnn_fc_hidden_layer_units_num'],
                                                          params_range['cnn_fc_hidden_layer_units_num_min'],
                                                          params_range['cnn_fc_hidden_layer_units_num_max'])
    params['rnn_embedding_output'] = uniform2int(hyperparameters['rnn_embedding_output'],
                                                  params_range['rnn_embedding_output_min'],
                                                  params_range['rnn_embedding_output_max'])
    params['rnn_unit_num'] = uniform2int(hyperparameters['rnn_unit_num'],
                                          params_range['rnn_unit_num_min'],
                                          params_range['rnn_unit_num_max'])
    params['rnn_fc_hidden_layer_num'] = hyperparameters['rnn_fc_hidden_layer_num']
    params['rnn_fc_hidden_layer_units_num'] = uniform2int(hyperparameters['rnn_fc_hidden_layer_units_num'],
                                                           params_range['rnn_fc_hidden_layer_units_num_min'],
                                                           params_range['rnn_fc_hidden_layer_units_num_max'])
    
    params['bio_fc_hidden_layer_num'] = hyperparameters['bio_fc_hidden_layer_num']
    params['bio_fc_hidden_layer_units_num'] = uniform2int(hyperparameters['bio_fc_hidden_layer_units_num'],
                                                          params_range['bio_fc_hidden_layer_units_num_min'],
                                                          params_range['bio_fc_hidden_layer_units_num_max'])
    return train()
if __name__=='__main__':
    train()
