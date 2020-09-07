import keras
from keras.preprocessing import text,sequence
from keras.layers import Input
from keras.layers.core import *
from keras.models import *
from keras.callbacks import Callback,LambdaCallback
from keras.optimizers import *
import keras.backend as K
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from LearnUtil import *

def model(params):
    onehot_input = Input(name='onehot_input', shape = (21,4, 1,))
    biological_input = Input(name='bio_input', shape = (11,))
    cnnmodel = load_model(params['cnn_load_file'])
    rnnmodel = load_model(params['rnn_load_file'])
    cnnmodel.trainable = False
    rnnmodel.trainable = False
    x_cnn = cnnmodel(onehot_input)
    x_rnn = rnnmodel(onehot_input)
    x = keras.layers.concatenate([x_rnn,x_cnn])
    ######Biofeat######
    x_bio = mlp(biological_input,
                output_layer_activation='tanh',output_dim=1,output_use_bias=True,
                hidden_layer_num=params['bio_fc_hidden_layer_num'],hidden_layer_units_num=params['bio_fc_hidden_layer_units_num'],
                hidden_layer_activation='relu',dropout=params['bio_fc_dropout'],
                name='biofeat_embedding')
    output = keras.layers.concatenate([x,x_bio])
    output = mlp(output,
                output_layer_activation='linear',output_dim=1,output_use_bias=True,
                hidden_layer_num=0,hidden_layer_units_num=0,
                hidden_layer_activation='relu',dropout=0,output_regularizer='l2')
    model = Model(inputs=[onehot_input, biological_input],
                 outputs=[output])
    return model

def train(params,
          train_input,train_biofeat,train_label,
          validate_input,validate_biofeat,validate_label,
          test_input,test_biofeat,test_label):
    global best
    best = -1
    m = model(params)
    batch_size = params['train_batch_size']
    learningrate = params['train_base_learning_rate']
    epochs = params['train_epochs_num']
    m.compile(loss='mse', optimizer=Adam(lr=learningrate))
    
    batch_end_callback = LambdaCallback(on_epoch_end=
                                        lambda batch,logs: 
                                        print(get_score_at_test(m,[test_input,test_biofeat],test_label,
                                                                issave=True,savepath=params['ensemble_save_file'])))
    m.fit([train_input,train_biofeat],train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=([validate_input,validate_biofeat],validate_label),
          callbacks=[batch_end_callback])
    return {'loss': -1*best, 'status': STATUS_OK}

