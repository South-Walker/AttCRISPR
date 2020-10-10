import keras
from keras.preprocessing import text,sequence
from keras.layers import Input, BatchNormalization, Softmax
from keras.layers.core import *
from keras.models import *
from keras.callbacks import Callback,LambdaCallback
from keras.optimizers import *
import keras.backend as K
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from LearnUtil import *

def model(params,cnn_trainable=False,rnn_trainable=False,load_weight=False):
    onehot_input = Input(name='onehot_input', shape = (21,4, 1,))
    biological_input = Input(name='bio_input', shape = (11,))
    cnnmodel = load_model(params['cnn_load_file'])
    rnnmodel = load_model(params['rnn_load_file'])
    cnnmodel.trainable = cnn_trainable
    rnnmodel.trainable = rnn_trainable
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
    output = Dense(units=1,kernel_initializer=keras.initializers.RandomNormal(mean=0.3, stddev=0.05),
                   use_bias=False,bias_initializer='zero',name='last_weight_avg')(output)
    model = Model(inputs=[onehot_input, biological_input],
                 outputs=[output])
    if load_weight:
        model.load_weights(params['ensemble_load_file']) 
    return model

def train(params,
          train_input,train_biofeat,train_label,
          validate_input,validate_biofeat,validate_label,
          test_input,test_biofeat,test_label,
          cnn_trainable=False,rnn_trainable=False,load_weight=False):
    result = Result()
    m = model(params,
              cnn_trainable=cnn_trainable,rnn_trainable=rnn_trainable,load_weight=load_weight)
    batch_size = params['train_batch_size']
    learningrate = params['train_base_learning_rate']
    epochs = params['train_epochs_num']
    m.compile(loss='mse', optimizer=Adam(lr=learningrate))
    
    batch_end_callback = LambdaCallback(on_epoch_end=
                                        lambda batch,logs: 
                                        print(get_score_at_test(m,[test_input,test_biofeat],result,test_label,
                                                                issave=True,savepath=params['ensemble_save_file'])))
    m.fit([train_input,train_biofeat],train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=([validate_input,validate_biofeat],validate_label),
          callbacks=[batch_end_callback])

    weight = m.get_layer('last_weight_avg').get_weights()
    print(weight)
    return {'loss': -1*result.Best, 'status': STATUS_OK}


