import keras
from keras.layers import Softmax,GlobalAveragePooling1D,Input, Conv2D, Flatten, BatchNormalization, Multiply,Cropping1D,dot, Bidirectional
from keras.layers.core import *
from keras.layers.recurrent import LSTM,GRU
from keras.models import *
from keras.callbacks import EarlyStopping, Callback,LambdaCallback
from keras.optimizers import *
import keras.backend as K
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from LearnUtil import *

def model(params):
    onehot_input = Input(name = 'onehot_input', shape = (21,4, 1,))
    embedded = Conv2D(params['rnn_embedding_output'], (1, 4),strides=(1,4), padding='Valid', activation=None)(onehot_input)
    embedded = Reshape((21,params['rnn_embedding_output'],))(embedded)

    ######encoder&decoder######
    encoder = LSTM(params['rnn_unit_num'],return_sequences=True,return_state=False,unroll=True)
    encoder = Bidirectional(encoder,merge_mode='sum',name='encoder')
    encoder_output = encoder(embedded)

    decoder_input = keras.layers.concatenate([embedded,encoder_output])
    decoder = LSTM(params['rnn_unit_num'],dropout=0.25,recurrent_dropout=0.25,
                     return_sequences=True,return_state=True,unroll=True)
    decoder = Bidirectional(decoder,merge_mode='sum',name='decoder')
    decoder_output,_,c_1,_,c_2 = decoder(decoder_input)
    

    encoderat = []
    decoderat = []
    for i in range(21):
        encoderat.append(
            Flatten()(Cropping1D(cropping=(i,21-1-i))(encoder_output)))
        decoderat.append(
            Flatten()(Cropping1D(cropping=(i,21-1-i))(decoder_output)))

    #context = Lambda(lambda inp: inp[0]+inp[1])([c_1,c_2])
    context = GlobalAveragePooling1D()(encoder_output)
    for i in range(21):
        decoderat[i] = Dense(params['rnn_unit_num'], kernel_initializer='zeros', activation=None,use_bias=False)(decoderat[i])
    
    ######attention######
    aat = []
    for i in range(21):
        atat = []
        for j in range(21):
            atat.append(
                dot([encoderat[j],decoderat[i]],axes=-1)
                )
        at = keras.layers.concatenate(atat)
        at = Softmax()(at)
        aat.append(Reshape((21,1,),name='temporal_attention_'+str(i))(at))
    rnn_output = []
    for i in range(21):
        weightavg = Lambda(lambda inp: inp[0]*inp[1])([encoder_output ,aat[i]])
        weightavg = Lambda(lambda inp: K.sum(inp,axis=-2,keepdims=False))(weightavg)
        rnn_output.append(keras.layers.concatenate([context,weightavg,decoderat[i]]))
    time_rnn_embeddedat = rnn_output 
    for i in range(21):
        time_rnn_embeddedat[i] = mlp(time_rnn_embeddedat[i],
                                     output_layer_activation='tanh',output_dim=1,output_use_bias=False,
                                     hidden_layer_num=params['rnn_fc_hidden_layer_num'],hidden_layer_units_num=params['rnn_fc_hidden_layer_units_num'],
                                     hidden_layer_activation='relu',dropout=0,
                                     name='rnn_output_at_'+str(i))
    rnn_embedded = keras.layers.concatenate(time_rnn_embeddedat,name='temporal_pos_score')
    #magic
    rnn_embedded = Dropout(rate=0.05)(rnn_embedded)
    output = Dense(units=1,kernel_regularizer='l2',name='temporal_score')(rnn_embedded)
    model = Model(inputs=[onehot_input],
                 outputs=[output],name='rnn')
    return model

def train(params,train_input,train_label,validate_input,validate_label,test_input,test_label):
    result = Result()
    m = model(params)
    batch_size = params['train_batch_size']
    learningrate = params['train_base_learning_rate']
    epochs = params['train_epochs_num']
    m.compile(loss='mse', optimizer=Adam(lr=learningrate))

    batch_end_callback = LambdaCallback(on_epoch_end=
                                        lambda batch,logs: 
                                        print(get_score_at_test(m,test_input,result,test_label,
                                                                issave=True,savepath=params['rnn_save_file'])))

    m.fit(train_input,train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=([validate_input],validate_label), 
          callbacks=[batch_end_callback])
    return {'loss': -1*result.Best, 'status': STATUS_OK}
