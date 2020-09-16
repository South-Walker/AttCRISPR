import keras
from keras.layers import Softmax,Input, Conv2D, Flatten, BatchNormalization, Multiply,Cropping1D,dot, Bidirectional
from keras.layers.core import *
from keras.layers.recurrent import LSTM,GRU
from keras.models import *
from keras.callbacks import EarlyStopping, Callback,LambdaCallback
from keras.optimizers import *
import keras.backend as K
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from LearnUtil import *
import math
def GaussianKernelBuffer(windowsize=8):
    gaussian = lambda d: math.exp( -(d*d)*2/windowsize/windowsize ) if abs(d)<=windowsize/2 else 0
    result = []
    for i in range(21):
        resultat = []
        for j in range(21):
            resultat.append(gaussian(i-j))
        nresultat = np.array(resultat)
        result.append(nresultat)
    for i in range(21):
        sum=0
        for j in range(21):
            sum+=result[i][j]
        time=21/sum
        for j in range(21):
            result[i][j]*=time
    return np.array(result)
def model(params):
    GaussianBuffer = GaussianKernelBuffer()
    onehot_input = Input(name = 'onehot_input', shape = (21,4, 1,))
    embedded = Conv2D(params['rnn_embedding_output'], (1, 4),strides=(1,4), padding='Valid', activation=None)(onehot_input)
    embedded = Reshape((21,params['rnn_embedding_output'],))(embedded)

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

    ######attention######
    aat = []
    for i in range(21):
        atat = []
        for j in range(21):
            align = dot([decoderat[j],encoderat[i]],axes=-1)
            atat.append(
                align
                )
        at = keras.layers.concatenate(atat)
        at = Softmax()(at)
        at = Reshape((1,21,))(at)
        aat.append(at)
    m = keras.layers.concatenate(aat,axis=-2)
    weight = Lambda(lambda inp: K.constant(GaussianBuffer)*inp ,name = 'temporal_attention')(m)
    weightavg = Lambda(lambda inp: K.batch_dot(inp[0],inp[1]),name='weight_avg')([weight,decoder_output])

    context = Lambda(lambda inp: K.sum(inp,axis=-2,keepdims=False))(encoder_output)
    context = Lambda(lambda inp: K.repeat(inp,21))(context)
    rnn_output = keras.layers.concatenate([context,weightavg])

    time_rnn_embeddedat = []
    for i in range(21):
        time_rnn_embeddedat.append(
            Flatten(name='rnn_flatten_'+str(i))(
                Cropping1D(cropping=(i,21-1-i))(rnn_output)
                )
            )
        time_rnn_embeddedat[i] = mlp(time_rnn_embeddedat[i],
                                     output_layer_activation='tanh',output_dim=1,output_use_bias=False,

                                     hidden_layer_num=1,hidden_layer_units_num=166,
                                     hidden_layer_activation='relu',dropout=0,
                                     name='rnn_output_at_'+str(i))
    rnn_embedded = keras.layers.concatenate(time_rnn_embeddedat,name='rnn_embedding')
    rnn_embedded = Dropout(rate=0.05)(rnn_embedded)
    x_rnn = mlp(rnn_embedded,
            output_layer_activation='linear',output_dim=1,output_use_bias=False,
            hidden_layer_num=0,hidden_layer_units_num=0,
            hidden_layer_activation='relu',dropout=0)
    model = Model(inputs=[onehot_input],
                 outputs=[x_rnn])
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
