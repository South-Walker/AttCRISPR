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
    #magic
    for i in range(21):
        sum=0
        for j in range(21):
            sum+=result[i][j]
        time=42/sum
        for j in range(21):
            result[i][j]*=time
    return np.array(result)
def model(params):
    GaussianBuffer = GaussianKernelBuffer()

    onehot_input = Input(name = 'onehot_input', shape = (21,4, 1,))
    embedded = Conv2D(params['rnn_embedding_output'], (1, 4),strides=(1,4), padding='Valid', activation=None)(onehot_input)
    embedded = Reshape((21,params['rnn_embedding_output'],))(embedded)
    embedded = SpatialDropout1D(0.25)(embedded)
    ######encoder&decoder######
    encoder = GRU(params['rnn_unit_num'],return_sequences=True,return_state=False,unroll=True)
    encoder = Bidirectional(encoder,merge_mode='sum',name='encoder')
    encoder_output = encoder(embedded)

    decoder_input = keras.layers.concatenate([embedded,encoder_output])
    #decoder_input = encoder_output
    decoder = GRU(params['rnn_unit_num'],return_sequences=True,return_state=True,unroll=True,
                  dropout=0.25,kernel_regularizer='l2',recurrent_regularizer='l2',recurrent_dropout=0.25)
    decoder = Bidirectional(decoder,merge_mode='sum',name='decoder')
    decoder_output,_,context = decoder(decoder_input)
    #context = Lambda(lambda inp: inp[1]+inp[0],name='context')([c1,c2])
    

    encoderat = []
    decoderat = []
    for i in range(21):
        encoderat.append(
            Flatten()(Cropping1D(cropping=(i,21-1-i))(encoder_output)))
        decoderat.append(
            Flatten()(Cropping1D(cropping=(i,21-1-i))(decoder_output)))
    ######attention######
    aat = []
    sqrtd = math.sqrt(params['rnn_unit_num'])
    for i in range(21):
        atat = []
        for j in range(21):
            #important of pos[j] in scoring pos[i]
            align = dot([encoderat[j],decoderat[i]],axes=-1)
            atat.append(
                align
                )
        ## add l2 regular
        at = keras.layers.concatenate(atat)
        #l2(0.00001) is opt while 0.001 is best
        at = Dense(21,activation='softmax',use_bias=True,activity_regularizer=keras.regularizers.l2(0.00001))(at)
        #at[j] important of pos[j] in scoring pos[i]

        at = Reshape((1,21,))(at)
        aat.append(at)
    #aat[i][j] important of pos[j] in scoring pos[i]
    m = keras.layers.concatenate(aat,axis=-2)
    weight = Lambda(lambda inp: K.constant(GaussianBuffer)*inp ,name = 'temporal_attention')(m)
    weightavg = Lambda(lambda inp: K.batch_dot(inp[0],inp[1]),name='weight_avg')([weight,encoder_output])
    context = Dense(params['rnn_unit_num'],activation='tanh', use_bias=True)(context)
    context = Lambda(lambda inp: K.repeat(inp,21))(context)

    context = BatchNormalization()(context)
    weightavg = BatchNormalization()(weightavg)
    
    score = Lambda(lambda inp: inp[0]*(inp[1]+inp[2]),name='score')([embedded,weightavg,context])
    rnn_embedded = Lambda(lambda inp: K.sum(inp, axis=-1, keepdims=False),name='last_score')(score)
    #magic
    rnn_embedded = Dropout(rate=0.05)(rnn_embedded)
    output = Dense(units=1,kernel_regularizer='l2',kernel_constraint=keras.constraints.NonNeg(),name='temporal_score',activation='tanh',use_bias=False)(rnn_embedded)
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
if __name__ == "__main__":
    import ParamsUtil
    from ParamsUtil import *
    data = Read_Data()
    input = data['input']
    label = data['label']
    input_train_onehot,input_train_biofeat,y_train = AddNoise(input['train']['onehot'],input['train']['biofeat'],
                                                              label['train'],rate=0,intensity=0)
    train(params['RNNParams'],input_train_onehot,y_train,
          input['validate']['onehot'],label['validate'],
          input['test']['onehot'],label['test'])