class Result(object):
    Best = -1
from sklearn.metrics import  mean_squared_error, r2_score
import scipy as sp
def get_score_at_test(model,input,result,label,issave=True,savepath=None):
    pred_label = model.predict(input)
    mse = mean_squared_error(label, pred_label)
    spearmanr = sp.stats.spearmanr(label, pred_label)[0]    
    r2 = r2_score(label, pred_label)
    
    if result.Best<spearmanr:
        result.Best = spearmanr
        if issave:
            model.save(savepath)
        print('best')
    return 'MES:' + str(mse),'Spearman:' + str(spearmanr) , 'r2:' + str(r2), 'best:' + str(result.Best)


import keras
from keras.layers import Input, Dense, BatchNormalization
from keras.layers.core import *
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
