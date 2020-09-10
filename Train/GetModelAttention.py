from keras.models import load_model
import math
import pickle
import numpy as np
from keras.models import *
import scipy as sp
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pkl = open('alldata.pkl','rb') 
x_onehot =  pickle.load(pkl)
x_biofeat = pickle.load(pkl)
y = pickle.load(pkl)
x_seq = pickle.load(pkl)

random_state=40
test_size = 0.15
x_train_onehot, x_test_onehot, y_train, y_test = train_test_split(x_onehot, y, test_size=test_size, random_state=random_state)
x_train_biofeat, x_test_biofeat, y_train, y_test = train_test_split(x_biofeat, y, test_size=test_size, random_state=random_state)
x_train_seq, x_test_seq, y_train, y_test = train_test_split(x_seq, y, test_size=test_size, random_state=random_state)

from keras.callbacks import Callback
from keras.callbacks import LambdaCallback
x_attention_onehot = x_train_onehot
x_attention_biofeat = x_train_biofeat
x_attention_seq = x_train_seq
y_attention = y_train
def get_temporal_attention(model):
    temporal_layer_models = []
    temporal_values = []
    for i in range(21):
        print(i)
        temporal_layer_model = Model(inputs=model.inputs[0], outputs=model.get_layer('temporal_attention_'+str(i)).output)
        temporal_values.append(
            np.mean(temporal_layer_model.predict(x_attention_onehot), axis=0)
            )
    return temporal_values
def get_spatial_attention(model):
    spatial_layer_model = Model(
        inputs=model.inputs[0], outputs=model.get_layer('spatial_attention_result').output)
    spatial_value = spatial_layer_model.predict(x_attention_onehot)
    spatial_value = spatial_value.reshape((spatial_value.shape[0],21,4))
    spatial_value = spatial_value * y_attention.reshape((spatial_value.shape[0],1,1))
    spatial_value = np.mean(spatial_value, axis=0)
    spatial_sum = np.sum(spatial_value, axis=1)
    for i in range(len(spatial_value)):
        t = spatial_sum[i]/4
        if i == 0:
            t*=2
        for j in range(len(spatial_value[0])):
            spatial_value[i][j] -= t
    spatial_value[0][1]=0
    spatial_value[0][3]=0
    return spatial_value.T

def bezierdim1(t,points,pointsnum=4):
    if t<0 or t>1:
        print('wrong')
    if pointsnum == 4:
        return t*t*t*points[3]+3*t*t*(1-t)*points[2]+3*t*(1-t)*(1-t)*points[1]+(1-t)*(1-t)*(1-t)*points[0]
    elif pointsnum == 3:
        return t*t*t*points[2]+3*t*t*(1-t)*points[2]+3*t*(1-t)*(1-t)*points[1]+(1-t)*(1-t)*(1-t)*points[0]
        #return t*t*points[2]+2*t*(1-t)*points[1]+(1-t)*(1-t)*points[0]
    return 0
def rec(n,m):
    if m == n:
        return 1
    elif m == 1:
        return n
    else:
        return rec(n-1,m-1)+rec(n-1,m)
    
def bezier(t,points,pointsnum): 
    tpow = []
    oneminuestpow = []
    cnums = []
    for i in range(pointsnum):
        if i==0:
            tpow.append(1)
            oneminuestpow.append(1)
            cnums.append(1)
        else :
            tpow.append(t*tpow[i-1])
            oneminuestpow.append((1-t)*oneminuestpow[i-1])
            cnums.append(rec((pointsnum-1),i))
    r = 0
    for i in range(pointsnum):
        r+=points[i]*tpow[i]*oneminuestpow[pointsnum-1-i]*cnums[i]

    return r

def Plot3D(spatial_attention):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')

    import numpy as np
    xstep = 0.5
    ystep = 0.2
    xbegin = 1.0
    ybegin = 0.0
    xend = 4.0001
    yend = 20.0001
    xx = np.arange(xbegin,xend,xstep)
    yy = np.arange(ybegin,yend,ystep)
    xend = 4
    yend = 20
    X,Y = np.meshgrid(xx,yy)
    Z = X*Y
    minvalue=999
    maxvalue=-999
    for i in range(Z.shape[0]):
        nowposy = ybegin+i*ystep
        
        u = float(nowposy)/float(yend-ybegin)

        point0 = bezier(u,spatial_attention[3],21)
        point1 = bezier(u,spatial_attention[0],21)
        point2 = bezier(u,spatial_attention[1],21)
        point3 = bezier(u,spatial_attention[2],21)
        print(u)

        for j in range(Z.shape[1]):
            nowposx = xbegin+j*xstep
            v = (nowposx-xbegin)/(xend-xbegin)
            Z[i][j] = bezier(v,[point0,point1,point2,point3],4)
            minvalue = Z[i][j] if Z[i][j]<minvalue else minvalue
            maxvalue = Z[i][j] if Z[i][j]>maxvalue else maxvalue
    x_scale_ls = [i+1 for i in range(4)]
    x_index_ls = ['T','A','C','G']
    y_scale_ls = [2*i for i in range(11)]
    y_index_ls = [str(2*i+1) for i in range(11)]
    ax1.plot_surface(X,Y,Z,cmap='rainbow')
    ax1.contour(X,Y,Z,zdim='z',offset=minvalue,cmap='rainbow')
    distance = maxvalue-minvalue
    maxvalue -= distance/4
    minvalue += distance/4
    z_scale_ls = [minvalue,maxvalue]
    z_index_ls = ['Disfavored','Favored']
    plt.xticks(x_scale_ls,x_index_ls)
    plt.yticks(y_scale_ls,y_index_ls)
    ax1.set_zticks(z_scale_ls)
    ax1.set_zticklabels(z_index_ls)
    ax1.set_xlabel('Base') 
    ax1.set_ylabel('Position')
    ax1.view_init(elev=17., azim=-141)
    plt.show()
def SinglePred(usebiofeat=True):
    onepkl = open('GTTGAGAAGGACCGCCACAAC.pkl','rb')
    onex_seq = pickle.load(onepkl)
    onex_onehot =  pickle.load(onepkl)
    onex_biofeat = pickle.load(onepkl)
    layer_model = Model(
        inputs=load_model.input, outputs=load_model.get_layer('conv_output').output)
    if usebiofeat:
        input = [onex_onehot,onex_biofeat]
    else:
        input = [onex_onehot]
    values = layer_model.predict(input)
    begin = 0
    step = 20

    values.tofile('singleall.out',sep=',',format='%s')

load_model = load_model('./WTBestRNN.h5')

load_model.summary()
#cnn_model = load_model.get_layer('cnn')
cnn_model = load_model
#rnn_model = load_model.get_layer('rnn')
rnn_model = load_model
temporal_attention = get_temporal_attention(rnn_model)
print(temporal_attention)
spatial_attention = get_spatial_attention(cnn_model)
for i in range(4):
    for j in range(21):
        print(str(spatial_attention[i][j]) + ",",end='')
    print('')


Plot3D(spatial_attention)
print('end')