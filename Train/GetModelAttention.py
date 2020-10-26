import keras
from keras.models import load_model
import math
import pickle
import numpy as np
from keras.models import *
import scipy as sp
from sklearn.model_selection import train_test_split
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from keras.callbacks import Callback
from keras.callbacks import LambdaCallback
x_attention_onehot = None
x_attention_biofeat = None
x_attention_seq = None
y_attention = None
def ReadData(dataset):
    dict = {'WT':'WTData.pkl','ESP':'ESPData.pkl','SP':'SPData.pkl'}    
    pkl = open(dict[dataset],'rb') 
    x_onehot =  pickle.load(pkl)
    x_biofeat = pickle.load(pkl)
    y = pickle.load(pkl)
    x_seq = pickle.load(pkl)
    global x_attention_onehot,x_attention_biofeat,x_attention_seq,y_attention
    x_attention_onehot = x_onehot
    x_attention_biofeat = x_biofeat
    x_attention_seq = x_seq
    y_attention = y
def get_local_temporal_attention(model,index):
    seq = x_attention_seq[index]
    onehot = x_attention_onehot[index].reshape((1,21,4,1))
    print('check:'+str(seq))
    
    last_layer_weight = model.get_layer('temporal_score').get_weights()[0].reshape((-1))
    temporal_layer_model = Model(
        inputs=model.inputs[0], outputs=model.get_layer('score_at_each_position').output)
    ##(datasize,21,4,1)
    score_at_each_position = temporal_layer_model.predict(onehot)
    baseline = np.mean(score_at_each_position, axis=0)

    baseline = np.multiply(baseline,last_layer_weight)
    
    second_layer_model = Model(
        inputs=model.inputs[0],outputs=model.get_layer('temporal_attention').output)
    #second = second_layer_model.predict(onehot)
    second = second_layer_model.predict(onehot)
    second = np.mean(second,axis=0)
    baseline = np.matmul(baseline,second)
    np.savetxt(str(index)+'.csv',baseline,fmt='%.5f',delimiter=',')
    return None
def get_temporal_attention(model):
    
    second_layer_model = Model(
        inputs=model.inputs[0],outputs=model.get_layer('temporal_attention').output)
    #second = second_layer_model.predict(onehot)
    second = second_layer_model.predict(x_attention_onehot)
    second = np.mean(second,axis=0)
    scale = np.sum(second)/21


    last_layer_weight = model.get_layer('temporal_score').get_weights()[0].reshape((-1))
    temporal_layer_model = Model(
        inputs=model.inputs[0], outputs=model.get_layer('score_at_each_position').output)
    ##(datasize,21)
    score_at_each_position = temporal_layer_model.predict(x_attention_onehot)
    baseline = np.mean(score_at_each_position, axis=0)
    baseline = np.multiply(baseline,last_layer_weight)
    baseline = baseline * scale
    np.savetxt('baseline.csv',baseline,fmt='%.5f',delimiter=',')
    return None
def get_spatial_attention(model):
    spatial_layer_model = Model(
        inputs=model.inputs[0], outputs=model.get_layer('spatial_attention_result').output)
    spatial_value = spatial_layer_model.predict(x_attention_onehot)
    spatial_value = spatial_value.reshape((spatial_value.shape[0],21,4))
    spatial_value = spatial_value * y_attention.reshape((spatial_value.shape[0],1,1))
    spatial_value = np.mean(spatial_value, axis=0)
    spatial_sum = np.sum(spatial_value, axis=1)
    for i in range(len(spatial_value)):
        ZScore(spatial_value[i])
        #t = spatial_sum[i]/4
        #if i == 0:
        #    t*=2
        #for j in range(len(spatial_value[0])):
        #    spatial_value[i][j] -= t
    spatial_value[0][1]=0
    spatial_value[0][3]=0
    return spatial_value.T
def ZScore(list):
    sum = 0
    l = len(list)
    nonzero=0
    for i in range(l):
        if list[i]!=0:
            nonzero+=1
    for i in range(l):
        sum += list[i]
    avg = sum / nonzero
    sumx_2 = 0
    for i in range(l):
        sumx_2 += (list[i]-avg)*(list[i]-avg) if list[i]!=0 else 0
    sigma = math.sqrt(sumx_2 / nonzero)
    for i in range(l):
        list[i] = (list[i]-avg) / sigma if list[i]!=0 else 0
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
    xstep = 0.05
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
    x_index_ls = ['C','A','T','G']
    y_scale_ls = [2*i for i in range(11)]
    y_index_ls = [str(2*i+1) for i in range(10)]
    y_index_ls.append('N')
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
    ax1.view_init(elev=20., azim=-166)
    plt.show()


rnnmodeldict = {'WT':'WTBestRNN.h5','ESP':'ESPBestRNN.h5','SP':'SPBestRNN.h5'}
cnnmodeldict = {'WT':'WTBestCNN.h5','ESP':'ESPBestCNN.h5','SP':'SPBestCNN.h5'}
dataname = 'WT'
ReadData(dataname)
rnn_model = load_model('./'+rnnmodeldict[dataname])
get_local_temporal_attention(rnn_model,7882)
get_temporal_attention(rnn_model)


cnn_model = load_model('./'+cnnmodeldict[dataname])
spatial_attention = get_spatial_attention(cnn_model)
np.savetxt('SP_global_attention.csv',spatial_attention,fmt='%.5f',delimiter=',')
Plot3D(spatial_attention)

print(spatial_attention)

print('end')