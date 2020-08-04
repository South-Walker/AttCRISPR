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


def getscore(model,y):
    y_pre = model.predict([x_test_onehot,x_test_biofeat,x_test_seq]).reshape(-1)

    return float(sp.stats.spearmanr(y, y_pre)[0])

from keras.callbacks import Callback
from keras.callbacks import LambdaCallback
x_attention_onehot = x_train_onehot
x_attention_biofeat = x_train_biofeat
x_attention_seq = x_train_seq
y_attention = y_train
def get_temporal_attention(model,usebiofeat=True):
    temporal_layer_model = Model(
        inputs=load_model.input, outputs=model.get_layer('rnn_embedding').output)
    if usebiofeat:
        input = [x_attention_onehot,x_attention_biofeat,x_attention_seq]
    else:
        input = [x_attention_onehot,x_attention_seq]
    temporal_value = temporal_layer_model.predict(input)
    weights = [0 for x in range(0,21)]
    for w in range(len(temporal_value)):
        for i in range(21):
            weights[i] += temporal_value[w][i]*y_attention[w]
    total = 0
    for i in range(21):
        total += weights[i]
    for i in range(21):
        weights[i] = weights[i] / total
    return weights;
def get_spatial_attention(model,usebiofeat=True):
    weight = get_temporal_attention(model,usebiofeat);
    base_code_dict = {'T': 1, 'A': 2, 'C': 3, 'G': 4,'START': 0} 
    code_base_dict = {1: 'T', 2: 'A', 3: 'C', 4: 'G'}
    at_pos_spatial_weights_of = {'A':[0 for x in range(0,21)],'C':[0 for x in range(0,21)],
                                  'G':[0 for x in range(0,21)],'T':[0 for x in range(0,21)]}
    at_pos_count_of = {'A':[0 for x in range(0,21)],'C':[0 for x in range(0,21)],
                       'G':[0 for x in range(0,21)],'T':[0 for x in range(0,21)]}
    spatial_layer_model = Model(
        inputs=load_model.input, outputs=model.get_layer('cnn_embedding').output)
    if usebiofeat:
        input = [x_attention_onehot,x_attention_biofeat,x_attention_seq]
    else:
        input = [x_attention_onehot,x_attention_seq]
    spatial_value = spatial_layer_model.predict(input)
    for i in range(len(x_attention_seq)):
        for basepos in range(21):
            base = code_base_dict[x_attention_seq[i][basepos]]
            at_pos_count_of[base][basepos]+=1
            at_pos_spatial_weights_of[base][basepos]+=spatial_value[i][basepos]*y_attention[i]*weight[basepos]
    for k in at_pos_spatial_weights_of.keys():
        for basepos in range(21):
            at_pos_spatial_weights_of[k][basepos]/=max(at_pos_count_of[k][basepos],1)
    return at_pos_spatial_weights_of;
def map_temporal_attention_to_sum0(temporal_attention):
    for pos in range(21):
        total = 0
        nonzero = 0
        for k in temporal_attention.keys():
            total += temporal_attention[k][pos]
            if temporal_attention[k][pos] != 0:
                nonzero += 1
        offset = (total)/max(nonzero,1)
        for k in temporal_attention.keys():
            if temporal_attention[k][pos] != 0:
                temporal_attention[k][pos] -= offset
def bezierdim1(t,points,pointsnum=4):
    if t<0 or t>1:
        print('wrong')
    if pointsnum == 4:
        return t*t*t*points[3]+3*t*t*(1-t)*points[2]+3*t*(1-t)*(1-t)*points[1]+(1-t)*(1-t)*(1-t)*points[0]
    elif pointsnum == 3:
        return t*t*t*points[2]+3*t*t*(1-t)*points[2]+3*t*(1-t)*(1-t)*points[1]+(1-t)*(1-t)*(1-t)*points[0]
        #return t*t*points[2]+2*t*(1-t)*points[1]+(1-t)*(1-t)*points[0]
    return 0
def Plot3D(spatial_attention):
    base_code_dict = {'T': 1, 'A': 2, 'C': 3, 'G': 4} 
    code_base_dict = {1: 'T', 2: 'A', 3: 'C', 4: 'G'}
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')

    import numpy as np
    xstep = 0.1
    ystep = 0.1
    xbegin = 1.0
    ybegin = 0.0
    xend = 4.001
    yend = 20.001
    xx = np.arange(xbegin,xend,xstep)
    yy = np.arange(ybegin,yend,ystep)
    X,Y = np.meshgrid(xx,yy)
    Z = X*Y
    minvalue=999
    maxvalue=-999
    beginat = 0
    endat = min(20, beginat+3)
    for i in range(Z.shape[0]):
        nowposy = ybegin+i*ystep
        if nowposy > endat:
            beginat+=2
            endat=min(20,beginat+3)
        u = float(nowposy-beginat)/float(endat-beginat)
        point0 = bezierdim1(u,spatial_attention[code_base_dict[1]][beginat:endat+1],
                            pointsnum=endat-beginat+1)
        point1 = bezierdim1(u,spatial_attention[code_base_dict[2]][beginat:endat+1],
                            pointsnum=endat-beginat+1)
        point2 = bezierdim1(u,spatial_attention[code_base_dict[3]][beginat:endat+1],
                            pointsnum=endat-beginat+1)
        point3 = bezierdim1(u,spatial_attention[code_base_dict[4]][beginat:endat+1],
                            pointsnum=endat-beginat+1)
        for j in range(Z.shape[1]):
            nowposx = xbegin+j*xstep
            v = (nowposx-xbegin)/(xend-xbegin)
            Z[i][j] = bezierdim1(v,[point0,point1,point2,point3],pointsnum=4)
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
        input = [onex_onehot,onex_biofeat,onex_seq]
    else:
        input = [onex_onehot,onex_onehot]
    values = layer_model.predict(input)
    begin = 0
    step = 20

    values.tofile('singleall.out',sep=',',format='%s')

load_model = load_model('./conv_50.h5')
#SinglePred()



batch_end_print_callback = LambdaCallback(
    on_epoch_end=lambda batch,logs: print(getscore(load_model,y_test)))

isusebiofeat = True
spatial_weights = get_spatial_attention(load_model,isusebiofeat)
temporal_weights = get_temporal_attention(load_model,isusebiofeat)
map_temporal_attention_to_sum0(spatial_weights)
print(temporal_weights)
print()
for k in spatial_weights.keys():
    print(spatial_weights[k])
Plot3D(spatial_weights)
print('end')