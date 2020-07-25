import os
import pandas as pd
import pickle
import numpy as np
import sys



def my_feature(df_model, feature_options):
    feature_options['order'] = 1
    feature_sets = featurize_data( df_model, feature_options )
    inputs, dim, dimsum, feature_names = concatenate_feature_sets( feature_sets )
    return inputs, dim, dimsum, feature_names, feature_sets

def get_embedding_data(data, feature_options):
    #feature_options['order'] = 1
    #generating biofeatures
    r = my_feature( data, feature_options )
    lst_features = [0, 1, 2, 3, -7, -6, -5, -4, -3, -2, -1]
    feat_names = list( r[3][i] for i in lst_features )
    biofeat = r[0][:, lst_features]
    df_biofeat = pd.DataFrame( data=biofeat, columns=feat_names )
    # biological feature reperesentation
    X_biofeat = np.array( df_biofeat )
    return X_biofeat

def getfeat(sgRNAs, model_type='esp'):
    pandas.set_option( 'Precision', 5 )
    df = pandas.DataFrame( {'21mer': sgRNAs}, columns=['21mer'] )
    feature_options['order'] = 1
    feature_sets = featurize_data( df, feature_options )
    return get_embedding_data(df,feature_options)

def savebiofeature():
    f=open("sourcedata.txt","r")
    line = f.readline()
    index = 0
    name = line.replace('\n', '').split('\t')
    sgrnas = []
    while line:
        index+=1
        line = f.readline().replace('\n', '')
        a = line.split('\t')
        if len(a[0])==21:
            sgrnas.append(a[0])
    featset = getfeat(sgrnas)
    npfeatset = np.array(featset,dtype = float)
    f.close()
    fo = open('./biofeat.pkl','wb')
    pickle.dump(npfeatset,fo)
    fo.close()
def encode(sgrna):
    code_dict = {'A': [[1], [0], [0], [0]], 'T': [[0], [1], [0], [0]], 'G': [[0], [0], [1], [0]], 'C': [[0], [0], [0], [1]]}     
    onehot = []
    for i in range(len(sgrna)):    
        mers = []
        for j in range(21):
            mers.append(code_dict[sgrna[i][j]])
        onehot.append(mers)
    np_onehot = np.array(onehot).reshape(-1,21,4,1)    
    return np_onehot
def saveonehot():
    f=open("sourcedata.txt","r")
    line = f.readline()
    index = 0
    name = line.replace('\n', '').split('\t')
    sgrnas = []
    while line:
        index+=1
        line = f.readline().replace('\n', '')
        a = line.split('\t')
        if len(a[0])==21:
            sgrnas.append(a[0])
    np_onehot = encode(sgrnas)
    f.close()
    fo = open('./onehot.pkl','wb')
    pickle.dump(np_onehot,fo)
    fo.close()
def saveeff(index=1):
    f=open("sourcedata.txt","r")
    line = f.readline()
    i = 0
    name = line.replace('\n','').split('\t')
    eff = []
    while line:
        i += 1
        line = f.readline().replace('\n','')
        a = line.split('\t')
        if len(a[0])==21:
            if len(a)<=index or a[index]=='':
                eff.append(-1)
            else:
                eff.append(float(a[index]))
    np_eff = np.array(eff,dtype = float)
    fo = open('./efficiency.pkl','wb')
    pickle.dump(np_eff,fo)
    fo.close()
def sequencing(sgrnas):
    seq_dict = {'T': 1, 'A': 2, 'C': 3, 'G': 4,'START': 0}     
    seqs = []
    for i in range(len(sgrnas)):    
        mers = []
        mers.append(seq_dict['START'])
        for j in range(21):
            mers.append(seq_dict[sgrnas[i][j]])
        seqs.append(mers)
    np_seqs = np.array(seqs).reshape(-1,22)    
    return np_seqs

def saveseq():
    f=open("sourcedata.txt","r")
    line = f.readline()
    index = 0
    name = line.replace('\n', '').split('\t')
    sgrnas = []
    while line:
        index+=1
        line = f.readline().replace('\n', '')
        a = line.split('\t')
        if len(a[0])==21:
            sgrnas.append(a[0])
    np_seqs = sequencing(sgrnas)
    f.close()
    fo = open('./sequence.pkl','wb')
    pickle.dump(np_seqs,fo)
    fo.close()
saveseq()