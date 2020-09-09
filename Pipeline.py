import ParamsUtil
from ParamsUtil import *
######

def AddNoise(onehot,biofeat,label,rate=50,intensity=0.30):
    from random import randint
    import copy
    t_onehot = onehot.tolist()
    t_biofeat = biofeat.tolist()
    t_label = label.tolist()
    x=0
    for i in range(len(onehot)):
        if randint(0,100) > rate:
            continue
        j = randint(4,16)
        for k in range(4):
            if onehot[i][j][k][0] == 1:
               x = k
        offset = randint(1,4)
        noise = intensity/3
        newx = copy.deepcopy(onehot[i])
        for k in range(4):
            newx[j][k][0] = (1-intensity) if x==k else noise
        t_onehot.append(newx)
        t_biofeat.append(copy.deepcopy(t_biofeat[i]))
        t_label.append(copy.deepcopy(t_label[i]))


    index=np.arange(len(t_onehot))
    np.random.shuffle(index)
    
    t_onehot = np.array(t_onehot).reshape(-1,21,4,1)
    t_biofeat = np.array(t_biofeat).reshape(-1,11)
    t_label = np.array(t_label).reshape(-1)

    onehot=t_onehot[index,:,:,:]
    biofeat=t_biofeat[index,:] 
    label=t_label[index]

    return onehot,biofeat,label

from pretrainRNN import train as RNN
from PretrainCNN import train as CNN
from Ensemble import train as Ensemble

def Pipeline(pretrainCNN=False,pretrainRNN=False,ensemble=False,fineTuning=False):
    data = Read_Data()
    input = data['input']
    label = data['label']
    r = None
    if pretrainCNN:
        input_train_onehot = input['train']['onehot']
        y_train = label['train']
        input_train_onehot,input_train_biofeat,y_train = AddNoise(input['train']['onehot'],input['train']['biofeat'],
                                                                  label['train'],rate=50,intensity=0.3)
        r = CNN(params['CNNParams'],input_train_onehot,y_train,
             input['validate']['onehot'],label['validate'],
             input['test']['onehot'],label['test'])
    if pretrainRNN:
        input_train_onehot,input_train_biofeat,y_train = AddNoise(input['train']['onehot'],input['train']['biofeat'],
                                                                  label['train'],rate=50,intensity=0.3)
        r = RNN(params['RNNParams'],input_train_onehot,y_train,
             input['validate']['onehot'],label['validate'],
             input['test']['onehot'],label['test'])
    if ensemble:
        r = Ensemble(params['EnsembleParams'],
                 input['train']['onehot'],input['train']['biofeat'],label['train'],
                 input['validate']['onehot'],input['validate']['biofeat'],label['validate'],
                 input['test']['onehot'],input['test']['biofeat'],label['test'],
                 cnn_trainable=False,rnn_trainable=False)
    if fineTuning:
        r = Ensemble(params['FineTuning'],
                input['train']['onehot'],input['train']['biofeat'],label['train'],
                input['validate']['onehot'],input['validate']['biofeat'],label['validate'],
                input['test']['onehot'],input['test']['biofeat'],label['test'],
                cnn_trainable=True,rnn_trainable=True,load_weight=True)
    return r
if __name__ == "__main__":
    Pipeline(pretrainCNN=False,pretrainRNN=True,ensemble=False,fineTuning=False) 