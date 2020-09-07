import ParamsUtil
from ParamsUtil import *
######

best = -1
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def AddNoise(onehot,biofeat,label,rate=32,intensity=0.25):
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
        noise_x = (x+offset)%4
        newx = copy.deepcopy(onehot[i])
        newx[j][x][0] = 1-intensity
        newx[j][noise_x][0] = intensity
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


from PretrainCNN import train as CNN
from PretrainRNN import train as RNN
from Ensemble import train as Ensemble

def Pipeline(pretrainCNN=False,pretrainRNN=False,ensemble=False):
    data = Read_Data()
    input = data['input']
    label = data['label']
    
    if pretrainCNN:
        input_train_onehot,input_train_biofeat,y_train = AddNoise(input['train']['onehot'],input['train']['biofeat'],
                                                                  label['train'],rate=32,intensity=0.25)
        CNN(params['CNNParams'],input_train_onehot,y_train,
             input['validate']['onehot'],label['validate'],
             input['test']['onehot'],label['test'])
    if pretrainRNN:
        input_train_onehot,input_train_biofeat,y_train = AddNoise(input['train']['onehot'],input['train']['biofeat'],
                                                                  label['train'],rate=32,intensity=0.1)
        RNN(params['RNNParams'],input_train_onehot,y_train,
             input['validate']['onehot'],label['validate'],
             input['test']['onehot'],label['test'])
    if ensemble:
        Ensemble(params['EnsembleParams'],
                 input['train']['onehot'],input['train']['biofeat'],label['train'],
                 input['validate']['onehot'],input['validate']['biofeat'],label['validate'],
                 input['test']['onehot'],input['test']['biofeat'],label['test'])
if __name__ == "__main__":
    Pipeline(pretrainCNN=True,pretrainRNN=False,ensemble=False)