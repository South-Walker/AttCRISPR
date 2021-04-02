import ParamsUtil
from ParamsUtil import *
######

from TrainRNN import train as RNN
from TrainCNN import train as CNN
from Ensemble import train as Ensemble

def Pipeline(dataset,pretrainCNN=False,pretrainRNN=False,ensemble=False,fineTuning=False):
    data = ReadData(dataset)
    params = GetParams(dataset)
    input = data['input']
    label = data['label']
    r = None
    if pretrainCNN:
        r = CNN(params['CNNParams'],input['train']['onehot'],label['train'],
             input['test']['onehot'],label['test'])
    if pretrainRNN:
        r = RNN(params['RNNParams'],input['train']['onehot'],label['train'],
             input['test']['onehot'],label['test'])
    if ensemble:
        input_train_onehot,input_train_biofeat,y_train = input['train']['onehot'],input['train']['biofeat'],label['train']
        r = Ensemble(params['EnsembleParams'],
                 input['train']['onehot'],input['train']['biofeat'],label['train'],
                 input['test']['onehot'],input['test']['biofeat'],label['test'],
                 cnn_trainable=False,rnn_trainable=False)
    if fineTuning:
        r = Ensemble(params['FineTuning'],
                input['train']['onehot'],input['train']['biofeat'],label['train'],
                input['test']['onehot'],input['test']['biofeat'],label['test'],
                cnn_trainable=True,rnn_trainable=True,load_weight=True)
    return r
if __name__ == "__main__":
    Pipeline('WT',pretrainCNN=False,pretrainRNN=False,ensemble=False,fineTuning=True) 