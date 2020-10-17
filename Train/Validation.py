import ParamsUtil
from ParamsUtil import *
######

from TrainRNN import train as RNN
from TrainCNN import train as CNN
from Ensemble import train as Ensemble

def ValidateCNN(dataset,validateall=True,validateat=0):
    datas = ReadValidationData(dataset)
    result = []
    time = 0
    params = GetParams(dataset)
    filename = params['CNNParams']['cnn_save_file']
    for data in datas:
        time += 1
        if not validateall:
            if validateat > time:
                continue
            elif validateat < time:
                break
        params['CNNParams']['cnn_save_file'] = filename + '_'+str(time)
        input = data['input']
        label = data['label']
        r = None
        r = CNN(params['CNNParams'],input['train']['onehot'],label['train'],
             input['test']['onehot'],label['test'],issave=True)
        result.append(r['loss'])

        print(result)
def ValidateRNN(dataset,validateall=True,validateat=0):
    datas = ReadValidationData(dataset)
    result = []
    time = 0
    params = GetParams(dataset)
    filename = params['RNNParams']['rnn_save_file']
    for data in datas:
        time += 1
        if not validateall:
            if validateat > time:
                continue
            elif validateat < time:
                break
        params['RNNParams']['rnn_save_file'] = filename + '_'+str(time)
        input = data['input']
        label = data['label']
        r = None
        r = RNN(params['RNNParams'],input['train']['onehot'],label['train'],
             input['test']['onehot'],label['test'],issave=True)
        result.append(r['loss'])
        print(result)
def ValidateEnsemble(dataset,withbiofeature=False,validateall=True,validateat=0):
    datas = ReadValidationData(dataset)
    result = []
    time = 0
    params = GetParams(dataset)
    cnnfilename = params['EnsembleParams']['cnn_load_file']
    rnnfilename = params['EnsembleParams']['rnn_load_file']
    ensemblefilename = params['EnsembleParams']['ensemble_save_file']
    for data in datas:
        time += 1
        if not validateall:
            if validateat > time:
                continue
            elif validateat < time:
                break
        input = data['input']
        label = data['label']
        params['EnsembleParams']['cnn_load_file'] = cnnfilename+'_'+str(time)
        params['EnsembleParams']['rnn_load_file'] = rnnfilename+'_'+str(time)
        params['EnsembleParams']['ensemble_save_file'] = ensemblefilename+'_'+str(time)
        params['FineTuning']['cnn_load_file'] = params['EnsembleParams']['cnn_load_file']
        params['FineTuning']['rnn_load_file'] = params['EnsembleParams']['rnn_load_file']
        params['FineTuning']['ensemble_load_file'] = params['EnsembleParams']['ensemble_save_file']
        r = None
        r = Ensemble(params['EnsembleParams'],input['train']['onehot'],input['train']['biofeat'],label['train'],
             input['test']['onehot'],input['test']['biofeat'],label['test'],issave=True,
             withbiofeature=withbiofeature)
        result.append(r['loss'])
        print(result)
if __name__ == "__main__":
    #ValidateRNN('ESP',validateall=False,validateat=1)
    #ValidateCNN('ESP',validateall=True)
    ValidateEnsemble('WT',withbiofeature=False,validateall=True)