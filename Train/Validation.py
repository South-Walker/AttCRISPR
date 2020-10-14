import ParamsUtil
from ParamsUtil import *
######

from TrainRNN import train as RNN
from TrainCNN import train as CNN
from Ensemble import train as Ensemble

def ValidateCNN(dataset):
    datas = ReadValidationData(dataset)
    result = []
    time = 0
    for data in datas:
        time+=1
        params = GetParams(dataset)
        params['cnn_save_file'] = params['cnn_save_file']+'_'+str(time)
        input = data['input']
        label = data['label']
        r = None
        r = CNN(params['CNNParams'],input['train']['onehot'],label['train'],
             input['test']['onehot'],label['test'],issave=True)
        result.append(r.best)

        print(result)
if __name__ == "__main__":
    ValidateCNN('WT')

