import ParamsUtil
from ParamsUtil import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from Pipeline import Pipeline
######

params['CNNParams']['train_epochs_num'] = 50
params['RNNParams']['train_epochs_num'] = 50
params['EnsembleParams']['train_epochs_num'] = 50
params['FineTuning']['train_epochs_num'] = 50



def HyperParametersSearch(hyperparameters):
    nonSearch = hyperparameters['nonSearch']
    paramsRange = nonSearch['Range']
    paramsSetName = nonSearch['ParamsSetName']
    name2params = lambda name: int(
        hyperparameters[name]*(paramsRange[name][1]-paramsRange[name][0])
        +paramsRange[name][0]
        )
    for i in hyperparameters.keys():
        if i == 'nonSearch':
            continue
        params[paramsSetName][i] = name2params(i)
    if paramsSetName == 'RNNParams':
        return Pipeline(pretrainCNN=False,pretrainRNN=True)
    if paramsSetName == 'CNNParams':
        return Pipeline(pretrainCNN=True,pretrainRNN=False)
def HyperParameters(paramsRange,paramsSetName):
    space = {
        'nonSearch':
        {
            'Range':paramsRange[paramsSetName],
            'ParamsSetName':paramsSetName
            }
        }
    for i in paramsRange[paramsSetName].keys():
        space.update({i:hp.uniform(i,0,1)})
    trials = Trials()
    result = fmin(HyperParametersSearch, space, algo=tpe.suggest, max_evals=5, trials=trials)
    print('best: ')
    print(result)

if __name__ == "__main__":
    HyperParameters(params['ParamsRanges'],'RNNParams')