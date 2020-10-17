from sklearn.metrics import  mean_squared_error, r2_score
import scipy as sp
def get_score_at_test(model,input,label,issave=True,savepath=None):
    pred_label = model.predict(input)
    mse = mean_squared_error(label, pred_label)
    spearmanr = sp.stats.spearmanr(label, pred_label)[0]    
    r2 = r2_score(label, pred_label)
    return 'MES:' + str(mse),'Spearman:' + str(spearmanr) , 'r2:' + str(r2)

from ParamsUtil import *

dataset = 'WT'
datas = ReadValidationData(dataset)
from keras.models import load_model
params = GetParams(dataset)
filename = params['CNNParams']['cnn_save_file']
time=0
for data in datas:
    time+=1
    model = load_model('WTBestCNN.h5'+'_'+str(time))
    print(get_score_at_test(model,
                            data['input']['test']['onehot'],data['label']['test']))
    #model = load_model('WTBestRNN.h5')
    #print(get_score_at_test(model,
                        #data['input']['test']['onehot'],data['label']['test']))
