
import pickle
def Read_Data(filename):
    pkl = open(filename,'rb')
    x_onehot = pickle.load(pkl)
    x_biofeat = pickle.load(pkl)
    ZScore(x_biofeat)
    label = pickle.load(pkl)
    x_seq = pickle.load(pkl)
    return SplitData(x_onehot,x_biofeat,label)

dataset = []
dataset.append('WTData.pkl')
dataset.append('ESPData.pkl')
dataset.append('SPData.pkl')
for f in dataset:
    pkl = open(f,'rb')
    x_onehot = pickle.load(pkl)
    x_biofeat = pickle.load(pkl)
    label = pickle.load(pkl)
    x_seq = pickle.load(pkl)
    print(len(x_onehot))
