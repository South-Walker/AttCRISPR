import math
import pickle
import numpy as np

pkl = open('onehot.pkl','rb')
x_onehot = pickle.load(pkl)
pkl = open('efficiency.pkl','rb')
y = pickle.load(pkl)
pkl = open('biofeat.pkl','rb')
x_biofeat = pickle.load(pkl)
pkl = open('sequence.pkl','rb')
x_seq = pickle.load(pkl)

new_x_onehot = []
new_x_biofeat = []
new_y = []
new_x_seq = []

for i in range(len(y)):
    if y[i] == -1:
        continue
    new_x_biofeat.append(x_biofeat[i])
    new_y.append(y[i])
    new_x_onehot.append(x_onehot[i])
    new_x_seq.append(x_seq[i])
new_x_onehot = np.array(new_x_onehot).reshape(-1,21,4,1)
new_x_biofeat = np.array(new_x_biofeat).reshape(-1,11)
new_y = np.array(new_y,dtype = float)
new_x_seq = np.array(new_x_seq).reshape(-1,22)

fo = open('./alldata.pkl','wb')
pickle.dump(new_x_onehot,fo)
pickle.dump(new_x_biofeat,fo)
pickle.dump(new_y,fo)
pickle.dump(new_x_seq,fo)
fo.close()