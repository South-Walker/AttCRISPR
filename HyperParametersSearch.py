from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import AttCRISPR
from AttCRISPR import train_with
space = {
    'cnn_fc_hidden_layer_num':hp.choice('cnn_fc_hidden_layer_num', [2,3,4]),
    'cnn_filters_num':hp.choice('cnn_filters_num', [20,25]),
    'cnn_fc_hidden_layer_units_num':hp.uniform('cnn_fc_hidden_layer_units_num',0,1),
    'rnn_embedding_output':hp.uniform('rnn_embedding_output',0,1),
    'rnn_unit_num':hp.uniform('rnn_unit_num',0,1),
    'rnn_fc_hidden_layer_num':hp.choice('rnn_fc_hidden_layer_num', [1,2,3]),
    'rnn_fc_hidden_layer_units_num':hp.uniform('rnn_fc_hidden_layer_units_num',0,1),
    'bio_fc_hidden_layer_num':hp.choice('bio_fc_hidden_layer_num', [0,1]),
    'bio_fc_hidden_layer_units_num':hp.uniform('bio_fc_hidden_layer_units_num',0,1),
    }
trials = Trials()
best = fmin(train_with, space, algo=tpe.suggest, max_evals=150, trials=trials)
print('best: ')
print(best)
