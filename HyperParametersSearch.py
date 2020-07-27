from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import AttCRISPR
from AttCRISPR import train_with
space = {
    'train_batch_size':hp.uniform('train_batch_size',0,1),
    'cnn_fc_hidden_layer_num':hp.uniform('cnn_fc_hidden_layer_num',0,1),
    'cnn_fc_hidden_layer_units_num':hp.uniform('cnn_fc_hidden_layer_units_num',0,1),
    'cnn_fc_dropout':hp.uniform('cnn_fc_dropout',.15,.60),
    'lstm_embedding_output':hp.uniform('lstm_embedding_output',0,1),
    'lstm_embedding_dropout':hp.uniform('lstm_embedding_dropout',.25,.75),
    'lstm_unit_num':hp.uniform('lstm_unit_num',0,1),
    'lstm_dropout':hp.uniform('lstm_dropout',.15,.60),
    'lstm_recurrent_dropout':hp.uniform('lstm_recurrent_dropout',.25,.75),
    'lstm_fc_hidden_layer_num':hp.uniform('lstm_fc_hidden_layer_num',0,1),
    'lstm_fc_hidden_layer_units_num':hp.uniform('lstm_fc_hidden_layer_units_num',0,1),
    'lstm_fc_dropout':hp.uniform('lstm_fc_dropout',.25,.75),
    'bio_fc_hidden_layer_num':hp.uniform('bio_fc_hidden_layer_num',0,1),
    'bio_fc_hidden_layer_units_num':hp.uniform('bio_fc_hidden_layer_units_num',0,1),
    'bio_fc_dropout':hp.uniform('bio_fc_dropout',.25,.75)
    }
trials = Trials()
best = fmin(train_with, space, algo=tpe.suggest, max_evals=100, trials=trials)
print('best: ')
print(best)
