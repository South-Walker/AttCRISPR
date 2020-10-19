CNNParams = {
    'train_batch_size':16,
    'train_epochs_num':75,
    'train_base_learning_rate':0.00002,
    'cnn_save_file':'ESPBestCNN.h5',
    'cnn_fc_hidden_layer_num':2,
    'cnn_fc_hidden_layer_units_num':289,
    'cnn_fc_dropout':0.5,
    'cnn_filters_num':69,
    'cnn_conv_num':3,
    'cnn_last_layer_units_num':37
    }
from keras.optimizers import *
RNNParams = {
    'train_batch_size':128,
    'train_epochs_num':1000,
    'train_base_learning_rate':0.0001,
    'optimizer':Adamax,
    'rnn_save_file':'ESPBestRNN.h5',
    'rnn_window_size':4,
    'rnn_embedding_output':75,
    'rnn_last_activation':'linear',
    'rnn_last_use_bias':True,
    'rnn_use_context_state':True,
    'rnn_unit_num':200,
    'rnn_last_dropout':0.40
    }

EnsembleParams = {
    'cnn_load_file':'ESPBestCNN.h5',
    'rnn_load_file':'ESPBestRNN.h5',
    'ensemble_save_file':'ESPEnsemble.h5',
    'train_batch_size':128,
    'train_epochs_num':30,
    'train_base_learning_rate':0.00005,
    'bio_fc_hidden_layer_num':2,
    'bio_fc_hidden_layer_units_num':150,
    'bio_fc_dropout':0.05
    }
FineTuning = {
    'cnn_load_file':'ESPBestCNN.h5',
    'rnn_load_file':'ESPBestRNN.h5',
    'ensemble_load_file':'ESPEnsemble.h5',
    'ensemble_save_file':'ESPFineTuning.h5',
    'train_batch_size':128,
    'train_epochs_num':100,
    'train_base_learning_rate':0.00002,
    'bio_fc_hidden_layer_num':2,
    'bio_fc_hidden_layer_units_num':150,
    'bio_fc_dropout':0.05
    }
ParamsRanges = {
    'CNNParams':
    {
        'cnn_fc_hidden_layer_units_num':[200,600],
        'cnn_filters_num':[30,70],
        'cnn_last_layer_units_num':[10,110]
    },
    'RNNParams':
    {
        'rnn_last_score_num':[100,400],
        'rnn_embedding_output':[10,110],
        'rnn_unit_num':[40,240]
    }
    }
Params = {
    'data_file':'ESPData.pkl',
    'CNNParams':CNNParams,
    'RNNParams':RNNParams,
    'EnsembleParams':EnsembleParams,
    'FineTuning':FineTuning,
    'ParamsRanges':ParamsRanges
    }