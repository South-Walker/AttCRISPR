CNNParams = {
    'train_batch_size':16,
    'train_epochs_num':75,
    'train_base_learning_rate':0.00002,
    'cnn_save_file':'WTBestCNN.h5',
    'cnn_fc_hidden_layer_num':2,
    'cnn_fc_hidden_layer_units_num':600,
    'cnn_fc_dropout':0.5,
    'cnn_filters_num':61,
    'cnn_conv_num':3,
    'cnn_last_layer_units_num':54
    }
RNNParams = {
    'train_batch_size':64,
    'train_epochs_num':75,
    'train_base_learning_rate':0.0001,
    'rnn_save_file':'WTBestRNN.h5',
    'rnn_embedding_output':77,
    'rnn_unit_num':125,
    'rnn_fc_hidden_layer_num':1,
    'rnn_fc_hidden_layer_units_num':140
    }
EnsembleParams = {
    'cnn_load_file':'WTBestCNN.h5',
    'rnn_load_file':'WTBestRNN.h5',
    'ensemble_save_file':'WTEnsemble.h5',
    'train_batch_size':16,
    'train_epochs_num':75,
    'train_base_learning_rate':0.00002,
    'bio_fc_hidden_layer_num':1,
    'bio_fc_hidden_layer_units_num':87,
    'bio_fc_dropout':0.05
    }
FineTuning = {
    'cnn_load_file':'WTBestCNN.h5',
    'rnn_load_file':'WTBestRNN.h5',
    'ensemble_load_file':'WTEnsemble.h5',
    'ensemble_save_file':'WTFineTuning.h5',
    'train_batch_size':16,
    'train_epochs_num':300,
    'train_base_learning_rate':0.00002,
    'bio_fc_hidden_layer_num':1,
    'bio_fc_hidden_layer_units_num':87,
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
        'rnn_embedding_output':[40,140],
        'rnn_unit_num':[30,130],
        'rnn_fc_hidden_layer_units_num':[66,266],
    }
    }
Params = {
    'data_file':'WTData.pkl',
    'CNNParams':CNNParams,
    'RNNParams':RNNParams,
    'EnsembleParams':EnsembleParams,
    'FineTuning':FineTuning,
    'ParamsRanges':ParamsRanges
    }