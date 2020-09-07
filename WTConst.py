CNNParams = {
    'train_batch_size':16,
    'train_epochs_num':75,
    'train_base_learning_rate':0.00002,
    'cnn_save_file':'WTBestCNN.h5',
    'cnn_fc_hidden_layer_num':2,
    'cnn_fc_hidden_layer_units_num':421,
    'cnn_fc_dropout':0.5,
    'cnn_filters_num':50,
    'cnn_conv_num':3,
    'cnn_last_layer_units_num':60
    }
RNNParams = {
    'train_batch_size':128,
    'train_epochs_num':75,
    'train_base_learning_rate':0.001,
    'rnn_save_file':'WTBestRNN.h5',
    'rnn_embedding_output':97,
    'rnn_unit_num':99,
    'rnn_fc_hidden_layer_num':1,
    'rnn_fc_hidden_layer_units_num':166
    }
EnsembleParams = {
    'cnn_load_file':'WTBestCNN.h5',
    'rnn_load_file':'WTBestRNN.h5',
    'ensemble_save_file':'WTEnsemble.h5',
    'train_batch_size':128,
    'train_epochs_num':75,
    'train_base_learning_rate':0.00016,
    'bio_fc_hidden_layer_num':1,
    'bio_fc_hidden_layer_units_num':87,
    'bio_fc_dropout':0.05
    }
Params = {
    'data_file':'alldata.pkl',
    'CNNParams':CNNParams,
    'RNNParams':RNNParams,
    'EnsembleParams':EnsembleParams
    }