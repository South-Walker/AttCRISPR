CNNParams = {
    'train_batch_size':16,
    'train_epochs_num':75,
    'train_base_learning_rate':0.00002,
    'cnn_save_file':'ESPBestCNN.h5',
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
    'rnn_save_file':'ESPBestRNN.h5',
    'rnn_embedding_output':97,
    'rnn_unit_num':99,
    'rnn_fc_hidden_layer_num':1,
    'rnn_fc_hidden_layer_units_num':166
    }
EnsembleParams = {
    'cnn_load_file':'ESPBestCNN.h5',
    'rnn_load_file':'ESPBestRNN.h5',
    'ensemble_save_file':'ESPEnsemble.h5',
    'train_batch_size':16,
    'train_epochs_num':75,
    'train_base_learning_rate':0.00002,
    'bio_fc_hidden_layer_num':1,
    'bio_fc_hidden_layer_units_num':87,
    'bio_fc_dropout':0.05
    }
FineTuning = {
    'cnn_load_file':'ESPBestCNN.h5',
    'rnn_load_file':'ESPBestRNN.h5',
    'ensemble_load_file':'ESPEnsemble.h5',
    'ensemble_save_file':'ESPFineTuning.h5',
    'train_batch_size':64,
    'train_epochs_num':300,
    'train_base_learning_rate':0.001,
    'bio_fc_hidden_layer_num':1,
    'bio_fc_hidden_layer_units_num':87,
    'bio_fc_dropout':0.05
    }
Params = {
    'data_file':'espalldata.pkl',
    'CNNParams':CNNParams,
    'RNNParams':RNNParams,
    'EnsembleParams':EnsembleParams,
    'FineTuning':FineTuning
    }