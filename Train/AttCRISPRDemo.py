from Pipeline import Pipeline as train
import sys
if __name__ == "__main__":
    assert len(sys.argv)==2
    assert sys.argv[1] in ['WT','ESP','SP']
    train(sys.argv[1],pretrainCNN=True,pretrainRNN=True,ensemble=True,fineTuning=True) 