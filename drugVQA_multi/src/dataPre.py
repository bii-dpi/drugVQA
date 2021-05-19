from model import *
from utils import *
from torch.utils.data import Dataset, DataLoader

import os
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-g', '--gpus', default=2, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
args = parser.parse_args()
args.world_size = args.gpus * args.nodes
os.environ['MASTER_ADDR'] = '172.20.156.107'
os.environ['MASTER_PORT'] = '2212'

fold = "cv_1_"

testFoldPath = f'../data/DUDE/dataPre/{fold}val_fold'
trainFoldPath = f'../data/DUDE/dataPre/{fold}train_fold'

contactPath = '../data/DUDE/contactMap'
contactDictPath = '../data/DUDE/dataPre/DUDE-contactDict'
smileLettersPath  = '../data/DUDE/voc/combinedVoc-wholeFour.voc'
seqLettersPath = '../data/DUDE/voc/sequence.voc'
trainDataSet = getTrainDataSet(trainFoldPath)
seqContactDict = getSeqContactDict(contactPath, contactDictPath)
smiles_letters = getLetters(smileLettersPath)
sequence_letters = getLetters(seqLettersPath)

dataDict = getDataDict(testFoldPath)

N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)

modelArgs = {}
modelArgs['batch_size'] = 1
modelArgs['lstm_hid_dim'] = 64
modelArgs['d_a'] = 32
modelArgs['r'] = 10
modelArgs['n_chars_smi'] = 247
modelArgs['n_chars_seq'] = 21
modelArgs['dropout'] = 0.2
modelArgs['in_channels'] = 8
modelArgs['cnn_channels'] = 32
modelArgs['cnn_layers'] = 4
modelArgs['emb_dim'] = 30
modelArgs['dense_hid'] = 64
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1
#modelArgs['device'] = device

# trainDataSet:[[smile, seq, label], ....]    seqContactDict:{seq:contactMap,....}
train_dataset = ProDataset(dataSet = trainDataSet, seqContactDict = seqContactDict)
train_loader = DataLoader(dataset = train_dataset, batch_size=modelArgs['batch_size'], shuffle=True, drop_last = True)


trainArgs = {}
trainArgs['modelArgs'] = modelArgs
trainArgs['epochs'] = 24
trainArgs['lr'] = 0.0007
trainArgs['train_dataset'] = train_dataset
trainArgs['train_loader'] = train_loader
trainArgs['doTest'] = False
trainArgs['test_proteins'] = list(dataDict.keys())
trainArgs['testDataDict'] = dataDict
trainArgs['seqContactDict'] = seqContactDict
trainArgs['use_regularizer'] = False
trainArgs['penal_coeff'] = 0.03
trainArgs['clip'] = True
trainArgs['criterion'] = torch.nn.BCELoss()
trainArgs['doSave'] = True
trainArgs['saveNamePre'] = fold
trainArgs['nodes'] = args.nodes
trainArgs['gpus'] = args.gpus
trainArgs['nr'] = args.nr
trainArgs['world_size'] = args.world_size
trainArgs['seed'] = 12345
#trainArgs['device'] = device

"""
# testDataSet:[[smile, seq, label],....]    seqContactDict:{seq:contactMap,....}
testDataSet = getTrainDataSet(testFoldPath)
test_dataset = ProDataset(dataSet = testDataSet, seqContactDict = seqContactDict)
test_loader = DataLoader(dataset = test_dataset, batch_size=modelArgs['batch_size'], shuffle=True, drop_last = True)


testArgs = {}
testArgs['test_loader'] = test_loader
testArgs['model'] = DrugVQA(modelArgs, block = ResidualBlock)
testArgs['test_proteins'] = list(dataDict.keys())
testArgs['testDataDict'] = dataDict
testArgs['seqContactDict'] = seqContactDict
testArgs['use_regularizer'] = False
testArgs['penal_coeff'] = 0.03
testArgs['criterion'] = torch.nn.BCELoss()
testArgs['saveNamePre'] = fold
#testArgs['device'] = device
testArgs['nodes'] = args.nodes
testArgs['gpus'] = args.gpus
testArgs['nr'] = args.nr
testArgs['world_size'] = args.world_size
testArgs['seed'] = 12345
"""

