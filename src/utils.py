import re
import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader



def create_variable(tensor, device):
    return tensor.to(device)


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string


# Create necessary variables, lengths, and target
def make_variables(lines, properties, letters, device):
    sequence_and_length = [line2voc_arr(line, letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, properties, device)


def make_variables_seq(lines, letters, device):
    sequence_and_length = [line2voc_arr(line, letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences_seq(vectorized_seqs, seq_lengths, device)


def line2voc_arr(line, letters):
    arr = []
    regex = '(\[[^\[\]]{1,10}\])'
    line = replace_halogen(line)
    char_list = re.split(regex, line)
    for li, char in enumerate(char_list):
        if char.startswith('['):
               arr.append(letterToIndex(char, letters))
        else:
            chars = [unit for unit in char]

            for i, unit in enumerate(chars):
                arr.append(letterToIndex(unit, letters))
    return arr, len(arr)


def letterToIndex(letter, smiles_letters):
    return smiles_letters.index(letter)


# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths, properties, device):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Also sort the target (countries) in the same order
    target = properties.double()
    if len(properties):
        target = target[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor, device), \
            create_variable(seq_lengths, device), \
            create_variable(target, device)


def pad_sequences_seq(vectorized_seqs, seq_lengths, device):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#     print(seq_tensor)
    seq_tensor = seq_tensor[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor, device), create_variable(seq_lengths, device)


def construct_vocabulary(smiles_list, fname):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = ds.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open(fname, 'w') as f:
        f.write('<pad>' + "\n")
        for char in add_chars:
            f.write(char + "\n")
    return add_chars


def readLinesStrip(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines


def getProteinSeq(path, contactMapName):
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    return seq


def getProtein(path, contactMapName, contactMap = True):
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    if(contactMap):
        contactMap = []
        for i in range(2, len(proteins)):
            contactMap.append(proteins[i])
        return seq, contactMap
    else:
        return seq


def getContactMap(contactMap):
    contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
    feature2D = np.expand_dims(contactmap_np, axis=0)
    feature2D = torch.FloatTensor(feature2D)
    return feature2D


def getSeqContactDict(contactPath, contactDictPath, validate_fold_path=None):# make a seq-contactMap dict
    """
    contactDict = open(contactDictPath).readlines()
    seqContactDict = {}
    for data in contactDict:
        _, contactMapName = data.strip().split(':')
        seq, contactMap = getProtein(contactPath, contactMapName)
        contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
        feature2D = np.expand_dims(contactmap_np, axis=0)
        try:
            feature2D = torch.FloatTensor(feature2D)
        except:
            raise Exception(seq)
        seqContactDict[seq] = feature2D
    return seqContactDict
    """
    if validate_fold_path is not None:
        with open(validate_fold_path, "r") as f:
            involved_sequences = np.unique([line.split()[1]
                                            for line in f.readlines()])

    with open(contactDictPath, "r") as f:
        seqContactDict = f.readlines()
    seqContactDict = [line.strip("\n").split(":") for line in seqContactDict]
    seqContactDict = dict(zip([line[0] for line in seqContactDict],
                                [line[1] for line in seqContactDict]))
    if validate_fold_path is not None:
        seqContactDict = {sequence: contactMapName
                          for sequence, contactMapName in seqContactDict.items()
                          if sequence in involved_sequences}
    seqContactDict = {sequence: getContactMap(getProtein(contactPath, contactMapName)[1])
                        for sequence, contactMapName in seqContactDict.items()}
    return seqContactDict


def getLetters(path):
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars


def getTrainDataSet(trainFoldPath):
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    trainDataSet = [cpi.strip().split() for cpi in trainCpi_list]
    return trainDataSet#[[smiles, sequence, interaction],.....]


"""
def getDataDict(testFoldPath):
    # Load target name-sequence mapping dictionary.
    with open(f"../data/DUDE/dataPre/DUDE-contactDict") as f:
        contact_dict = dict([line.split(":") for line in f.readlines()])
    with open(testFoldPath, 'r') as f:
        testCpi_list = f.read().strip().split('\n')
    testDataSet = [cpi.strip().split() for cpi in testCpi_list]
    dataDict = dict()
    for target_name in contact_dict.values():
        dataDict[target_name] = []
        for example in testDataSet:
            if contact_dict[example[1]] == target_name:
                dataDict[target_name].append(example)
    for target_name in contact_dict.values():
        if not dataDict[target_name]:
            dataDict.pop(target_name, None)
    return dataDict
"""


class ProDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataSet, seqContactDict):
        self.dataSet = dataSet#list:[[smile, seq, label],....]
        self.len = len(dataSet)
        self.dict = seqContactDict#dict:{seq:contactMap,....}
        self.properties = [int(x[2]) for x in dataSet]# labels
        self.property_list = list(sorted(set(self.properties)))

    def __getitem__(self, index):
        smiles, seq, label = self.dataSet[index]
        contactMap = self.dict[seq]
        return smiles, contactMap, int(label)

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)


def load_latest_model(fname_prefix, epochs, model, device):
    for i in range(1, epochs + 2):
        if not os.path.isfile(f"../model_pkl/DUDE/{fname_prefix}{i}.pkl"):
            break
    if i > 1:
        model.load_state_dict(torch.load(f"../model_pkl/DUDE/{fname_prefix}{i - 1}.pkl",
                                            map_location=device))
    print(f"Training until epoch {i - 1} completed.")
    return model.to(device), i - 1


def get_ROCE(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key = lambda x:x[1], reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate * n)/100)):
                break
    roce = (tp1 * n)/(p*fp1)
    return roce

