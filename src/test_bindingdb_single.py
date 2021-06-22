import argparse

from model import *
from utils import *
from run_model import validate
from progressbar import progressbar
import torch.utils.data as data_utils

parser = argparse.ArgumentParser()

parser.add_argument("RV_SEED_INDEX", type=int)
parser.add_argument("SEED_INDEX", type=int)
parser.add_argument("FOLD_TYPE", type=str)
parser.add_argument("FOLD_NUM", type=int)
parser.add_argument("CUDA_NUM", type=int)

args = parser.parse_args()


RV_SEED_INDEX = args.RV_SEED_INDEX
RV_SEED = [123456789,
           619234965,
           862954379,
           296493420]


SEED_INDEX = args.SEED_INDEX

SEED = [87459307486392109,
        48674128193724691,
        71947128564786214]
torch.manual_seed(SEED[SEED_INDEX])

FOLD_TYPE = args.FOLD_TYPE
if FOLD_TYPE == "rv":
    PREFIX = f"{RV_SEED[RV_SEED_INDEX]}_orig_{FOLD_TYPE}_{SEED[SEED_INDEX]}"
elif FOLD_TYPE == "cv":
    PREFIX = f"orig_{FOLD_TYPE}_{SEED[SEED_INDEX]}"
FOLD = f"{FOLD_TYPE}_{args.FOLD_NUM}"
CUDA_NUM = args.CUDA_NUM
device = torch.device(f"cuda:{CUDA_NUM}")
print(f"Fold {FOLD} on CUDA {CUDA_NUM} with seed {SEED[SEED_INDEX]}.")


# Model args
model_args = {}
model_args["batch_size"] = 1
model_args["lstm_hid_dim"] = 64
model_args["d_a"] = 32
model_args["r"] = 10
model_args["n_chars_smi"] = 247
model_args["n_chars_seq"] = 21
model_args["dropout"] = 0.2
model_args["in_channels"] = 8
model_args["cnn_channels"] = 32
model_args["cnn_layers"] = 4
model_args["emb_dim"] = 30
model_args["dense_hid"] = 64
model_args["task_type"] = 0
model_args["n_classes"] = 1
model_args["device"] = device


# Data
contact_path = "../data/BindingDB/contact_map"
contact_dict_path = "../data/BindingDB/contact_map/BindingDB-contactDict"
seq_contact_dict = getSeqContactDict(contact_path, contact_dict_path)

smile_letters_path  = "../data/DUDE/voc/combinedVoc-wholeFour.voc"
seq_letters_path = "../data/DUDE/voc/sequence.voc"
smiles_letters = getLetters(smile_letters_path)
sequence_letters = getLetters(seq_letters_path)
N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)


validate_fold_path = f"../data/BindingDB/data_pre/bindingdb_examples_filtered_50"
# validate_dataset: [[smile, seq, label],....]    seq_contact_dict:{seq:contactMap,....}
validate_dataset = getTrainDataSet(validate_fold_path)
validate_dataset = ProDataset(dataSet=validate_dataset, seqContactDict=seq_contact_dict)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=model_args["batch_size"],
                                drop_last=True)


# Validation arguments
validate_args = {}

validate_args['smiles_letters'] = smiles_letters

validate_args["epochs"] = 50
validate_args["validate_loader"] = validate_loader
validate_args["seq_contact_dict"] = seq_contact_dict

validate_args["model_fname_prefix"] = f"{PREFIX}_{FOLD}_"
validate_args["fname_prefix"] = f"{PREFIX}_{FOLD}_bindingdb_"
validate_args["device"] = device

validate_args["use_regularizer"] = False
validate_args["penal_coeff"] = 0.03
validate_args["criterion"] = torch.nn.BCELoss()


for i in progressbar(range(validate_args['epochs'], 0, -2)):
    curr_path = f"../model_pkl/DUDE/{validate_args['model_fname_prefix']}{i}.pkl"
    validate_args['model'] = DrugVQA(model_args, block=ResidualBlock)
    validate_args['model'].load_state_dict(torch.load(curr_path,
                                            map_location=device))
    validate_args['model'] = validate_args['model'].to(device)
    validate(validate_args, i)

