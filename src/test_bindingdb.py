import argparse

from model import *
from utils import *
from run_model import validate
import torch.utils.data as data_utils


parser = argparse.ArgumentParser()

parser.add_argument("SEED_INDEX", type=int)
parser.add_argument("CUDA_NUM", type=int)

args = parser.parse_args()


SEED_INDEX = args.SEED_INDEX

SEED = [87459307486392109,
        48674128193724691,
        71947128564786214]
torch.manual_seed(SEED[SEED_INDEX])

PREFIX = f"{SEED[SEED_INDEX]}"
CUDA_NUM = args.CUDA_NUM
device = torch.device(f"cuda:{CUDA_NUM}")
print(f"CUDA {CUDA_NUM} with seed {SEED[SEED_INDEX]}.")


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
validate_fold_path = f"../../get_data/drugVQA/data/btd_testing"
contact_path = "../../get_data/drugVQA/DUDE"
contact_dict_path = "../../get_data/drugVQA/DUDE/DUDE_contactdict"
seq_contact_dict = getSeqContactDict(contact_path, contact_dict_path,
                                     validate_fold_path)

smile_letters_path  = "../../get_data/drugVQA/voc/combinedVoc-wholeFour.voc"
seq_letters_path = "../../get_data/drugVQA/voc/sequence.voc"
smiles_letters = getLetters(smile_letters_path)
sequence_letters = getLetters(seq_letters_path)
N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)


# validate_dataset: [[smile, seq, label],....]    seq_contact_dict:{seq:contactMap,....}
validate_dataset = getTrainDataSet(validate_fold_path)
validate_dataset = validate_dataset[:100000]
validate_dataset = ProDataset(dataSet=validate_dataset, seqContactDict=seq_contact_dict)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=model_args["batch_size"],
                                drop_last=True)


# Validation arguments
validate_args = {}

validate_args['smiles_letters'] = smiles_letters
validate_args['direction'] = "btd"

validate_args["epochs"] = 50
validate_args["validate_loader"] = validate_loader
validate_args["seq_contact_dict"] = seq_contact_dict

validate_args["model_fname_prefix"] = f"{SEED_INDEX}_50"
validate_args["fname_prefix"] = f"../results/BindingDB/{SEED_INDEX}_"
validate_args["device"] = device

validate_args["use_regularizer"] = False
validate_args["penal_coeff"] = 0.03
validate_args["criterion"] = torch.nn.BCELoss()


curr_path = f"../model_pkl/BindingDB/{validate_args['model_fname_prefix']}.pkl"
validate_args['model'] = DrugVQA(model_args, block=ResidualBlock)
validate_args['model'].load_state_dict(torch.load(curr_path,
                                        map_location=device))
validate_args['model'] = validate_args['model'].to(device)
validate(validate_args)

