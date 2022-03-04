import argparse

from model import *
from utils import *
from run_model import train
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
train_fold_path = f"../../get_data/drugVQA/data/btd_training"

contact_path = "../../get_data/drugVQA/BindingDB"
contact_dict_path = "../../get_data/drugVQA/BindingDB/BindingDB_contactdict"
seq_contact_dict = getSeqContactDict(contact_path, contact_dict_path, train_fold_path)

smile_letters_path  = "../../get_data/drugVQA/voc/combinedVoc-wholeFour.voc"
seq_letters_path = "../../get_data/drugVQA/voc/sequence.voc"
smiles_letters = getLetters(smile_letters_path)
sequence_letters = getLetters(seq_letters_path)
N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)

# train_dataset: [[smile, seq, label], ....]    seq_contact_dict:{seq:contactMap,....}
train_dataset = getTrainDataSet(train_fold_path)
train_dataset = ProDataset(dataSet=train_dataset, seqContactDict=seq_contact_dict)
#train_dataset = data_utils.Subset(train_dataset, indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=model_args["batch_size"],
                            drop_last=True, shuffle=True)


# Training arguments
train_args = {}

train_args["base"] = "BindingDB"
train_args["train_loader"] = train_loader
train_args["smiles_letters"] = smiles_letters
train_args["seq_contact_dict"] = seq_contact_dict
train_args["epochs"] = 50

train_args["fname_prefix"] = f"{SEED_INDEX}_"
train_args["device"] = device
train_args["model"] = DrugVQA(model_args, block=ResidualBlock)
train_args["model"], train_args["train_from"] = load_latest_model(train_args["fname_prefix"],
                                                                    train_args["epochs"],
                                                                    train_args["model"],
                                                                    train_args["base"],
                                                                    device)

train_args["lr"] = 0.0007
train_args["use_regularizer"] = False
train_args["penal_coeff"] = 0.03
train_args["clip"] = True
train_args["criterion"] = torch.nn.BCELoss()
train_args["optimizer"] = torch.optim.Adam(train_args['model'].parameters(), lr=train_args['lr'])
train_args["seed"] = SEED[SEED_INDEX]


train(train_args)

