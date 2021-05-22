from model import *
from utils import *
import torch.utils.data as data_utils

SEED_INDEX = 2

#indices = torch.arange(100)
SEED = [87459307486392109,
        48674128193724691,
        71947128564786214]
torch.manual_seed(SEED[SEED_INDEX])

PREFIX = f"orig_{SEED[SEED_INDEX]}"
FOLD = "cv_1"
CUDA_NUM = 1
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
contact_path = "../data/DUDE/contact_map"
contact_dict_path = "../data/DUDE/data_pre/DUDE-contactDict"
seq_contact_dict = getSeqContactDict(contact_path, contact_dict_path)

smile_letters_path  = "../data/DUDE/voc/combinedVoc-wholeFour.voc"
seq_letters_path = "../data/DUDE/voc/sequence.voc"
smiles_letters = getLetters(smile_letters_path)
sequence_letters = getLetters(seq_letters_path)
N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)

train_fold_path = f"../data/DUDE/data_pre/{FOLD}_train_fold"
# train_dataset: [[smile, seq, label], ....]    seq_contact_dict:{seq:contactMap,....}
train_dataset = getTrainDataSet(train_fold_path)
train_dataset = ProDataset(dataSet=train_dataset, seqContactDict=seq_contact_dict)
#train_dataset = data_utils.Subset(train_dataset, indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=model_args["batch_size"],
                            drop_last=True, shuffle=True)

validate_fold_path = f"../data/DUDE/data_pre/{FOLD}_val_fold"
# validate_dataset: [[smile, seq, label],....]    seq_contact_dict:{seq:contactMap,....}
validate_dataset = getTrainDataSet(validate_fold_path)
validate_dataset = ProDataset(dataSet=validate_dataset, seqContactDict=seq_contact_dict)
#validate_dataset = data_utils.Subset(validate_dataset, indices)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=model_args["batch_size"],
                                drop_last=True, shuffle=True)


# Training arguments
train_args = {}

train_args["train_loader"] = train_loader
train_args["seq_contact_dict"] = seq_contact_dict
train_args["epochs"] = 50

train_args["fname_prefix"] = f"{PREFIX}_{FOLD}_"
train_args["device"] = device
train_args["model"] = DrugVQA(model_args, block=ResidualBlock)
train_args["model"], train_args["trained_to"] = load_latest_model(train_args["fname_prefix"],
                                        train_args["epochs"],
                                        train_args["model"],
                                        device)

train_args["lr"] = 0.0007
train_args["use_regularizer"] = False
train_args["penal_coeff"] = 0.03
train_args["clip"] = True
train_args["criterion"] = torch.nn.BCELoss()
train_args["optimizer"] = torch.optim.Adam(train_args['model'].parameters(), lr=train_args['lr'])
train_args["seed"] = SEED[SEED_INDEX]


# Validation arguments
validate_args = {}

validate_args["validate_loader"] = validate_loader
validate_args["seq_contact_dict"] = seq_contact_dict

validate_args["fname_prefix"] = f"{PREFIX}_{FOLD}_"
validate_args["device"] = device

validate_args["use_regularizer"] = False
validate_args["penal_coeff"] = 0.03
validate_args["criterion"] = torch.nn.BCELoss()

