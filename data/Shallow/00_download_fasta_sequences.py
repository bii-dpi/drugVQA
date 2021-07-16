import os
import pickle

import numpy as np
import pandas as pd

from time import sleep
from urllib.request import urlopen
from progressbar import progressbar


def get_pdb_ids(fname):
    with open(fname, "r") as f:
        return np.unique([line.split(":")[1].split("_")[0] for line in f.readlines()])


def get_fasta(pdb_id):
    return urlopen(f"https://www.rcsb.org/fasta/entry/{pdb_id}").read().decode('utf-8')


bindingdb_pdb_ids = get_pdb_ids("../BindingDB/contact_map/BindingDB_contactdict")
dude_pdb_ids = get_pdb_ids("../DUDE/contact_map/DUDE_contactdict")


if "dude_fasta_dict.pkl" not in os.listdir():
    print("Downloading DUD-E FASTAs...")
    dude_fasta_dict = {}
    for pdb_id in progressbar(dude_pdb_ids):
        dude_fasta_dict[pdb_id] = get_fasta(pdb_id)
#        sleep(1)

    with open("dude_fasta_dict.pkl", "wb") as f:
        pickle.dump(dude_fasta_dict, f)
else:
    dude_fasta_dict = pd.read_pickle("dude_fasta_dict.pkl")


if "bindingdb_fasta_dict.pkl" not in os.listdir():
    print("Downloading BindingDB FASTAs...")
    bindingdb_fasta_dict = {}
    for pdb_id in progressbar(bindingdb_pdb_ids):
        bindingdb_fasta_dict[pdb_id] = get_fasta(pdb_id)
#        sleep(1)

    with open("bindingdb_fasta_dict.pkl", "wb") as f:
        pickle.dump(bindingdb_fasta_dict, f)
else:
    bindingdb_fasta_dict = pd.read_pickle("bindingdb_fasta_dict.pkl")


with open("fastas_dude", "w") as f:
    f.writelines(list(dude_fasta_dict.values()))

with open("fastas_bindingdb", "w") as f:
    f.writelines(list(bindingdb_fasta_dict.values()))

with open("../BindingDB/contact_map/all_fastas", "w") as f:
    f.writelines(list(dude_fasta_dict.values()) +
                 list(bindingdb_fasta_dict.values()))

