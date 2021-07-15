import os
import pickle

import pandas as pd

from time import sleep
from urllib.request import urlopen
from progressbar import progressbar


def get_fasta(pdb_id):
    return urlopen(f"https://www.rcsb.org/fasta/entry/{pdb_id}").read().decode('utf-8')


dude_cmaps = [fname for fname in os.listdir("../DUDE/contact_map/")
                if fname.endswith("_full")]
bindingdb_cmaps = [fname for fname in os.listdir("../BindingDB/contact_map/")
                   if fname.endswith("_cm")]

dude_pdb_ids = [fname.split("_")[1][:-1].upper() for fname in dude_cmaps]
bindingdb_pdb_ids = [fname.split("_")[0] for fname in bindingdb_cmaps]

if "dude_fasta_dict.pkl" not in os.listdir():
    print("Downloading DUD-E FASTAs...")
    dude_fasta_dict = {}
    for pdb_id in progressbar(dude_pdb_ids):
        dude_fasta_dict[pdb_id] = get_fasta(pdb_id)
        sleep(1)

    with open("dude_fasta_dict.pkl", "wb") as f:
        pickle.dump(dude_fasta_dict, f)
else:
    dude_fasta_dict = pd.read_pickle("dude_fasta_dict.pkl")


if "bindingdb_fasta_dict.pkl" not in os.listdir():
    print("Downloading BindingDB FASTAs...")
    bindingdb_fasta_dict = {}
    for pdb_id in progressbar(bindingdb_pdb_ids):
        bindingdb_fasta_dict[pdb_id] = get_fasta(pdb_id)
        sleep(1)

    with open("bindingdb_fasta_dict.pkl", "wb") as f:
        pickle.dump(bindingdb_fasta_dict, f)
else:
    bindingdb_fasta_dict = pd.read_pickle("bindingdb_fasta_dict.pkl")


with open("fastas_dude", "w") as f:
    f.writelines(list(dude_fasta_dict.values()))

with open("fastas_bindingdb", "w") as f:
    f.writelines(list(bindingdb_fasta_dict.values()))

