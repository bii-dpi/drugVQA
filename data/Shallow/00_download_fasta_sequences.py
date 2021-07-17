import os
import pickle

import numpy as np
import pandas as pd

from time import sleep
from urllib.request import urlopen
from progressbar import progressbar


def get_dict(path):
    with open(path, "r") as f:
        contact_dict = [line.split(":") for line in f.readlines()]
    return dict(zip([line[0] for line in contact_dict],
                    [line[1].split("_cm")[0] for line in contact_dict]))


def get_fasta(sequence, pdb_id):
    fasta = urlopen(f"https://www.rcsb.org/fasta/entry/{pdb_id}").read().decode('utf-8')
    return fasta.split("\n")[0] + f"\n{sequence}"


bindingdb_contact_dict= get_dict("../BindingDB/contact_map/BindingDB_contactdict")
dude_contact_dict = get_dict("../DUDE/contact_map/DUDE_contactdict")

if "dude_fasta_dict.pkl" not in os.listdir():
    print("Downloading DUD-E FASTAs...")
    dude_fasta_dict = {}
    for sequence in progressbar(dude_contact_dict.keys()):
        dude_fasta_dict[dude_contact_dict[sequence]] = \
            get_fasta(sequence,
                      dude_contact_dict[sequence])
#        sleep(1)

    with open("dude_fasta_dict.pkl", "wb") as f:
        pickle.dump(dude_fasta_dict, f)
else:
    dude_fasta_dict = pd.read_pickle("dude_fasta_dict.pkl")

if "bindingdb_fasta_dict.pkl" not in os.listdir():
    print("Downloading BindingDB FASTAs...")
    bindingdb_fasta_dict = {}
    for sequence in progressbar(bindingdb_contact_dict.keys()):
        bindingdb_fasta_dict[bindingdb_contact_dict[sequence]] = \
            get_fasta(sequence,
                      bindingdb_contact_dict[sequence])
#        sleep(1)

    with open("bindingdb_fasta_dict.pkl", "wb") as f:
        pickle.dump(bindingdb_fasta_dict, f)
else:
    bindingdb_fasta_dict = pd.read_pickle("bindingdb_fasta_dict.pkl")


with open("fastas_dude", "w") as f:
    f.write("\n".join(list(dude_fasta_dict.values())))

with open("fastas_bindingdb", "w") as f:
    f.write("\n".join(list(bindingdb_fasta_dict.values())))

with open("../BindingDB/contact_map/all_fastas", "w") as f:
    f.write("\n".join(list(dude_fasta_dict.values()) +
                      list(bindingdb_fasta_dict.values())))

