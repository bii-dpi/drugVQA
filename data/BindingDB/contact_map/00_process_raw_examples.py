import pickle

import numpy as np
import pandas as pd

from Bio.Seq import Seq
from progressbar import progressbar
from rdkit import Chem, DataStructs
from rdkit.Chem.Descriptors import ExactMolWt
from concurrent.futures import ProcessPoolExecutor


SHUFFLE_SEED = 12345
ILLEGAL_LIST = ["[c-]", "[N@@]", "[Re-]", "[S@@+]", "[S@+]"]

with open("../../DUDE/contact_map/DUDE_contactdict", "r") as f:
    DUDE_pdb_ids = [line.strip("\n").split(":")[1].strip("_cm")
                    for line in f.readlines()]


# Line-checking.
def ligand_is_valid(smiles):
    if not smiles:
        return False

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        print(f"Unknown SMILES: {smiles}")
        return False

    try:
        Chem.SanitizeMol(mol)
    except:
        print(f"Unknown SMILES: {smiles}")
        return False

    return ExactMolWt(Chem.MolFromSmiles(smiles)) >= 300


def ligand_has_illegal(smiles):
    for illegal_element in ILLEGAL_LIST:
        if illegal_element in smiles:
            return True
    return False


def protein_is_valid(sequence):
    try:
        Seq(sequence)
        return True
    except:
        print(f"Unknown sequence: {sequence}")
        return False


def pdb_id_is_valid(pdb_id):
    return len(pdb_id) == 4


def is_valid(line):
    return ligand_is_valid(line[0]) and \
            not ligand_has_illegal(line[0]) and \
            protein_is_valid(line[1]) and \
            pdb_id_is_valid(line[2])


# Affinity-checking.
def process_affinity(nm):
    try:
        nm = float(nm)
    except:
        if nm.startswith(">") or nm.startswith("<"):
            return process_affinity(nm[1:])
        raise Exception(f"Unknown nM: {nm}")
    return nm


def get_affinity(affinities):
    for affinity in affinities:
        if affinity:
            try:
                return process_affinity(affinity)
            except Exception as e:
                print(e)
    return None


def ligand_is_active(affinities):
    affinity = get_affinity(affinities)
    if affinity is None or affinity > 1000:
        return False
    return True


def process_line(line):
    line = [entry.strip().strip("\n")
            for entry in line.split("\t")]
    """
    0:1 Ligand SMILES
    1:8 Ki (nM)
    2:9 IC50 (nM)
    3:10 Kd (nM)
    4:11 EC50 (nM)
    5:35 ZINC ID of Ligand
    6:37 BindingDB Target Chain Sequence
    7:38 PDB ID(s) of Target Chain
    """
    line = np.array(line)
    affinities = line[8:12]
    line = line[[1, 37, 38]]
    line[1] = line[1].upper().strip().strip("\n")
    line[2] = line[2].split(",")[0]
    if not (is_valid(line) and
            ligand_is_active(affinities)):
        return []
    return line.tolist()


# Organizing by PDB ID.
def get_selected_examples(pdb_id):
    if pdb_id in DUDE_pdb_ids:
        return []
    selected_examples = [example for example in examples
                         if example[2] == pdb_id]
    all_sequences = np.unique([example[1] for example in selected_examples])
    all_sequences = [sequence for sequence in all_sequences
                     if len(sequence) < 1000]
    if not all_sequences:
        return []

    max_count = 0
    for curr_sequence in all_sequences:
        curr_seq_examples = [example for example in selected_examples
                             if example[1] == curr_sequence]
        curr_count = len(curr_seq_examples)
        if curr_count > max_count:
            max_count = curr_count
            selected_seq_examples = [example for example in curr_seq_examples]

    return selected_seq_examples


with open("BindingDB_ChEMBL.tsv", "r") as f:
    text = f.readlines()[1:]


print(f"Processing {len(text)} raw examples...")
try:
    examples = pd.read_pickle("actives_intermediate.pkl")
except:
    examples = []
    for example in progressbar(text):
        examples.append(process_line(example))
    examples = [example for example in examples if example]
    print(len(examples))
    examples = list(set(tuple(example) for example in examples))
    examples = [list(example) for example in examples]
    print(len(examples))

    with open("actives_intermediate.pkl", "wb") as f:
        pickle.dump(examples, f)

print(f"{len(examples) / len(text):.2f}={len(examples)} actives kept.")
del text

print("")

print("Organizing examples by PDB ID...")
all_pdb_ids = np.unique([example[2] for example in examples])
num_pdb_ids = len(all_pdb_ids)
print(f"{num_pdb_ids} distinct PDB IDs.")

try:
    def get_num_examples(pdb_id, new_examples):
        return len([example for example in new_examples
                    if example[2] == pdb_id])

    new_examples = pd.read_pickle("../data_pre/actives.pkl")
    all_pdb_ids = np.unique([example[2] for example in new_examples])
    num_pdb_ids = len(all_pdb_ids)
    examples_len = [get_num_examples(pdb_id, new_examples)
                    for pdb_id in all_pdb_ids]
except:
    new_examples = []
    examples_len = []
    for pdb_id in progressbar(all_pdb_ids):
        curr_examples = get_selected_examples(pdb_id)
        if len(curr_examples) < 1000:
            num_pdb_ids -= 1
            continue
        examples_len.append(len(curr_examples))
        new_examples += curr_examples

    with open("../data_pre/actives.pkl", "wb") as f:
        pickle.dump(new_examples, f)

print(f"{num_pdb_ids} PDB IDs kept.")
print(f"{len(new_examples) / len(examples):.2f}={len(new_examples)} actives kept"
      f"with about {np.mean(examples_len):.0f} per PDB ID.")
examples = new_examples
del new_examples

print("Writing sequence-to-id map...")
with open("sequence_to_id_map.pkl", "wb") as f:
    pickle.dump(dict(zip([active[1] for active in examples],
                         [active[2] for active in examples])), f)

print("Done.")

