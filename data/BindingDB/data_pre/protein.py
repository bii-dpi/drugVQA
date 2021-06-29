import os
import pickle

import numpy as np
import pandas as pd

from urllib.request import urlopen


def get_dict(fname):
    with open(fname, "r") as f:
        lines = f.readlines()[1:]
    return dict(zip([lines[i].strip("\n") for i in range(0, len(lines), 2)],
                    [lines[i].strip("\n") for i in range(1, len(lines), 2)]))


# Data
bindingdb_examples = pd.read_csv("bindingdb_examples.tsv", sep="\t")

## Name-to-sequence.
bindingdb_dict = get_dict("mapped_bindingdb_sequences.txt")
dude_dict = get_dict("mapped_dude_sequences.txt")

## Sequence-to-cm ID.
with open("../contact_map/BindingDB-contactDict", "r") as f:
    sequence_to_id_dict = f.readlines()

sequence_to_id_dict = {line.split(":")[0]:line.split(":")[1].strip("\n")
                        for line in sequence_to_id_dict}

## Name-to-similarities.
if "sim_matrix.pkl" not in os.listdir():
    response = urlopen(
        "https://www.ebi.ac.uk/Tools/services/rest/clustalo/result/clustalo-I20210621-232858-0515-28318368-p1m/pim"
    )
    sim_matrix = response.read().decode("utf-8")
    with open("sim_matrix.pkl", "wb") as f:
        pickle.dump(sim_matrix, f)
else:
    sim_matrix = pd.read_pickle("sim_matrix.pkl")

sim_matrix = [line.split() for line in sim_matrix.split("\n")[6:-1]]
names = [line[1] for line in sim_matrix]
sim_matrix = [line[2:] for line in sim_matrix]
sim_dict = {names[i] : np.array(sim_matrix[i], dtype=float).tolist()
            for i in range(len(names))}
sim_dict = {name : dict(zip(names, similarities))
            for name, similarities in sim_dict.items()}


# Protein class definition
class Protein:
    def __init__(self, name):
        # Descriptors
        self.name = name
        self.sequence = Protein.get_sequence(name)
        # Group membership
        self.is_bindingdb = Protein.is_bindingdb(name)
        ## If is DUD-E, _cm will not exist.
        self.cm_exists = Protein.cm_exists(name)
        # Other data
        for key in sim_dict.keys():
            if name in key:
                self.sims = sim_dict[key]
        try:
            print(self.sims)
        except:
            print(name)
        self.set_examples()


    @staticmethod
    def get_sequence(name):
        for key in bindingdb_dict.keys():
            if name in key:
                return bindingdb_dict[key]
        for key in dude_dict.keys():
            if name in key:
                return dude_dict[key]
        raise Exception(f"{name} not in BindingDB or DUD-E dict.")


    @staticmethod
    def is_bindingdb(name):
        for key in bindingdb_dict.keys():
            if key in name:
                return True
        return False


    @staticmethod
    def cm_exists(name):
        sequence = Protein.get_sequence(name)
        try:
            pdb_id = sequence_to_id_dict[sequence]
        except:
            return False
        return f"{pdb_id}" in os.listdir("../contact_map/")


    @staticmethod
    def get_interactivity(nm):
        if not nm:
            return None
        try:
            nm = float(nm)
            if nm * 1000 <= 1:
                return "1"
            elif nm * 1000 >= 50:
                return "0"
            return None
        except:
            if nm.startswith(">") or nm.startswith("<"):
                return get_interaction(nm[1:])
            else:
                raise Exception(f"Unknown nM: {nm}")


    def in_sim_matrix(self):
        return self.in_sim_matrix


    def get_len(self):
        return len(self.sequence)


    def get_dissim_names(self, threshold=20):
        return [name for name in self.sims.keys()
                if cm_exists(name) and
                self.sims[name] <= threshold]


    def set_examples(self):
        def process_row(examples, i):
            line = (f"{examples.ligand_smiles.iloc[i]} "
                    f"{examples.target_sequence.iloc[i]}")
            if not pd.isna(examples.ki.iloc[i]):
                nm = examples.ki.iloc[i]
            elif not pd.isna(examples.ic50.iloc[i]):
                nm = examples.ic50.iloc[i]
            elif not pd.isna(examples.kd.iloc[i]):
                nm = examples.kd.iloc[i]
            elif not pd.isna(examples.ec50.iloc[i]):
                nm = examples.ec50.iloc[i]
            try:
                return f"{line} {Protein.get_interactivity(nm)}"
            except:
                return ""

        self.examples = bindingdb_examples[
                                bindingdb_examples.\
                                target_sequence ==
                                self.sequence
                              ]
        self.examples = [process_row(self.examples, i)
                         for i in range(len(self.examples))]
        self.examples = [example for example in self.examples
                         if example]


    def get_examples(self):
        return self.examples


    def get_actives(self):
        return [" ".join(example) for example in self.examples
                if int(example[1])]


    def get_inactives(self):
        return [" ".join(example) for example in self.examples
                if not int(example[1])]


    def get_sim(self, name):
        for key in self.sims.keys():
            if name in key:
                return self.sims[key]
        raise Exception(f"{name} not found in self.sims")


    def get_dude_sim_mean(self):
        return np.nanmean([self.get_sim(other_name) for other_name
                           in self.sims.keys()
                           if not Protein.is_bindingdb(other_name) and
                           self.name != other_name])


    def __repr__(self):
        return f"{self.name[:4]} {Protein.is_bindingdb(self.name)}"

