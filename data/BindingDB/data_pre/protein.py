import os
import pickle

import numpy as np
import pandas as pd

from urllib.request import urlopen


INACTIVE_THRESHOLD = 25
SHARING_DISSIM_THRESHOLD = 20
SHUFFLE_SEED = 12345


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
            if key in name:
                self.sims = sim_dict[key]
        self.set_examples()


    # Used in initialization.
    @staticmethod
    def get_sequence(name):
        for key in bindingdb_dict.keys():
            if name in key or name in key:
                return bindingdb_dict[key]
        for key in dude_dict.keys():
            if name in key or name in key:
                return dude_dict[key]
        raise Exception(f"{name} not in BindingDB or DUD-E dict.")


    @staticmethod
    def is_bindingdb(name):
        for key in bindingdb_dict.keys():
            if key in name or name in key:
                return True
        return False


    @staticmethod
    def cm_exists(name):
        sequence = Protein.get_sequence(name)
        try:
            pdb_id = sequence_to_id_dict[sequence]
        except:
            return False
        return pdb_id in os.listdir("../contact_map/")


    @staticmethod
    def get_interactivity(nm):
        try:
            nm = float(nm)
        except:
            if nm is None:
                return nm
            if nm.startswith(">") or nm.startswith("<"):
                return Protein.get_interactivity(nm[1:])
            else:
                raise Exception(f"Unknown nM: {nm}")

        if nm <= 1 * 1000:
            return "1"
        elif nm >= INACTIVE_THRESHOLD * 1000:
            return "0"
        return None


    # Used in setting and adding examples.
    def set_examples(self):
        def process_row(examples, i):
            line = (f"{examples.ligand_smiles.iloc[i]} "
                    f"{examples.target_sequence.iloc[i]}")

            nm = None
            if not pd.isna(examples.ki.iloc[i]):
                nm = examples.ki.iloc[i]
            elif not pd.isna(examples.ic50.iloc[i]):
                nm = examples.ic50.iloc[i]
            elif not pd.isna(examples.kd.iloc[i]):
                nm = examples.kd.iloc[i]
            elif not pd.isna(examples.ec50.iloc[i]):
                nm = examples.ec50.iloc[i]

            interactivity = Protein.get_interactivity(nm)

            if interactivity is not None:
                return f"{line} {interactivity}"
            else:
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


    def get_sim(self, other_name):
        for key in self.sims.keys():
            if key in other_name:
                return self.sims[key]
        raise Exception(f"{other_name} not found in self.sims")


    def get_dissim_names(self):
        return [other_name for other_name in self.sims.keys()
                if self.sims[other_name] <= SHARING_DISSIM_THRESHOLD]


    def add_inactives(self, proteins_list):
        def modify_actives(other_actives):
            return [f"{active.split()[0]} {self.sequence} 0"
                    for active in other_actives]

        for protein in proteins_list:
            if [name for name in self.get_dissim_names()
                if name in protein.get_name()]:
                self.examples += modify_actives(protein.get_actives())


    # For resampling.
    def resample_inactives(self):
        if self.get_ratio() < 50:
            return None

        inactives = self.get_inactives()
        np.random.seed(SHUFFLE_SEED)
        np.random.shuffle(inactives)

        actives = self.get_actives()

        self.examples = actives + inactives[:(len(actives) * 50)]


    # Miscellaneous getters.
    def get_dude_sim_mean(self):
        return np.nanmean([self.get_sim(other_name) for other_name
                           in self.sims.keys()
                           if not Protein.is_bindingdb(other_name) and
                           self.name != other_name])


    def get_examples(self):
        return self.examples


    def get_actives(self):
        return [example for example in self.examples
                if int(example.split()[2])]


    def get_inactives(self):
        return [example for example in self.examples
                if not int(example.split()[2])]


    def get_ratio(self):
        return len(self.get_inactives()) / len(self.get_actives())


    def get_name(self):
        return self.name


    def get_len(self):
        return len(self.sequence)


    def __repr__(self):
        return f"{self.name[:4]} {Protein.is_bindingdb(self.name)}"

