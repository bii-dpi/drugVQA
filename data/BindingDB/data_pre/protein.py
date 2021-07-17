import os
import pickle

import numpy as np
import pandas as pd


SHUFFLE_SEED = 12345


def get_contactdict(fname):
    with open(path, "r") as f:
        sequence_to_id_dict = f.readlines()

    return {line.split(":")[0]:line.split(":")[1].strip("\n")
            for line in sequence_to_id_dict}


# Data
bindingdb_examples = pd.read_csv("bindingdb_examples.tsv", sep="\t")


## Sequence-to-cm ID.
bindingdb_contactdict = get_contactdict("../contact_map/BindingDB-contactDict")
bindingdb_contactdict = get_contactdict("../../DUDE/contact_map/DUDE-contactDict")

## Name-to-similarities.
sim_matrix = pd.read_pickle("sim_matrix.pkl")


# Protein class definition
class Protein:
    def __init__(self, pdb_id):
        # Descriptors
        self.pdb_id = pdb_id  # These are always full names.
        self.sequence = Protein.get_sequence(pdb_id)
        # Group membership
        self.is_bindingdb = Protein.is_bindingdb(pdb_id)
        # Other data
        self.set_examples()


    # Used in initialization.
    @staticmethod
    def get_sequence(pdb_id):
        try:
            return bindingdb_dict[pdb_id]
        except:
            return dude_dict[pdb_id]


    @staticmethod
    def is_bindingdb(pdb_id):
        if pdb_id in bindingdb_dict.key():
            return True
        return False


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
        return None


    # Used in setting and adding examples.
    def set_actives(self):
        def process_row(examples, i):
            line = (f"{examples.ligand_smiles.iloc[i]} "
                    f"{examples.target_sequence.iloc[i]}")

            nm = None
            if not pd.isna(actives.ki.iloc[i]):
                nm = actives.ki.iloc[i]
            elif not pd.isna(actives.ic50.iloc[i]):
                nm = actives.ic50.iloc[i]
            elif not pd.isna(actives.kd.iloc[i]):
                nm = actives.kd.iloc[i]
            elif not pd.isna(actives.ec50.iloc[i]):
                nm = actives.ec50.iloc[i]

            interactivity = Protein.get_interactivity(nm)

            if interactivity is not None:
                return f"{line} {interactivity}"
            else:
                return ""

        self.actives = bindingdb_examples[
                                bindingdb_examples.\
                                target_sequence ==
                                self.sequence
                              ]
        self.actives = [process_row(self.actives, i)
                         for i in range(len(self.actives))]
        self.actives = [active for active in self.actives
                         if active]


    def get_sim(self, other_pdb_id):
        return self.sims[other_pdb_id]


    # Miscellaneous getters.
    def get_dude_sim_mean(self):
        return np.nanmean([self.get_sim(other_pdb_id) for other_pdb_id
                           in self.sims.keys()
                           if not Protein.is_bindingdb(other_pdb_id) and
                           other_pdb_id != self.pdb_id])


    def get_sims(self, proteins_list):
        return [self.get_sim(protein.get_pdb_id())
                for protein in proteins_list
                if protein != self]


    def get_actives(self):
        return self.actives


    def get_pdb_id(self):
        return self.pdb_id


    def get_len(self):
        return len(self.sequence)


    def __eq__(self, other):
        if isinstance(other, Protein):
            return self.pdb_id == other.get_pdb_id()
        return False


    def __repr__(self):
        return f"{self.pdb_id[:4]} {Protein.is_bindingdb(self.pdb_id)}"

