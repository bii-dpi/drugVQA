import os
import pickle

import numpy as np
import pandas as pd


SHUFFLE_SEED = 12345
ILLEGAL_LIST = ["[c-]", "[N@@]", "[Re-]", "[S@@+]", "[S@+]"]


def get_dict(path):
    with open(path, "r") as f:
        sequence_to_id_dict = f.readlines()

    return {line.split(":")[1].strip("_cm\n"):line.split(":")[0]
            for line in sequence_to_id_dict}


# Data
bindingdb_examples = pd.read_csv("bindingdb_examples.csv")

bindingdb_dict = get_dict("../contact_map/BindingDB_contactdict")
dude_dict = get_dict("../../DUDE/contact_map/DUDE_contactdict")

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
        self.sims = sim_matrix[pdb_id]
        self.set_actives()


    # Used in initialization.
    @staticmethod
    def get_sequence(pdb_id):
        try:
            return bindingdb_dict[pdb_id]
        except:
            return dude_dict[pdb_id]


    @staticmethod
    def is_bindingdb(pdb_id):
        if pdb_id in bindingdb_dict.keys():
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


        def has_illegal(active):
            smiles_string = active.split()[0]
            for illegal_element in ILLEGAL_LIST:
                if illegal_element in smiles_string:
                    return True
            return False


        self.actives = bindingdb_examples[
                                bindingdb_examples.\
                                target_sequence ==
                                self.sequence
                              ]
        self.actives = [process_row(self.actives, i)
                         for i in range(len(self.actives))]
        self.actives = [active for active in self.actives
                         if active and not has_illegal(active)]


    # Miscellaneous getters.
    def get_actives(self):
        return self.actives


    def get_sim(self, other_pdb_id):
        return self.sims[other_pdb_id]


    def get_dude_sim_mean(self):
        return np.nanmean([self.get_sim(other_pdb_id)
                           for other_pdb_id in self.sims.keys()
                           if not Protein.is_bindingdb(other_pdb_id)
                           and other_pdb_id != self.pdb_id])


    def get_sims(self, proteins_list):
        return [self.get_sim(protein.get_pdb_id())
                for protein in proteins_list
                if protein != self]


    def get_pdb_id(self):
        return self.pdb_id


    def get_len(self):
        return len(self.sequence)


    def __eq__(self, other):
        if isinstance(other, Protein):
            return self.pdb_id == other.get_pdb_id()
        return False


    def __repr__(self):
        return f"{self.pdb_id} {self.is_bindingdb}"

