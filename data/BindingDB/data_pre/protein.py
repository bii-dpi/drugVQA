import os
import pickle

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs


SHUFFLE_SEED = 12345


def get_dict(path):
    with open(path, "r") as f:
        sequence_to_id_dict = f.readlines()

    return {line.split(":")[1].strip("_cm\n"):line.split(":")[0]
            for line in sequence_to_id_dict}


# Data
actives = pd.read_pickle("actives.pkl")

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
        return pdb_id in bindingdb_dict.keys()


    # Used in setting and adding examples.
    def set_actives(self):
        def process_row(example):
            return " ".join(example[:2]) + " 1"


        def get_lig_sim(candidate_active, active):
             candidate_fprint = Chem.RDKFingerprint(Chem.MolFromSmiles(candidate_active))
             active_fprint = Chem.RDKFingerprint(Chem.MolFromSmiles(active))
             return DataStructs.FingerprintSimilarity(candidate_fprint,
                                                      active_fprint)


        def get_max_sim(candidate_active, actives_to_keep):
            if not actives_to_keep:
                return 0
            return np.max([get_lig_sim(candidate_active, active)
                           for active in actives_to_keep])


        self.actives = [process_row(active) for active in actives
                        if active[2] == self.pdb_id]

        np.random.shuffle(self.actives)

        actives_to_keep = []
        for candidate_active in self.actives:
            if len(actives_to_keep) == 1000:
                break
            if get_max_sim(candidate_active, actives_to_keep) < 0.7:
                actives_to_keep.append(candidate_active)
        self.actives = actives_to_keep


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



