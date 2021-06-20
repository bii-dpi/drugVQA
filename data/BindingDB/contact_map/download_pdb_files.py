import pandas as pd

from Bio.PDB import *


pdbl = PDBList()

bindingdb_sequences = pd.read_pickle("../data_pre/mapped_bindingdb_sequences.pkl")
bindingdb_sequences = [element.split("\n")[2] for element in bindingdb_sequences]

sequence_mapping = pd.read_pickle("sequence_to_id_map.pkl")

pdb_ids = [sequence_mapping[sequence].split(",")[0]
            for sequence in bindingdb_sequences
            if sequence in sequence_mapping.keys()]

pdbl.download_pdb_files(pdb_ids, file_format="pdb", pdir="pdb_files")

