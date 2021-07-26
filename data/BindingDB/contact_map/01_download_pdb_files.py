import pandas as pd

from Bio.PDB import *
from pickle import dump


pdb_ids = pd.read_pickle("sequence_to_id_map.pkl").values()

# Download PDBs.
pdbl = PDBList()

pdbl.download_pdb_files(pdb_ids, file_format="pdb", pdir="pdb_files")

