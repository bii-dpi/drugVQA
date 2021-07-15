import os

from Bio.PDB import *


pdbl = PDBList()

pdb_ids = [fname.split("_")[0] for fname in os.listdir()
           if fname.endswith("_cm")]

pdbl.download_pdb_files(pdb_ids, file_format="pdb", pdir="pdb_files")

