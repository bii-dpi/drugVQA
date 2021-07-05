import os

from Bio.PDB import *


pdbl = PDBList()

pdb_ids = [fname.split("_")[1][:-1] for fname in os.listdir()
           if fname.endswith("_full")]

print(pdb_ids)

pdbl.download_pdb_files(pdb_ids, file_format="pdb", pdir="pdb_files")

