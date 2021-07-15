import os
import glob


def rename_file(fname):
    no_prefix_fname = "_".join(fname.split("_")[1:])
    os.rename(fname, "orig_cv_" + no_prefix_fname)


all_files = [file for file in glob.glob("orig_*")
                if not file.startswith("orig_rv")]

for fname in all_files:
    rename_file(fname)
