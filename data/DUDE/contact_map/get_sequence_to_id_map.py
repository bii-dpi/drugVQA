import os
import pickle


def get_sequence(fname):
    with open(fname, "r") as f:
        return f.readlines()[1].strip("\n")


fnames = [fname for fname in os.listdir() if fname.endswith("_full")]
sequence_to_id = {get_sequence(fname): fname.split("_")[1][:-1] for
                  fname in fnames}

with open("sequence_to_id_map.pkl", "wb") as f:
    pickle.dump(sequence_to_id, f)

