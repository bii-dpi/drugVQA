import os
import pandas as pd


def get_seq(fname):
    with open(fname, "r") as f:
        return f.read().split("\n")[1]


"""
cm_files = [fname for fname in os.listdir()
            if fname.endswith("_cm")]

text = ""
for fname in cm_files:
    text += f"{get_seq(fname)}:{fname}\n"

with open("BindingDB-contactDict", "w") as f:
    f.write(text)

"""

contact_dict = pd.read_pickle("sequence_to_id_map.pkl")
contact_dict = {sequence: ids.split(",")[0] for
                sequence, ids in contact_dict.items()}

text = ""
for sequence in contact_dict.keys():
    if f"{contact_dict[sequence]}_cm" in os.listdir():
        text += f"{sequence}:{contact_dict[sequence]}_cm\n"

with open("BindingDB-contactDict", "w") as f:
    f.write(text)
