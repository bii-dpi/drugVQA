import os
import pandas as pd


contact_dict = pd.read_pickle("sequence_to_id_map.pkl")

text = ""
for sequence in contact_dict.keys():
    if f"{contact_dict[sequence]}_cm" in os.listdir():
        text += f"{sequence}:{contact_dict[sequence]}_cm\n"

with open("BindingDB_contactdict", "w") as f:
    f.write(text)

