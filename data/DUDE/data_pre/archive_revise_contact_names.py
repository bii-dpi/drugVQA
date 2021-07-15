import os

# Write new contact dict.
with open("DUDE-contactDict", "r") as f:
    text = [line.split(":") for line in f.readlines()]
text = [line[0] + ":" + line[1].split("_")[1][:-1].upper() + "_cm"
        for line in text]

with open("DUDE_contactdict", "w") as f:
    f.write("\n".join(text))

# Change name of contact maps.
def get_new_name(cmap_fname):
    return cmap_fname.split("_")[1][:-1].upper() + "_cm"

cmap_fnames = [fname for fname in os.listdir("../contact_map")
               if fname.endswith("_full") ]
new_fnames = [get_new_name(fname) for fname in cmap_fnames]

for i in range(len(cmap_fnames)):
    os.system(f"mv ../contact_map/{cmap_fnames[i]} ../contact_map/{new_fnames[i]}")

