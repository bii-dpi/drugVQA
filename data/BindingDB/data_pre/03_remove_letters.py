NUM_SELECTED = -1
ILLEGAL_LIST = ["[c-]", "[N@@]"]


def has_illegal(line):
    smiles_string = line.split()[0]
    for illegal_element in ILLEGAL_LIST:
        if illegal_element in smiles_string:
            return True
    return False


with open(f"new_bindingdb_examples_{NUM_SELECTED}", "r") as f:
    text = f.readlines()

print(len(text))
text = [line for line in text if not has_illegal(line)]
print(len(text))

with open(f"new_bindingdb_examples_{NUM_SELECTED}", "w") as f:
    f.writelines(text)
