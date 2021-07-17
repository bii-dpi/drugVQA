def get_fasta(path):
    with open(path, "r") as f:
        text = f.readlines()
    text = [line.split(":") for line in text]
    text = [[line[0], line[1].split("_")[0]] for line in text]
    text = [f">{line[1]}\n{line[0]}" for line in text]
    return text


bindingdb_fasta = get_fasta("BindingDB/contact_map/BindingDB_contactdict")
dude_fasta = get_fasta("DUDE/contact_map/DUDE_contactdict")

with open("Shallow/fastas_bindingdb", "w") as f:
    f.write("\n".join(bindingdb_fasta))

with open("Shallow/fastas_dude", "w") as f:
    f.write("\n".join(dude_fasta))

with open("all_fasta", "w") as f:
    f.write("\n".join(dude_fasta + bindingdb_fasta))

