with open("all_fastas", "r") as f:
    text = f.readlines()

fasta_dict = dict(zip([text[i] for i in range(0, len(text), 2)],
                      [text[i] for i in range(1, len(text), 2)]))

all_fastas = [f"{key}{value}" for key, value in fasta_dict.items()]
with open("all_fastas_proc", "w") as f:
    f.writelines(all_fastas)

