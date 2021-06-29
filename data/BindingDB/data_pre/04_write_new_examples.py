from protein import *

NUM_EXAMPLES = 10

bindingdb_proteins = [Protein(name) for name in list(bindingdb_dict.keys())]
dude_proteins = [Protein(name) for name in list(dude_dict.keys())]

dude_sim_means = [protein.get_dude_sim_mean() for protein in bindingdb_proteins]
print(dude_sim_means)

selected_proteins = \
    np.array([protein for protein in proteins_list])\
    [np.argsort(dude_sim_means)][:NUM_EXAMPLES]

print([protein.get_dude_sim_mean() for protein in selected_proteins])
