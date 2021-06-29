from protein import *

NUM_EXAMPLES = 10

bindingdb_proteins = [Protein(name) for name in list(bindingdb_dict.keys())]
print(len(bindingdb_proteins))
bindingdb_proteins = [Protein(name) for name in list(bindingdb_dict.keys())
                      if Protein.cm_exists(name)]
print(len(bindingdb_proteins))
bindingdb_proteins = [protein for protein in bindingdb_proteins
                      if len(protein.get_examples()) >= 50]
dude_proteins = [Protein(name) for name in list(dude_dict.keys())]

dude_sim_means = [protein.get_dude_sim_mean() for protein in bindingdb_proteins]

selected_proteins = \
    np.array([protein for protein in bindingdb_proteins])\
    [np.argsort(dude_sim_means)][:NUM_EXAMPLES]

#[protein.add_actives(bindingdb_proteins) for proteins in

print([protein.get_ratio() for protein in selected_proteins])
