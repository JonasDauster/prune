import random
import pandas as pd

def mutate_v1(dna):
    dna_list = list(dna)
    mutation_site = random.randint(0, len(dna_list) - 1)
    dna_list[mutation_site] = random.choice(list('ATCG'))
    return ''.join(dna_list)

def mutate_list(dna,space):
    dna_list = list(dna)
    for x in range(space,len(dna_list)-space):
        dna_list =list(dna)
        mutation_site = x
        dna_list[mutation_site] = random.choice(list('ATCG'))
        yield ''.join(dna_list)


chunck_list = []
for chunck in mutate_list("CTGTCTCTTATACACATCT",3):
    chunck_list.append(chunck)

print(len(chunck_list))
df = pd.DataFrame(chunck_list)
df.to_csv("Mutated.csv")