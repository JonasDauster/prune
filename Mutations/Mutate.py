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
#change number for distance form start and end where the mutaded seqeunces begin
# with 3 it is CTG TCTCTTATACACA TCT this middle part
for chunck in mutate_list("CTGTCTCTTATACACATCT",3):
    chunck_list.append(chunck)

print(len(chunck_list))
#mutated seqeuence for new adapter creation and training
f=open('Mutated.txt','w')
w='\n'.join(chunck_list)
f.write(w)
f.close()
