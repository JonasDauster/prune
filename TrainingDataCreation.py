import glob
from Bio import SeqIO
import random
import pandas as pd

def slidingWindow(sequence, winSize, step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize) / step) + 1

    # Do the work
    for i in range(0, int(numOfChunks) * step, step):
        yield sequence[i:i + winSize]
        

def positive(filepath):
    rlist1 = []
    fastq_sequences = SeqIO.parse(open(filepath), 'fastq')
    for fastq in fastq_sequences:
        sequence = str(fastq.seq)
        rlist1.append(sequence)


    return rlist1


def findFiles(path): return glob.glob(path)


filelist = []

#folder with your seqeunces to identify ( example in example data) loads all txt files in folder so multiple can be given
for file in findFiles('Nextera/*.txt'):
    filelist.append(file)

sequence=[]
label=[]
#folder with your normal/backround sequences
positive_list = positive("train_5000_reads_per_bam.fastq")
#change to get more or less training data 
sample_number = 400

for x in range(sample_number):
    r_file = random.choice(filelist)
    lines = open(r_file).read().strip().split('\n')
    to_insert = random.choice(positive_list)
    sequence.append(to_insert)
    label.append("no_insert")
    adapter_fragment = random.choice(lines)
    length = len(adapter_fragment)

    chunk_list = []
    for chunks in slidingWindow(to_insert, length, 1):
        chunk_list.append(chunks)

    print(to_insert)
    r_chunk = random.choice(chunk_list)
    new_line = to_insert.replace(r_chunk, adapter_fragment)
    print(new_line)
    sequence.append(new_line)
    label.append("insert")



insert_df = pd.DataFrame(sequence,label,columns=["sequence"])
print(insert_df)
# file for the neural network
insert_df.to_csv("inserted.csv")
