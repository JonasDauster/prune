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


sequence=[]
label=[]
positive_list = positive("NCFB_D_MG000106_2020_R1.fastq")
for x in range(5000):
    to_insert = random.choice(positive_list)
    sequence.append(to_insert)
    if x == 1:
        label.append("insert")
    else:
        label.append("no_insert")




insert_df = pd.DataFrame(sequence,label,columns=["sequence"])
print(insert_df)
insert_df.to_csv("test.csv")
