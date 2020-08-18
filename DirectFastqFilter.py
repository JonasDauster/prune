from Bio import SeqIO
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
# enter the fastq you want to filter in the lines below, two times
positive_list = positive("NCFB_1000_seqs.fastq")
records = list(SeqIO.parse("NCFB_1000_seqs.fastq", "fastq"))
for x in range(len(positive_list)):
    to_insert = positive_list[x]
    #to_insert = random.choice(positive_list)
    sequence.append(to_insert)
    if x == 1:
        label.append("insert")
    else:
        label.append("no_insert")




insert_df = pd.DataFrame(sequence,label,columns=["sequence"])
print(insert_df)
insert_df.to_csv("test_test.csv")

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
import pandas as pd
# fix random seed for reproducibility
#np.random.seed(5)
from keras.preprocessing import sequence

# load and preprocess
df = pd.read_csv("test_test.csv")
df.columns = ['seq_label','sequence']
print(df['sequence'])
Ori = df['sequence'].tolist()
Orilabel = df["seq_label"].tolist()
from textwrap import wrap

# cut to kmers
kmer_size = 1
#cut to kmers
df['sequence'] = df.apply(lambda x: wrap(x['sequence'], kmer_size), axis=1)

print(df.dtypes)
print(df['sequence'])
print(df['seq_label'].value_counts().sort_index())

df['sequence'] = [','.join(map(str, l)) for l in df['sequence']]
print("possible Form")
print(df.head(100))
max_length = df.sequence.map(lambda x: len(x)).max()
print(max_length)
max_length = max_length/kmer_size
df['sequence'] = df.apply(lambda x: text_to_word_sequence(x['sequence'], split=','), axis=1)
df['sequence'] = df['sequence'].astype(str)
vocab_max = 4 ** kmer_size
print(vocab_max)
# integer encode the document
df['sequence'] = df.apply(lambda x: one_hot(x['sequence'], vocab_max ), axis=1)
print(df['sequence'])

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
dataset = df.values
Y = dataset[:,0]
print(Y)
encoder_label = LabelEncoder()
encoder_label.fit(Y)
encoded_Y = encoder_label.transform(Y)
print(encoded_Y)
dummy_y = np_utils.to_categorical(encoded_Y)

target_softmax = dummy_y

print(df.head(100))
max_lengthtest = 150
train_numpybig = df["sequence"].values
train_numpybig  = sequence.pad_sequences(train_numpybig,max_lengthtest,padding='post',truncating='post')

print(target_softmax)
print(train_numpybig)

# load model
from tensorflow import keras
model = keras.models.load_model('modelconvmut.h5')
model.summary()


prediction = model.predict(train_numpybig)
print(prediction)
rnndf = []
score = 0
newrecord = []
for i in range(len(Ori)):
    print(Orilabel[i])
    print(records[i].id)
    print(Ori[i])
    print(records[i].seq)
    print('%s => %d' % (train_numpybig[i].tolist(), prediction[i].argmax()))
    print(prediction[i])
    if encoded_Y[i] == prediction[i].argmax():
        score=score+1
    if prediction[i].argmax() == 0:
        rnndf.append(Ori[i])
    if prediction[i].argmax() == 1:
        newrecord.append(records[i])
    else:
        print("no contamination")


print(score/len(Ori))
df=pd.DataFrame(rnndf)
print(newrecord)
print(records)
# fastq without serached for seqeunce
SeqIO.write(newrecord, "example.fastq", "fastq")
# all lines containig the searched for seqeunce
df.to_csv("/working2/rcug_lw/pythonProjects/mistgabel/Jonas/to_check.csv",index=False)
