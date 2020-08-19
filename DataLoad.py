#start this and also the CNN file with the same random seed or else the one hot encoding will be diffrent
#in a command line interface you can do this with PYTHONHASHSEED=6 python3 nameofscript.py 
# you can enter any number behind the hashseed, just stick to the same one for both files

import numpy as np
from keras.models import Sequential
import tensorflow as tf
import os
from Bio import SeqIO
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
import pandas as pd
from keras.preprocessing import sequence

# load and preprocess
# insert the data you want the network to classify here
df = pd.read_csv("real_data.csv")
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
max_length = max_length/kmer_size
df['sequence'] = df.apply(lambda x: text_to_word_sequence(x['sequence'], split=','), axis=1)
df['sequence'] = df['sequence'].astype(str)
# vocab_max is the number of possible words/sequences
vocab_max = 4 ** kmer_size
print(vocab_max)
# integer encode the document
df['sequence'] = df.apply(lambda x: one_hot(x['sequence'], round(vocab_max)), axis=1)
print(df['sequence'])

from sklearn.utils import shuffle, compute_class_weight

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
dataset = df.values
Y = dataset[:,0]
encoder_label = LabelEncoder()
encoder_label.fit(Y)
encoded_Y = encoder_label.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

target_softmax = dummy_y

from sklearn.utils import class_weight
print(df.head(100))
#max_lengthtest can be used insted of max_length to cut fragments shorter than the longest one
max_lengthtest = 150
train_numpybig = df["sequence"].values
train_numpybig  = sequence.pad_sequences(train_numpybig,max_length,padding='post',truncating='post')

# load model
from tensorflow import keras
model = keras.models.load_model('modelconv.h5')
model.summary()


prediction = model.predict(train_numpybig)
print(prediction)
rnndf = []
rnn_nodf = []
score = 0
for i in range(len(Ori)):
    print(Ori[i])
    print('%s => %d' % (train_numpybig[i].tolist(), prediction[i].argmax()))
    print(prediction[i])
    if encoded_Y[i] == prediction[i].argmax():
        score=score+1
    if prediction[i].argmax() == 0:
        rnndf.append(Ori[i])
        print("found something")
    if prediction[i].argmax() == 1:
        rnn_nodf.append(Ori[i])
        print("nothing found")
    else:
        print("nothing found")


#only if correct labels are provided
print("Score")
print((score/len(Ori))*100)
df=pd.DataFrame(rnndf)
df_no = pd.DataFrame(rnn_nodf)
#file contains the lines with the sequences searched for
df.to_csv("network_hits.csv",index=False)
#file contains the others
df_no.to_csv("no_hits.csv",index=False)
