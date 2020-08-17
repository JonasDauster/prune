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
# if you want to load real data it should before go through the same process as training data
# an seperated file for that will be added shortly
df = pd.read_csv("inserted_test.csv")
df.columns = ['seq_label','sequence']
print(df['sequence'])
Ori = df['sequence'].tolist()
Orilabel = df["seq_label"].tolist()
from textwrap import wrap

# cut to kmers
kmer_size = 1
#cut to kmers
df['sequence'] = df.apply(lambda x: wrap(x['sequence'], kmer_size), axis=1)

#to ensure keras one_hot encodes always the same
# when starting script a Hash seed should be set so that the output is the same over mulitple files , set to fixed seed above
# labels = df[['seq_label']].values.tolist()
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
# sequence cut 1 is just the first 16mers of ervery sample, because it is almost impossible to process uneven data with keras
# if the data is scaled beforehand there shoul dbe no need for this column(need to think about how we make the data even, and also shuffel it,also imbalenced classes can pose a problem)
cut_off = 3
df.loc[:, 'sequence_cut1'] = df.sequence.map(lambda x: x[0:3])
df['sequence'] = df['sequence'].astype(str)
df['sequence_cut1'] = df['sequence_cut1'].astype(str)
# vocab_max is the number of possible words/sequences, multiplication by  1.3 is normally just applied for natrual languages could be removed
vocab_max = 4 ** kmer_size
vocab_max1 = 4 ** kmer_size
print(vocab_max)
print(vocab_max1)
# integer encode the document
df['sequence'] = df.apply(lambda x: one_hot(x['sequence'], round(vocab_max * 1, 3)), axis=1)
df['sequence_cut1'] = df.apply(lambda x: one_hot(x['sequence_cut1'], round(vocab_max1)), axis=1)
print(df['sequence'])
#df['seq_label'] = pd.Categorical(df['seq_label'])
#df[seq_'label'] = df.label.cat.codes
from sklearn.utils import shuffle, compute_class_weight

#shuffel gets some more accuracy
#df = shuffle(df)
#joined = Ori.merge(df, left_on='New_ID', right_on='New_ID')
#joined.to_csv("/working2/rcug_lw/pythonProjects/mistgabel/Jonas/joined.csv")

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

from sklearn.utils import class_weight
#class_weights = class_weight.compute_class_weight('balanced',np.unique(Y),Y)
print(df.head(100))
max_lengthtest = 150
train_numpy = pd.DataFrame(df['sequence_cut1'].values.tolist()).values
#train_numpybig = pd.DataFrame(df['sequence'].values.tolist()).values
train_numpybig = df["sequence"].values
train_numpybig  = sequence.pad_sequences(train_numpybig,max_lengthtest,padding='post',truncating='post')

# load model
from tensorflow import keras
model = keras.models.load_model('modelconv.h5')
model.summary()
#scores = model.evaluate(train_numpybig,target_softmax, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


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
print(score/len(Ori))
df=pd.DataFrame(rnndf)
df_no = pd.DataFrame(rnn_nodf)
#file contains the lines with the sequences searched for
df.to_csv("network_hits.csv",index=False)
#file contains the others
df_no.to_csv("no_hits.csv",index=False)
