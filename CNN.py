import pandas as pd
import numpy as np
from keras.models import Sequential
import tensorflow as tf
import os
from Bio import SeqIO
from keras.preprocessing import sequence

# load and preprocess
df = pd.read_csv("inserted.csv")
df.columns = ['seq_label','sequence']
print(df['sequence'])
from textwrap import wrap

# cut to kmers
kmer_size = 1
#cut to kmers
df['sequence'] = df.apply(lambda x: wrap(x['sequence'], kmer_size), axis=1)

#to ensure keras one_hot encodes always the same
# when starting script a Hash seed should be set so that the output is the same over mulitple files , set to fixed seed when starting via PHYTONSEED=0 and than phyton3 and programm

print(df.dtypes)
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
df['sequence'] = df.apply(lambda x: one_hot(x['sequence'], round(vocab_max)), axis=1)
df['sequence_cut1'] = df.apply(lambda x: one_hot(x['sequence_cut1'], round(vocab_max1)), axis=1)
#df['seq_label'] = pd.Categorical(df['seq_label'])
#df[seq_'label'] = df.label.cat.codes
from sklearn.utils import shuffle, compute_class_weight

#shuffel gets some more accuracy
df = shuffle(df)

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
dataset = df.values
Y = dataset[:,0]
encoder_label = LabelEncoder()
encoder_label.fit(Y)
encoded_Y = encoder_label.transform(Y)
print(encoded_Y)
dummy_y = np_utils.to_categorical(encoded_Y)

target_softmax = dummy_y

from sklearn.utils import class_weight
#class_weights = class_weight.compute_class_weight('balanced',np.unique(Y),Y)
max_lengthtest = 150
train_numpy = pd.DataFrame(df['sequence_cut1'].values.tolist()).values
#train_numpybig = pd.DataFrame(df['sequence'].values.tolist()).values
train_numpybig = df["sequence"].values
train_numpybig  = sequence.pad_sequences(train_numpybig,max_lengthtest,padding='post',truncating='post')
#with np.printoptions(threshold=np.inf):
   # print(train_numpybig, file=open('testout','a'))
print(target_softmax)
print(train_numpybig)



from tensorflow.keras import layers

# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(vocab_max,64)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 10, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 10, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(target_softmax.shape[1], activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#train with full data and add validation split in fit or use test set
model.fit(train_numpybig,target_softmax,epochs=10,batch_size=2)

scores = model.evaluate(train_numpybig,target_softmax, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("modelconv.h5")

