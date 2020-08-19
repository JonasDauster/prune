import pandas as pd
import numpy as np
from keras.models import Sequential
import tensorflow as tf
import os
from Bio import SeqIO
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence

# load and preprocess
# insert here the csv created via TrainingDataCreation.py
df = pd.read_csv("inserted.csv")
df.columns = ['seq_label','sequence']
print(df['sequence'])
from textwrap import wrap

# cut to kmers
kmer_size = 1
#cut to kmers
df['sequence'] = df.apply(lambda x: wrap(x['sequence'], kmer_size), axis=1)
df['sequence'] = [','.join(map(str, l)) for l in df['sequence']]
max_length = df.sequence.map(lambda x: len(x)).max()
max_length = max_length/kmer_size

df['sequence'] = df.apply(lambda x: text_to_word_sequence(x['sequence'], split=','), axis=1)
df['sequence'] = df['sequence'].astype(str)
vocab_max = 4 ** kmer_size
print(vocab_max)
# integer encode the document
df['sequence'] = df.apply(lambda x: one_hot(x['sequence'], round(vocab_max)), axis=1)

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
dummy_y = np_utils.to_categorical(encoded_Y)

target_softmax = dummy_y

from sklearn.utils import class_weight
#can be used for fixed length shorter than the longest fragment
max_lengthtest = 150
train_numpybig = df["sequence"].values
train_numpybig  = sequence.pad_sequences(train_numpybig,max_length,padding='post',truncating='post')
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

