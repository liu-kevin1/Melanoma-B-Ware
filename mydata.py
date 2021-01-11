# import collections
# import pathlib
# import re
# import string
import time

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

global last_log_time
last_log_time = time.time()

def log(message):
    global last_log_time
    current_time = time.time()
    time_difference = current_time - last_log_time
    print(message, " | Time Elapsed: %.4f" % (time_difference))
    last_log_time = current_time

log("Starting")
# def import_data():
#     # obj = json.loads('./Revised.json').decode('utf-8')
#     # bbox = obj['bounding_box']
#     # return np.array([bbox['x'], bbox['y'], bbox['height'], bbox['width']], dtype='f')
#     with open('./Revised.json') as json_file:
#         data = json.load(json_file)
#         # print(data[0]['text'])
#         # print(data)
#         for line in data:
#             # for p in line:
#             print(line)
#         # print(np.array(data))

# import_data()

# df = pd.read_csv('Revised.csv')
true = pd.read_csv('./True.csv')
# true = true.head(5000) # REMOVE/COMMENT THIS LINE IF YOU WANT TO TAKE THE ENTIRE CSV
fake = pd.read_csv('./Fake.csv')
# fake = fake.head(5000) # REMOVE/COMMENT THIS LINE IF YOU WANT TO TAKE THE ENTIRE CSV
# print(true.keys())
# print(fake.keys())
log("Read input files")

true['truth'] = 1
fake['truth'] = 0

# Combining the title and text columns on both true and false csvs so that the input data is simplified.
true['article'] = true['title'] + true['text'] 
fake['article'] = fake['title'] + fake['text'] 

combined_data = pd.concat([fake, true])

log("Combined data")

features = combined_data['article']
labels = combined_data['truth']

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)

log("Split data into Training and Testing")

max_words = 2000
max_len = 400

token = Tokenizer(num_words=max_words, lower=True, split=' ')
token.fit_on_texts(x_train.values)
sequences = token.texts_to_sequences(x_train.values)
train_sequences_padded = pad_sequences(sequences, maxlen=max_len)

log("Tokenized data")

# features = df.copy()
# labels = df.pop('truth')

# df.pop('index')

# dataset = tf.data.Dataset.from_tensor_slices((df.values, labels.values))

# for feat, label in dataset.take(5):
#   print ('Features: {}, Labels: {}'.format(feat, label))

# # using sklearn's train_test_split function below (test_size=0.25 means test set will be 1/4 size portion of training set)
# x_train, x_test, y_train, y_test = train_test_split(df['text'], df['truth'], test_size=0.25, shuffle=True)

# # datatypes of all four of these sets are a pandas "Series" which act very similar to Python lists, but I think the indices are inconsistent

# # seeing what our sets would print (x sets should print text, while y sets should print labels aka truth)

# print("-" * 10)
# print(sequences)
# print("-" * 10)
# print(train_sequences_padded)
# print("-" * 10)
# print(x_train[0], y_train[0])
# print("\n" + "----------------------------------------" + "\n")
# print(x_test[0], y_test[0])

model = keras.Sequential()
model.add(keras.layers.Embedding(max_words, 16, input_length=max_len))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(256, activation='softmax'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

log("Model created")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_sequences_padded, y_train, batch_size=32, epochs=5, validation_split=0.75)

log("Training finished")

# Using the tokenizer for our test data
token.fit_on_texts(x_test.values)
sequences = token.texts_to_sequences(x_test.values)
train_sequences_padded = pad_sequences(sequences, maxlen=max_len)

results = model.evaluate(train_sequences_padded, y_test)

log("Testing finished")

print("Loss: %.5f\nAccuracy: %.5f" % (results[0], results[1]))

token = Tokenizer(num_words=max_words, lower=True, split=' ')
sequences = token.texts_to_sequences("test one two there trump bad smh fake news")
train_sequences_padded = pad_sequences(sequences, maxlen=max_len)

print(model.predict(train_sequences_padded)[0][0])