# import collections
# import pathlib
# import re
# import string

# import tensorflow as tf

import numpy as np
import pandas as pd
import time
global last_log_time
last_log_time = time.time()

import json

from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from tensorflow.keras import layers
# from tensorflow.keras import losses
# from tensorflow.keras import preprocessing
# from tensorflow.keras import utils
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# import tensorflow_datasets as tfds

from mydata import token
from mydata import true

def log(message):
    global last_log_time
    current_time = time.time()
    time_difference = current_time - last_log_time
    print(message, " | Time Elapsed: %.4f" % (time_difference))
    last_log_time = current_time

log("start")

data = 0
with open('./jsonfile.json') as json_file:
    data = json.load(json_file)

log("loaded json")

# data['input']

model = keras.models.load_model('./model-to-py/model-to-py.h5')

# log("loaded model")

# # df = pd.read_csv('Revised.csv')
# true = pd.read_csv('./True.csv')
# # true = true.head(5000) # REMOVE/COMMENT THIS LINE IF YOU WANT TO TAKE THE ENTIRE CSV
# fake = pd.read_csv('./Fake.csv')
# # fake = fake.head(5000) # REMOVE/COMMENT THIS LINE IF YOU WANT TO TAKE THE ENTIRE CSV
# # print(true.keys())
# # print(fake.keys())
# # log("Read input files")

# log("loaded csvs")

# true['truth'] = 1
# fake['truth'] = 0

# # Combining the title and text columns on both true and false csvs so that the input data is simplified.
# true['article'] = true['title'] + " " + true['text'] 
# fake['article'] = fake['title'] + " " + fake['text'] 

# log("created extra columns in csvs")

# combined_data = pd.concat([fake, true])

# log("combined csvs")

# # log("Combined data")

# features = combined_data['article']
# labels = combined_data['truth']

# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)

# log("split data")

# log("Split data into Training and Testing")

max_words = 2000
max_len = 800

# token = Tokenizer(num_words=max_words, lower=True, split=' ')
# log("created tokenizer")
# token.fit_on_texts(x_train.values)
# log("fit tokenizer on texts")
sequences = token.texts_to_sequences([true['article'].iloc[0]])
log("converted texts to sequences with tokenizer")
# print(true['article'].iloc[0])
train_sequences_padded = pad_sequences(sequences, maxlen=max_len)
log("padded sequences")
# print(train_sequences_padded)

print(model.predict(train_sequences_padded)[0][0])
log("created model prediction")