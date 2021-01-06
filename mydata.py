# import collections
# import pathlib
# import re
# import string

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import json

# from tensorflow.keras import layers
# from tensorflow.keras import losses
# from tensorflow.keras import preprocessing
# from tensorflow.keras import utils
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# import tensorflow_datasets as tfds

def import_data():
    # obj = json.loads('./Revised.json').decode('utf-8')
    # bbox = obj['bounding_box']
    # return np.array([bbox['x'], bbox['y'], bbox['height'], bbox['width']], dtype='f')
    with open('./Revised.json') as json_file:
        data = json.load(json_file)
        # print(data[0]['text'])
        # print(data)
        for line in data:
            # for p in line:
            print(line)
        # print(np.array(data))

import_data()

df = pd.read_csv('Revised.csv')

labels = df.pop('truth')
df.pop('index')

dataset = tf.data.Dataset.from_tensor_slices((df.values, labels.values))

for feat, label in dataset.take(5):
  print ('Features: {}, Labels: {}'.format(feat, label))

# # using sklearn's train_test_split function below (test_size=0.25 means test set will be 1/4 size portion of training set)
# x_train, x_test, y_train, y_test = train_test_split(df['text'], df['truth'], test_size=0.25, shuffle=True)

# # datatypes of all four of these sets are a pandas "Series" which act very similar to Python lists, but I think the indices are inconsistent

# # seeing what our sets would print (x sets should print text, while y sets should print labels aka truth)
# print(x_train, y_train)
# print("\n" + "----------------------------------------" + "\n")
# print(x_test, y_test)