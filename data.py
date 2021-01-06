# import collections
# import pathlib
# import re
# import string

# import tensorflow as tf

import numpy as np
import pandas as pd

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