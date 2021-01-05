import collections
import pathlib
import re
import string

# import tensorflow as tf

import numpy as np
import pandas as pd

import json

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# import tensorflow_datasets as tfds

def import_data():
    with open('./Revised.json') as f:
        data = json.load(f)

import_data()