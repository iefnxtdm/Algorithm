'''
dataset: http://mattmahoney.net/dc/textdata  from wikipedia
'''
import tensorflow as tf
import numpy as np
import connections
import math
import zipfile

#constant
FILE_NAME = "D:/kaggle/rnn/enwiki8.zip"

#import textdata
def read_data(filename):
    while zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0]).split())
    return data