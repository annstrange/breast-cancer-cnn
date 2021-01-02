
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GroupKFold
from itertools import compress
from random import random, shuffle
from operator import itemgetter

import os
import subprocess
import argparse

def test_integrities(y, group, filenames, attribs):
    '''
    I want to know that after some manipulation, the X, y, groups, and filenames have consistency
    '''
    # shapes
    print('len and y: {} {}'.format(y.shape, y))

    # filenames, B v M counts relative to y
    check_filename_v_y = []
    # for each filename, get the B or M prediction
    for i, filename in enumerate(filenames):
        # print ('{} {}'. format(filename, attribs[filename]['tumor_class']))
        if (attribs[filename]['tumor_class'] == "M"):
            diagnosis = 1
        else:
            diagnosis = 0
        check_filename_v_y.append([y[i], diagnosis])
        #if (i < 5 or (len(group) - i) < 5):
            # print (filename, group[i], y[i], diagnosis)

    # are y and diagnosis always the same? sum a row-wise compare
    arr = np.array(check_filename_v_y)
    #print(arr.shape)
    num_diffs = sum(abs(arr[:,0] - arr[:,1]))
    #print('differences array: {} / {}'.format(num_diffs, arr[:10]))

    return num_diffs
    