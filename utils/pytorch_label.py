#!/usr/bin/env python
import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
import csv
import os
from numpy import genfromtxt

def find_classes(dir):
    myDt = {0: 'cat', 1: 'elephant', 2: 'flamingo', 3: 'fox',
            4: 'ostrich', 5: 'panda', 6: 'panther', 7: 'plane',
            8: 'polar_bear', 9: 'tiger', 10: 'wolf', 11: 'dog'}
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {}
    for i in class_to_idx:
        idx_to_class[class_to_idx[i]] = myDt[int(i)]
    return idx_to_class
