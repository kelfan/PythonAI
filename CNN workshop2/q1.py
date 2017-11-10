# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pandas as pd 
dataset = pd.read_csv('international-airline-passengers.csv',delimiter=',', header=None)
# load International airline passengers dataset
#dataset = np.loadtxt("international-airline-passengers.csv", delimiter=",")
print(dataset)