# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy 
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from datetime import datetime


# fix random seed for reproducibility
numpy.random.seed(7)
# load International airline passengers dataset
dataset = pandas.read_csv('international-airline-passengers.csv',delimiter=',', header=None)
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0]
Y = dataset.iloc[:,1]

# http://jingyan.baidu.com/article/e75aca855a0103142edac63c.html
X = X.str.extract('-(\d)+')
Y = Y/622
print(X)
print(Y)

