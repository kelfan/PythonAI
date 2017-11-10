# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:25:13 2017

@author: chaofanz
"""

# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
#print(dataset.shape[0])
ln = dataset.shape[0]
l1 = round(ln*0.8)
#print(dataset[:l2,0:1])

trainSetX = dataset[:l1,0:8]
trainSetY = dataset[:l1,8]
testSetX = dataset[l1:,0:8]
testSetY = dataset[l1:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
"""
tuple (x_val, y_val) or tuple (x_val, y_val, val_sample_weights) to be used as held-out validation data. Will override validation_split.
validation_data (with your manually split data) arguments.
"""
# Fit the model
model.fit(trainSetX, trainSetY, epochs=5, batch_size=10,validation_data=(testSetX,testSetY))
# evaluate the model
scores = model.evaluate(testSetX, testSetY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))