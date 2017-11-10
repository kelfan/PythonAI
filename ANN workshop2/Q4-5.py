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
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
"""
validation_split 输入float浮点数,用来指定训练集的一定比例数据作为验证集
histroy 保存结果 
print(history.history.keys) 输出
    dict_keys(['val_acc', 'val_loss', 'loss', 'acc'])
"""
history = model.fit(X, Y, epochs=15, batch_size=10,validation_split=0.8)
print(history.history.keys())
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))