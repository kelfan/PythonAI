# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:25:13 2017

@author: chaofanz
"""

# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras import optimizers
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

"""
NO. of layers: 增减层数 如 model.add(Dense(1, activation='softplus'))
No. of hidden nodes: 每层里的节点 Dense(第一数字)
activation functions: 激活函数 activation=relu|sigmoid|hard_sigmoid|softplus
optimisers: 优化函数 optimizer='adam|..'
loss functions: Loss='binary_crossentropy'
batch sizes: batch_size=10
learning rates: sgd 和 compile
"""

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
# create classifier for use in scikit-learn
model = KerasClassifier(build_fn=create_model, nb_epoch=5, batch_size=10)
# evaluate model using 10-fold cross validation in scikit-learn
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, Y, cv=kfold)
print(results)
"""
结果是一串准确率的数组 
[ 0.74025975  0.66233767  0.70129871  0.64935065  0.58441559  0.68831169
  0.57142857  0.62337663  0.60526317  0.55263159]
"""


## Fit the model
#history = model.fit(X, Y, epochs=15, batch_size=10)
#print(history.history.keys())
## evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))