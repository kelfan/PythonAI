#==============================================================================
# 2.	Manually split your data, creating a training set consisting of the first 80% of entries and a test set for the remaining 20%
# 3.	Change your model.fit and model.evaluate calls to use the training and test sets respectively
#==============================================================================


# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
"""
把数据拆分为80%和20%
"""
X = dataset[:,0:8]
Y = dataset[:,8]
split =round(0.8*len(X))
X_train = X[:split]
X_test = X[split:]
Y_train = Y[:split]
Y_test =Y[split:]
# create model
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(8, activation='hard_sigmoid'))
model.add(Dense(1, activation='softplus'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
#model.fit(X_train, Y_train, epochs=150, batch_size=10, shuffle= True ,validation_split=0.8)
model.fit(X_train, Y_train, epochs=150, batch_size=10, shuffle= False )

# evaluate the model
history =model.fit(X, Y, epochs=150, batch_size=10, shuffle= True, validation_split=0.8)
print (history.history.keys())
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))