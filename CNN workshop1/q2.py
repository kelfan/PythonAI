
# coding: utf-8

# In[1]:


# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# In[2]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# In[3]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[4]:


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[5]:


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train_bk=X_train
X_test_bk=X_test
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


# In[6]:


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255


# In[7]:


# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# In[8]:


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[10]:


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# In[ ]:
"""
numpy.argmax 是返回沿轴axis最大值的索引。
"""
from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
#print(predictions)
"""
[[  5.50456662e-07   7.14333339e-08   3.51249255e-05 ...,   9.99291539e-01
    1.75686603e-06   4.94393644e-05]
 [  2.52770263e-07   2.83367379e-04   9.99681234e-01 ...,   2.14768245e-10
    4.00991303e-06   3.76711370e-11]
 [  1.56664919e-05   9.90905762e-01   9.60301026e-04 ...,   4.98042488e-03
    2.07881583e-03   1.98437956e-05]
 ..., 
 [  2.54542765e-09   4.35130287e-09   1.50691193e-08 ...,   1.75771769e-04
    1.40325155e-05   1.18702752e-04]
 [  1.44750925e-06   1.69422734e-07   6.25795522e-08 ...,   3.21043512e-06
    2.28942212e-04   2.42131879e-08]
 [  1.23423467e-06   3.39720541e-09   1.58347357e-05 ...,   7.91405663e-10
    1.72496606e-09   4.71684025e-09]]
"""
testArgmaxes = []
for i in range(len(X_test)):
    testArgmax = numpy.argmax(predictions[i])
    testArgmaxes.append(testArgmax)
#print(testArgmaxes)
"""
[7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5, 4, 0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 1, 2, 4, 4, 6, 3, 5, 5...
"""
# 取出prediction和正确的答案作对比
prounded  = [round(X_test[0]) for X_test in predictions]
yrounded = [round(y_test[0]) for y_test in y_test]
#print('prounded')
#print(prounded)
#print('yrounded')
#print(yrounded)
"""
prouned 
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
yrouned
[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
"""
#accuracy = accuracy_score(y_test, rounded, sample_weight=None)
aa= accuracy_score(prounded, yrounded)
print("predictions accuracy: %.2f%%" % (aa*100))
"""
predictions accuracy: 99.74%
"""
for i in range(len(prounded)):
    if prounded[i]!==yrounded[i]:
        print(prounded[i])
        print(testArgmaxes[i])
        plt.subplot(221)
        plt.imshow(X_test_bk[i],cmap=plt.get_cmap('gray'))
        # plt.imshow(X_test[i],cmap=plt.get_cmap('gray'))
        # TypeError: Invalid dimensions for image data
        plt.show()


