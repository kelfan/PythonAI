"""
http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#  LSTM network on the IMDB dataset
Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras
	- How to develop an LSTM model for a sequence classification problem.
	- How to reduce overfitting in your LSTM models through the use of dropout.
	- How to combine LSTM models with Convolutional Neural Networks that excel at learning spatial relationships.
"""

"""
Problem Description: IMDB movie review sentiment classification problem
	结果是positive还是negative的评论
	http://ai.stanford.edu/~amaas/data/sentiment/
	The Large Movie Review Dataset (often referred to as the IMDB dataset) contains 25,000 highly-polar movie reviews (good or bad) for training and the same amount again for testing. 
"""

"""
Word Embedding
	 a technique where words are encoded as real-valued vectors in a high dimensional space, where the similarity between words in terms of meaning translates to closeness in the vector space.
	 Keras provides a convenient way to convert positive integer representations of words into a word embedding by an Embedding layer.
	 We will map each word onto a 32 length real valued vector. We will also limit the total number of words that we are interested in modeling to the 5000 most frequent words, and zero out the rest. Finally, the sequence length (number of words) in each review varies, so we will constrain each review to be 500 words, truncating long reviews and pad the shorter reviews with zero values.
"""

# 导入相关库import numpy
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

# 导入前5000,其他为零
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# 统一数据的长度
# 例如把0变成000000
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(X_test)


"""
# 构建模型 create the model
"""
#  32 length vectors to represent each word
embedding_vecor_length = 32
model = Sequential()
# Embedded layer that uses 32 length vectors to represent each word
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# LSTM 输出100
model.add(LSTM(100))
# 输出1个 
# single neuron output => sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem
model.add(Dense(1, activation='sigmoid'))
# binary classification problem => log loss is used as the loss function (binary_crossentropy in Keras).
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# only 2 epochs because it quickly overfits the problem
# 64 reviews is used to space out weight updates.
model.fit(X_train, y_train, epochs=3, batch_size=64)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save_weights('./weights2')
