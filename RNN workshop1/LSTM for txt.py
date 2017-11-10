# Small LSTM Network to Generate Text for Alice in Wonderland
"""
统计ASCII编码的txt文本的根据前100个字符预测后1个字符是什么
http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/ 
"""

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

"""
把文本拆分同时转为数字 
set()
	把字符串按单个字符拆开
	print(set('abcdedf'))
	{'b', 'd', 'f', 'a', 'e', 'c'}
list()
	把字符串变成数组
	print(list(set('abcdedf')))
	['b', 'd', 'f', 'a', 'e', 'c']
sorted
	print(sorted(list(set('abcdedf'))))
	['a', 'b', 'c', 'd', 'e', 'f']
"""
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))

"""
enumerate: 把数组变成(数字,内容)格式
	>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
	>>> list(enumerate(seasons))
	[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
	>>> list(enumerate(seasons, start=1))
	[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
dict(): 把(数字,内容)格式变成 (key:value)格式 
	chars = sorted(list(set('abcdefgh')))
	print(list(enumerate(chars)))
		[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h')]
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	print(char_to_int)
		{'f': 5, 'a': 0, 'e': 4, 'b': 1, 'h': 7, 'g': 6, 'c': 2, 'd': 3}
本例中的输出例子 
	['\n', '\r', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', ':', ';', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xbb', '\xbf', '\xef']
"""
char_to_int = dict((c, i) for i, c in enumerate(chars))


# summarize the loaded data
# Total Characters:  147674
# Total Vocab:  47
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers

"""
按100字符切割 
	Each training pattern of the network is comprised of 100 time steps of one character (X) followed by one character output (y). 
例子 
	For example, if the sequence length is 5 (for simplicity) then the first two training patterns would be as follows:
		[chapter] 预测下一个字母
		CHAPT -> E 
		HAPTE -> R
"""
# split the book text up into subsequences with a fixed length of 100 characters
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	# 输入 i 到 i+100-1 个的字符
	seq_in = raw_text[i: i+seq_length]
	# 输出 i+100 个字符
	seq_out = raw_text[ i+seq_length ]
	# 把输入放入dataX
	dataX.append([char_to_int[char] for char in seq_in])
	# 把输出放入dataY
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
# 结果 Total Patterns:  147574
print( "Total Patterns: ", n_patterns)

# LSTM 只接受 [samples, time steps, feature] 格式
# reshape X to be [samples, time steps固定字符串长度, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize 把文字后面的数字变成0-1之间的小数
X = X / float(n_vocab)

"""
把代表结果的y值变成一个稀疏矩阵 
	Each y value is converted into a sparse vector with a length of 47, full of zeros except with a 1 in the column for the letter (integer) that the pattern represents.
	For example, when “n” (integer value 31) is one hot encoded it looks as follows:
		[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
		  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.
		  0.  0.  0.  0.  0.  0.  0.  0.]
"""
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

"""
定义LSTM模型 
LSTM层 
	256 个输出 a single hidden LSTM layer with 256 memory units
X.shape 
	(144312, 100, 1)
	行列高
	[samples, time steps固定字符串长度, features]
dropout 两成神经元
	uses dropout with a probability of 20
输出层 
	softmax 输出0-1之间的数字代表47个字符
	a Dense layer using the softmax activation function to output a probability prediction for each of the 47 characters between 0 and 1.
compile优化运行
	 optimizing the log loss (cross entropy),
	  ADAM optimization algorithm for speed
"""
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
# y.shape (144312, 47)
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

"""
使用checkpoints记录每次的loss，然后使用最低的
	use model checkpointing to record all of the network weights to file each time an improvement in loss is observed at the end of the epoch
	We will use the best set of weights (lowest loss) to instantiate our generative model in the next section.
保存文件名 
	weights-improvement-19-1.9435.hdf5
	可以把除了loss最小的以外的都删除了
"""
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

"""
训练模型 
	训练20回
	每次128个patterns
	callbacks是使用之前记录的weights
"""
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)