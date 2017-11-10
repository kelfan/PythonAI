# Load LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

"""
加载文件并转为小写
"""
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

"""
set 把文本变成单个字符
list 再把字符变成数组
sortted 按字符顺序排序 
char_to_int 变成(字符，数字)
int_to_char 变成(数字，字符)
	Also, when preparing the mapping of unique characters to integers, we must also create a reverse mapping that we can use to convert the integers back to characters so that we can understand the predictions.
"""
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

"""
Total Characters:  147674
Total Vocab:  47
"""
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

"""
前n-1个字符作为输入 [i:n] 
第n个字符作为输出 [n]
CHAPT -> E
HAPTE -> R
"""
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps固定长度100, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize 变成0-1之间的小数
X = X / float(n_vocab)

# one hot encode the output variable 变成47位0带一个1的稀疏矩阵
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
# X 是 [samples, time steps固定长度100, features]
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
# y.shape (144312, 47) 47代表47位的稀疏矩阵
model.add(Dense(y.shape[1], activation='softmax'))

"""
加载保存最好的（最少loss）weights
"""
# load the network weights
filename = "weights-improvement-10-2.2272.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

"""
print输出 
	Seed:
	" the court and got behind him, and
	very soon found an opportunity of taking it away. she did it so qu "	
"""
# pick a random seed
# 0-144311【总共的输入X的类型个数】 之间的随机小数
start = numpy.random.randint(0, len(dataX)-1)
# 抽取其中一种pattern 输出的是一个巨大的数组，最底层是数字代表的字符串
pattern = dataX[start]
print ("Seed:")
# 把随机抽查的pattern转换为字符再放到一起输出
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

"""
x 
	没reshape之前
		[[[ 0.65957447]
		  [ 0.44680851]
		  [ 0.0212766 ]
		  ..., 
		  [ 0.44680851]
		  [ 0.0212766 ]
		  [ 0.76595745]]]
	reshape 后 
		 [[[21]
		  [ 1]
		  [36]
		  ..., 
		  [ 1]
		  [36]
		  [31]]]
	x/float(n_vocab)
		[[[ 0.44680851]
		  [ 0.0212766 ]
		  [ 0.76595745]
		  ..., 
		  [ 0.0212766 ]
		  [ 0.76595745]
		  [ 0.65957447]]]
reshape 
	np.reshape(a, (2, 3)) # C-like index ordering
	array([[0, 1, 2],
	       [3, 4, 5]])
numpy.argmax() 返回最大的数字，或者把所有的变成组大最大的
	Returns the indices of the maximum values along an axis.
		>>> a = np.arange(6).reshape(2,3)
		>>> a
		array([[0, 1, 2],
		       [3, 4, 5]])
		>>> np.argmax(a)
		5
		>>> np.argmax(a, axis=0)
		array([1, 1, 1])
		>>> np.argmax(a, axis=1)
		array([2, 2])
	prediction
		输出一组概率 
		[[  1.52521487e-02   1.35913312e-01   2.23674718e-03 ...,   6.30083634e-03 1.20717261e-04   5.34149993e-04]]
	index 
		取出最大概率的index
		1
"""
# generate characters
# 运行1000次预测，就是1000个字符
for i in range(1000):
	# 把数组变成1行（放入测定的字符串长度）列，1高
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	# 把数组变成0-1之间的数字
	x = x / float(n_vocab)
	# 输出一组概率，放入代表字符的0-1之间的数字数组
	prediction = model.predict(x, verbose=0)
	# 输出最大概率的index
	index = numpy.argmax(prediction)
	# 把数字转换为字符
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	# 输出预测的结果
	sys.stdout.write(result)
	# 把新预测的字符放进去
	pattern.append(index)
	# 截取需要的长度，相当于放弃最开始的字符 [1到最后一个字符]
	pattern = pattern[1:len(pattern)]
print ("\nDone.")
