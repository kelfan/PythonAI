# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:05:09 2017

@author: chaofanz

This tutorial is broken down into 6 parts:
 
1 Initialize Network: 用随机数产生可以用于训练的network
2 Forward Propagate: 得到预测数据output,例如1或-1
3 Back Propagate Error: 得到变动的数值 delta
4 Train Network: 更新权重weights
5 Predict.
6 Seeds Dataset Case Study.
"""


# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

#==============================================================================
# 2. Forward Propagate
# 把向量的点积生成的0-1之间的小数作为output输出
# We can break forward propagation down into three parts:
#
# 2.1 Neuron Activation.
# 2.2 Neuron Transfer.
# 2.3 Forward Propagation.
#==============================================================================

"""
2.1 Neuron Activation
要点： 算出向量的点积 w1*x1+w2*x2+....
公式： activation = sum(weight_i * input_i) + bias
解释：
    len(weights[i]-1)是因为最后一个是bias
    activation += weights[i]*inputs[i] 等于 权重w和输入数据x两个向量的点积
"""
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

"""
2.2 Neuron Transfer
要点： 产出在S shape上的0-1之间的一个小数，相当于是激活函数，只不过输出1和-1换成了0-1之间的数字
公式： output = 1 / (1 + e^(-activation))
解释：
    可以使用不同的方式 如 
        sigmoid activation function/ also called the logistic function
        the tanh (hyperbolic tangent) function
        the rectifier transfer function
     e is the base of the natural logarithms (Euler’s number).
"""
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

"""
2.3 Forward Propagation
要点：训练network+输入数据row => output-layer的多个数值 
解释： 
    rows 或 inputs 如 [1,0,None] 相当于 x 输入数据向量 
    循环layer时会通过 new_inputs = [] 把前面的数据都覆盖的，所以最后输出只有 output-layer
    activate() 求weight和input数据的乘积
    transfer() 根据乘积求结果数值
"""
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

#==============================================================================
# 3. Back Propagate Error
# 根据预期与实际算出变动值delta输出
# Error is calculated between the expected outputs and the outputs forward propagated from the network. 
# This part is broken down into two sections.
# 3.1 Transfer Derivative.
# 3.2 Error Backpropagation.
#==============================================================================

"""
3.1. Transfer Derivative
要点:相当于学习率
sigmoid transfer function: derivative = output * (1.0 - output)
"""
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

"""
3.2. Error Backpropagation
要点: 数据+预期结果 => 变动数值delta == 相当于 η*(y-y')
	输入: 
		network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
				[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
		expected = [0, 1]
	过程: 
		先算output-layer的errors -> 算delta放入数组
		再算hidden-layer的errors[下一层的第一个weight乘以delta] -> 算delta放入数组
		各自算出正确数值与算出数值的error,再作为delta放入数组
	输出:
		[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': -0.0005348048046610517}]
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': -0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': 0.0771723774346327}]
公式:
	output-layer的error公式 -> 用来更新权重
		error = (expected - output) * transfer_derivative(output)
		相当于 (正确结果 - 算出来的结果)*学习率
	hidden-layer的error公式 -> 用来更新权重
		error = (weight_k * error_j) * transfer_derivative(output)
		相当于 (第k个neuron的weight * 在output-layer的第j个neuron的error-signal)*学习率
解释:
	数据参考:
		network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
				[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]},
	for i in reversed() 	取出不同层layer的数据,并且按反顺序/逆序循环
	if i != len(network)-1 	判断是否在output-layer层,true为是; len(network)-1 是因为index是从0开始算的
	for j in range() 		进入output或weight层
	network[i+1] 			是所在层的下一层,如hidden-layer下一层是output-layer
	for neuron in network[i+1] 	取出weight层
	errors.append()			预期结果 - 数组的'output'结果
	neuron['delta'] = ...	把算出的变动值放入数组
"""
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		# hidden-layer的处理方式
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		# output-layer的处理方式
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


#==============================================================================
# 4. Train Network
# 根据训练network,训练数据,学习率,训练次数,输出数=> 权重更新weights
# using stochastic gradient descent.
# This part is broken down into two sections:
# 4.1 Update Weights.
# 4.2 Train Network.
#==============================================================================

"""
4.1. Update Weights
要点:训练network+训练数据+学习率=> 权重weight更新
公式:
	weight = weight + learning_rate * error * input
输入: 
	l_rate: 0.5
	network:
		[{'weights': [-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 'output': 0.029980305604426185, 'delta': -0.0059546604162323625}, {'weights': [0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 'output': 0.9456229000211323, 'delta': 0.0026279652850863837}]
	row: 
		[[2.7810836,2.550537003,0],
		[1.465489372,2.362125076,0],
		[3.396561688,4.400293529,0],
		[1.38807019,1.850220317,0],
		[3.06407232,3.005305973,0],
		[7.627531214,2.759262235,1],
		[5.332441248,2.088626775,1],
		[6.922596716,1.77106367,1],
		[8.675418651,-0.242068655,1],
		[7.673756466,3.508563011,1]]
输出:
	没有输出,而是副作用改变 network 的weights
	[{'weights': [-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 'output': 0.029980305604426185, 'delta': -0.0059546604162323625}, {'weights': [0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 'output': 0.9456229000211323, 'delta': 0.0026279652850863837}]
解释:
	learning_rate 自定义
	row[:-1] 去掉最后一列 [7.673756466, 3.508563011, 1] => [7.673756466, 3.508563011]
	[neuron['output'] for neuron in network[i - 1]] 取出network上一层layer的neuron里的output
	for neuron in network[i]: 循环network里每一层layer的neuron
	neuron['weights'][j] += 是 更新每个输入对应的权重w
	neuron['weights'][-1] += l_rate * neuron['delta'] 因为input里只有两个数值->前面的循环没有更新最后的weight->需要单独更新
"""
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

"""
4.2. Train Network
要点:训练network,训练数据,学习率,训练次数,输出数=> 为update准备 outputs 和 delta
输入:
	network:
		[[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}, {'weights': [0.2550690257394217, 0.49543508709194095, 0.4494910647887381]}], [{'weights': [0.651592972722763, 0.7887233511355132, 0.0938595867742349]}, {'weights': [0.02834747652200631, 0.8357651039198697, 0.43276706790505337]}]]
	train:
		[[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1], [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]
	l_rate: 0.5
	n_epoch: 20
	n_outputs: 2
输出:
	>epoch=0, lrate=0.500, error=6.350 ...
说明:
	epoch 是训练的次数
	train 是训练数据X1,X2 
	outputs = ... 得到算出来的数据
	expected = ... 得出一个空的矩阵 
		[0, 0]
		[0, 0]
		[0, 0]
		[0, 0]
		[0, 0]
		[0, 0]
		[0, 0]
		[0, 0]
		[0, 0]
		[0, 0]
	expected[row[-1]]=1 不是当前的另一个为1,得到下面矩阵
		[1, 0]
		[1, 0]
		[1, 0]
		[1, 0]
		[1, 0]
		[0, 1]
		[0, 1]
		[0, 1]
		[0, 1]
		[0, 1]
	backward_propagate_error() 为每层weight添加delta 
		[{'weights': [-1.4688375095432327, 1.850887325439514, 1.0858178629550297], 'output': 0.029980305604426185, 'delta': -0.0059546604162323625}, {'weights': [0.37711098142462157, -0.0625909894552989, 0.2765123702642716], 'output': 0.9456229000211323, 'delta': 0.0026279652850863837}]
	update_weights() 更新每个X对应的Weights
		[{'weights': [2.515394649397849, -0.3391927502445985, -0.9671565426390275], 'output': 0.23648794202357587, 'delta': -0.04270059278364587}, {'weights': [-2.5584149848484263, 1.0036422106209202, 0.42383086467582715], 'output': 0.7790535202438367, 'delta': 0.03803132596437354}]
"""

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

"""
可能出错的版本
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
"""

#==============================================================================
# 1. Initialize a network 
# 输入： 
#       input-layer数量，hidden-layer数量，output-layer数量
#       The input layer is really just a row from our training dataset. 
# 过程： 
#       hidden-layer 拥有 n_hidden neurons 且每个neuron有 n_inputs+1 个 weights （additional one for the bias）
#       output-layer 有 n_hidden neurons 且每个neuron有 n_hidden+1 个 weights
# 输出： 随机数字生产的可以用来训练的network
# 例如： 
#       initialize_network(2, 1, 2) 输入2个，有1层hidden-layer和2层output-layer
#       [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}]
#       [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
#==============================================================================
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

#==============================================================================
# 5. Predict
# 要点: 训练network+数据 => 预测结果
# 过程:
# 	先算出output-layer的分类的数值/概率,再取出最大的数值,根据最大的数值得出index,就是输入的数值X对应的分类index
# 解释:
# 	outputs: [0.428716603655823, 0.8618396836408665]
#	max(outputs): 0.8618396836408665
#	outputs.index(max(outputs)):1
#==============================================================================
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

#==============================================================================
# 6. Test Backprop on Seeds dataset
# 输出: 
# 	Scores: [95.23809523809523, 97.61904761904762, 95.23809523809523, 92.85714285714286, 95.23809523809523]
# 	Mean Accuracy: 95.238%
#==============================================================================
seed(1)
# load and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

"""
Tune Algorithm Parameters. 
n_folds = 5
l_rate = 0.5
n_epoch = 50
n_hidden = 1
这个为基本 
改变n_hidden 2层为最合适
	1 = 79.048%
	2 = 94.286%
	3 = 91.905%
	4 = 92.857%
	5 = 93.810%
改变 n_epoch 越大结果越好,但速度越慢,50效率最好
	5 = 48.571%
	10 = 63.810%
	50 = 79.048%
	100 = 80.952%
	500 = 84.286%
改变 l_rate 1和2比较适合,小于则过度拟合,大于则拟合不够
	0.01 = 34.286%
	0.1 = 77.619%
	0.5 = 79.048%
	1 = 81.429%
	2 = 81.905%
	3 = 71.429%
	5 = 65.714%
	10 = 69.524%
改变 n_folds 把数据分成多少分交叉验证,20 效率最好
	1 = list index out of range
	2 = 80.476%
	3 = 79.048%
	5 = 79.048%
	8 = 81.250%
	10 = 82.857%
	15 = 81.905%
	20 = 84.000%
	25 = 83.500%
	30 =  82.381%
	100 = 83.000%
"""