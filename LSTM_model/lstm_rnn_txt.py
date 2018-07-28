import copy

import numpy as np

np.random.seed(0)

"""
- 生成 int 对应 二进制的array 的table
- 生成两个随机数, 例如 10 和 13
- a,b 是数字, c是a+b的答案, d是预测的结果
- 循环6位二进制得出weights 修正
    - 每一个layer_1 都是 这次的a,b + 上一次的layer_1
    - 预测是 layer_1 + synapse_1
    - 求出 预测和真实结果差
    - 再求出更新值
- 循环更新weights
"""

# get words
words = "a,abandon,abandoned,ability,able,about,above,abroad,absence,absent,absolute,absolutely,absorb,abuse,abuse,academic,accent,accept,acceptable,access,accident,accidental,accidentally,accommodation,accompany,according to,account,account for,accurate,accurately,accuse,achieve,achievement,acid,acknowledge,a couple,acquire,across,act,action,active,actively,activity,actor,actress,actual,actually,ad,adapt,add,addition,additional,add on,address,add up,add up to,adequate,adequately,adjust,admiration,admire,admit,adopt,adult,advance,advanced,advantage,adventure,advert,advertise,advertisement,advertising,advice,advise,affair,affect,affection,afford,afraid,after,afternoon,afterwards,again,against,age,aged,agency,agent,aggressive,ago,agree,agreement,ahead,aid,aim,air,aircraft,airport,alarm,alarmed"
word_list = words.split(",")

# create mapping of unique chars to integers
chars = sorted(list(set(word_list)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


# compute sigmoid nonlinearity
# 求出在sigmoid曲线中的x位置的y数
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative 导数
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# input variables
alpha = 0.1
input_dim = 1
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1  # 16位的weights
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1  # 16位的weights
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1  # 16 * 16 weights

synapse_0_update = np.zeros_like(synapse_0)  # 16 位 0
synapse_1_update = np.zeros_like(synapse_1)  # 16 位 0
synapse_h_update = np.zeros_like(synapse_h)  # 16 * 16 位 0

# training logic # 训练的次数
for j in range(100):
    print(j)
    layer_2_deltas = list()
    layer_1_values = list()  # restore hidden layer for next timestep
    layer_1_values.append(np.zeros(hidden_dim))

    future_layer_1_delta = np.zeros(hidden_dim)

    for word in chars:
        if len(word) < 2:
            continue
        for i in range(0, len(word) - 1):
            a = ord(word[i]) / 128
            b = ord(word[i + 1]) / 128
            X = np.array([[a]])
            y = np.array([[b]])

            prev_layer_1 = layer_1_values[-1]
            # hidden layer (input ~+ prev_hidden) # dot 是点积
            layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1],
                                                            synapse_h))  # layer_1_values[-1] 是最后一次的layer_1的weights; layer_1_values的每一行是16位weights
            # output layer (new binary representation) 求出预测数
            layer_2 = sigmoid(np.dot(layer_1, synapse_1))  # layer_2 是预测是数字

            # did we miss?... if so by how much?
            layer_2_error = y - layer_2  # 求出预测和实际的差
            layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))  # 每一个weight的修正数 8位weights修正数

            # store hidden layer so we can use it in the next timestep
            layer_1_values.append(copy.deepcopy(layer_1))

            # update layer
            # error at output layer
            layer_2_delta = layer_2_error
            # error at hidden layer
            layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + \
                             layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
            # let's update all our weights so we can try again
            synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            synapse_0_update += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

s = "abroad"
layer_2_deltas = list()
layer_1_values = list()  # restore hidden layer for next timestep
layer_1_values.append(np.zeros(hidden_dim))
for i in range(0, len(s)):
    a = ord(s[i]) / 128
    X = np.array([[a]])
    layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1],
                                                    synapse_h))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    print(str(layer_2 * 128))
    print(chr(np.round(layer_2 * 128)))
    layer_1_values.append(copy.deepcopy(layer_1))
