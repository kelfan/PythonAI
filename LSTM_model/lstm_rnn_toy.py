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


# compute sigmoid nonlinearity
# 求出在sigmoid曲线中的x位置的y数
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative 导数
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
# generate binary table
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
# int + binary array pair
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1 # 16位的weights
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1 # 16位的weights
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1 # 16 * 16 weights

synapse_0_update = np.zeros_like(synapse_0) # 16 位 0
synapse_1_update = np.zeros_like(synapse_1) # 16 位 0
synapse_h_update = np.zeros_like(synapse_h) # 16 * 16 位 0

# training logic # 训练的次数
for j in range(10000):

    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number / 2)  # int version
    a = int2binary[a_int]  # binary encoding like a: [0, 0, 0, 0, 1, 0, 0, 1]

    b_int = np.random.randint(largest_number / 2)  # int version
    b = int2binary[b_int]  # binary encoding like b: [0, 0, 1, 1, 1, 1, 0, 0]

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]

    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)  # like [0, 0, 0, 0, 0, 0, 0, 0]

    overallError = 0

    layer_2_deltas = list()
    layer_1_values = list()  # restore hidden layer for next timestep
    layer_1_values.append(np.zeros(hidden_dim))

    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        # generate input and output
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])  # like [[1, 0]] a, b数
        y = np.array([[c[binary_dim - position - 1]]]).T  # [[1]] a+b的c数

        # hidden layer (input ~+ prev_hidden) # dot 是点积
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h)) # layer_1_values[-1] 是最后一次的layer_1的weights; layer_1_values的每一行是16位weights

        # output layer (new binary representation) 求出预测数
        layer_2 = sigmoid(np.dot(layer_1, synapse_1)) # layer_2 是预测是数字

        # did we miss?... if so by how much?
        layer_2_error = y - layer_2 # 求出预测和实际的差
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2)) # 每一个weight的修正数 8位weights修正数
        overallError += np.abs(layer_2_error[0]) # 用来输出模型的错误指数

        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    # 根据结果更新weights
    for position in range(binary_dim):  # binary dim 是二进制的位数
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-position - 1]
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

    # print out progress
    if (j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
