# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:38:32 2017

@author: chaofanz
"""

from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
#            print(activation)
            neuron['output'] = transfer(activation)
#            print(neuron['output'])
            new_inputs.append(neuron['output'])
#            print(new_inputs)
        inputs = new_inputs
    return inputs

# test forward propagation
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)

#==============================================================================
# 解释为什么输出只有两个数字 
# 因为循环layer时用 new_inputs = [] 把之前的覆盖了
# 0.7105668883115941
# [0.7105668883115941]
# 0.6629970129852887
# [0.6629970129852887]
# 0.7253160725279748
# [0.6629970129852887, 0.7253160725279748]
# [0.6629970129852887, 0.7253160725279748]
#==============================================================================
