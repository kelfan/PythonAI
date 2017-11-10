# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:53:56 2017

@author: chaofanz
"""
    
# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

    

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        print(i)
        print(len(network)-1)
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                print(network[i+1])
                for neuron in network[i + 1]:
                    print('delta')
                    print(neuron['delta'])
                    print(neuron['weights'][j])
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
                print('hidden layer')
                print(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
                print('output-layer')
                print(expected[j] - neuron['output'])
        print('errors')
        print(errors)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
            print('last')
            print(neuron['delta'])
            print(errors[j])
 
# test backpropagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
    print(layer)