# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#==============================================================================
# This tutorial is broken down into 6 parts:
# 
# Initialize Network.
# Forward Propagate.
# Back Propagate Error.
# Train Network.
# Predict.
# Seeds Dataset Case Study.
#==============================================================================


#==============================================================================
# Below is a function named initialize_network() that creates a new neural network ready for training. It accepts three parameters, the number of inputs, the number of neurons to have in the hidden layer and the number of outputs.
# You can see that for the hidden layer we create n_hidden neurons and each neuron in the hidden layer has n_inputs + 1 weights, one for each input column in a dataset and an additional one for the bias.
# You can also see that the output layer that connects to the hidden layer has n_outputs neurons, each with n_hidden + 1 weights. This means that each neuron in the output layer connects to (has a weight for) each neuron in the hidden layer.
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
# We can break forward propagation down into three parts:
# 
# Neuron Activation.
# Neuron Transfer.
# Forward Propagation.
#==============================================================================


#==============================================================================
#1. Neuron Activation
# Below is an implementation of this in a function named activate(). You can see that the function assumes that the bias is the last weight in the list of weights. This helps here and later to make the code easier to read.
# activation = sum(weight_i * input_i) + bias
#==============================================================================

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation


#==============================================================================
# 2. Neuron Transfer
# output = 1 / (1 + e^(-activation))
# Below is a function named transfer() that implements the sigmoid equation.
#==============================================================================

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

#==============================================================================
# 3. Forward Propagation
# Below is a function named forward_propagate() that implements the forward propagation for a row of data from our dataset with our neural network.
#==============================================================================
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
