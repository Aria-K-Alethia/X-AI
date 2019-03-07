'''	
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		Multi-Layer perceptron
		Support customized activation function
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	"""
		overview:
			Multi-Layer Perceptron
		params:
			input_size: input data size
			hidden_sizes: list, including each layer's size
			activations: list, including each layers's activation
						  support: tanh, relu, 
			dropout: dropout rate
	"""
	_legal_activations = ['tanh', 'sigmoid', 'relu', 'leaky_relu', 'prelu']
	_module_name = ['Tanh', 'Sigmoid', 'ReLU', 'LeakyReLU', 'PReLU']
	_activation_map = {k: getattr(nn, v) for k, v in zip(_legal_activations, _module_name)}
	def __init__(self, input_size, hidden_sizes, activations, dropout = 0):
		super(MLP, self).__init__()
		assert all(item in self.__class__._activation_map for item in activations),\
				"activation function could only be selected from %s" \
				% (self.__class__._legal_activations)
		assert len(hidden_sizes) == len(activations), \
			"length of hidden_sizes and activations must be equal,"\
			+"but get: %d and %d" % (len(hidden_sizes), len(activations))
		self.input_size = input_size
		self.hidden_sizes = hidden_sizes
		self.activations = activations
		self.dropout = nn.Dropout(dropout)
		# construct each layer
		self.layers = nn.Sequential()
		for layer_idx, (hidden_size, activation) \
			in enumerate(zip(hidden_sizes, activations)):
			linear = nn.Linear(input_size, hidden_size)
			act = self.get_activation(activation)
			idx = str(layer_idx)
			self.layers.add_module('linear'+idx, linear)
			self.layers.add_module('act'+idx, act)
			if layer_idx != len(hidden_sizes) - 1:
				self.layers.add_module('dropout'+idx, self.dropout)
			input_size = hidden_size


	def get_activation(self, name):
		return self._activation_map[name]()


	def forward(self, x):
		'''
			overview:
				forward method for MLP
			params:
				x: [*, #input_size]
			return:
				y: [*, #hidden_sizes[-1]]
		'''
		return self.layers(x)
