'''	
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		Highway Network implementation of the paper
		<< Highway Network >>
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

import torch
import torch.nn as nn

class Highway(nn.Module):
	"""
		overview:
			Highway Network Layer
		params:
			input_size: here we assume
						input and hidden has
						same dimension
	"""
	def __init__(self, input_size):
		super(Highway, self).__init__()
		self.input_size = input_size
		self.trans_linear = nn.Linear(input_size, input_size)
		self.trans_gate = nn.Sigmoid()

	def forward(self, x, h):
		'''
			overview:
				forward method for highway network
			params:
				x: input [#batch, #len, #input_size]
				h: non-linear output [#batch, #len, #input_size]
			return:
				[#batch, #len, #input_size]

		'''
		t = self.trans_gate(self.trans_linear(x))
		return t * h + (1 - t) * x