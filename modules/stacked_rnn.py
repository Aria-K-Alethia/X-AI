'''	
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		stacked rnn for input feeding
		Luong et.al 2015, almost a copy from OpenNMT
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

import torch
import torch.nn as nn

class StackedLSTM(nn.Module):
	"""
        overview:
			stacked LSTM for input feeding
		params:
			see torch.nn.LSTMCell
	"""
	def __init__(self, num_layers, input_size,
				rnn_size, dropout, batch_first = True):
		super(StackedLSTM, self).__init__()
		self.num_layers = num_layers
		self.input_size = input_size
		self.rnn_size = rnn_size
		self.dropout = dropout
		self.cells = nn.ModuleList()
		for _ in range(num_layers):
			self.cells.append(
					nn.LSTMCell(input_size, rnn_size)
				)
			input_size = rnn_size
	def forward(self, input_feed, hidden):
		'''
			overview:
				forward method for stacked LSTM
			params:
				input_feed: input data
				hidden:  hidden state
						 [#layer, #batch, #hidden]
			return:
				input_feed, (h, c)
		'''
		h_0, c_0 = hidden
		h_1, c_1 = [], []
		for i, cell in enumerate(self.cells):
			h_1_i, c_1_i = cell(input_feed, (h_0[i], c_0[i]))
			input_feed = h_1_i
			if i + 1 != self.num_layers:
				input_feed = self.dropout(input_feed)
				h_1.append(h_1_i)
				c_1.append(c_1_i)
		h_1 = torch.stack(h_1)
		c_1 = torch.stack(c_1)
		return input_feed, (h_1, c_1)

		
