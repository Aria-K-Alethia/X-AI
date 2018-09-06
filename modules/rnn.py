# Copyright (c) 2018 Aria-K-Alethia@github.com
# Description:
#   GRU & LSTM layer implemented by pytorch
#	These layer can be easily modified and experimented

import torch
import torch.nn as nn


class MultilayerLSTM(nn.Module):
	"""
		overview:
			single direction, multi-layer LSTM
		params:
			see LSTM param
	"""
	def __init__(self, input_size, hidden_size, num_layers = 1,
				 bias = True, dropout = 0):
		super(MultilayerLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.bias = bias
		self.dropout = nn.Dropout(dropout)
		self.cells = nn.ModuleList()
		# create cells
		for _ in range(num_layers):
			self.cells.append(nn.LSTMCell(input_size, hidden_size, bias = bias))
			input_size = hidden_size

	def forward(self, input, hidden_state):
		'''
			overview:
				the input method for Multilayer-LSTM
		'''
		h0, c0 = hidden_state
		h1, c1 = [], []
		for i, cell in enumerate(self.cells):
			h, c = cell(input, (h0[i], c0[i]))
			input = h
			if i != self.num_layers - 1:
				input = self.dropout(input)
			h1.append(h)
			c1.append(c)
		h1 = torch.stack(h1)
		c1 = torch.stack(c1)
		return input, (h1, c1)




class LSTM(nn.Module):
	"""
		overview:
			a customized implementation of LSTM
		params:
			see the parameter of torch.nn.LSTM
	"""
	def __init__(self, input_size, hidden_size, num_layers = 1, bias = True,
                 batch_first = False, dropout = 0, bidirectional = False):
		super(LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.bias = bias
		self.batch_first = batch_first
		self.bidirectional = bidirectional
		self.num_direction = 2 if bidirectional else 1
		if not self.bidirectional:
			self.cell = MultilayerLSTM(input_size, hidden_size, num_layers, bias, 
							dropout)
		else:
			self.cell = nn.ModuleList(
					MultilayerLSTM(input_size, hidden_size, num_layers, bias, dropout)
					for _ in range(2)
				)

	def split_bidirectional_state(self, state):
		'''
			overview:
				split hidden state into forward & backward state
			params:
				state: [#numlayers*#num_direction, #bacth, #hidden]
			return:
				forward_state, backward_state
				both have the same shape [#numlayers, #batch, #hidden]
		'''
		temp = state[0].shape[0]
		forward_state = tuple([torch.stack([stat[i] for i in range(0, temp, 2)]) for stat in state])
		backward_state = tuple([torch.stack([stat[i] for i in range(1, temp, 2)]) for stat in state])
		return forward_state, backward_state

	def merge_bidirectional_state(self, forward_state, backward_state):
		'''
			overview:
				merge the bidirectionl states
				this is in fact the reverse operation of split_bidirectional_state
		'''
		assert all(state1.shape[0] == state2.shape[0] for state1, state2 in zip(forward_state, backward_state)),\
				'must have same layer for two states'
		layers = forward_state[0].shape[0]
		ret = []
		for i in range(2):
			buf = []
			for j in range(layers):
				buf.append(forward_state[i][j])
				buf.append(backward_state[i][j])
			ret.append(torch.stack(buf))
		return tuple(ret)

	def run_single_direction(self, input, hidden_state):
		'''
			overview:
				forward method for single direction LSTM
		'''
		assert not self.bidirectional
		memorys = []
		for i, emb in enumerate(input.split(1)):
			emb = emb.squeeze()
			memory, hidden_state = self.cell(emb, hidden_state)
			memorys.append(memory)
		memorys = torch.stack(memorys)
		return memorys, hidden_state


	def run_double_direction(self, input, hidden_state):
		'''
			overview:
				forward method for bidirectional LSTM
		'''
		assert self.bidirectional
		memorys = [[],[]]
		embs = input.split(1)
		embs = [emb.squeeze() for emb in embs]
		# get the state of each direction
		forward_state, backward_state = self.split_bidirectional_state(hidden_state)
		for emb in embs:
			memory, forward_state = self.cell[0](emb, forward_state)
			memorys[0].append(memory)
		# backward
		for emb in reversed(embs):
			memory, backward_state = self.cell[1](emb, backward_state)
			memorys[1].append(memory)
		# merge the memory
		memorys = [torch.stack(item) for item in memorys]
		memorys = torch.cat(memorys, -1)
		# merge the state
		hidden_state = self.merge_bidirectional_state(forward_state, backward_state)
		return memorys, hidden_state

	def forward(self, input, hidden_state = None):
		'''
			overview:
				forward method of LSTM
			params:
				input: [#len, #batch, #input_size]
				hidden_state: (h0, c0), both of them should have shape of
						    [#num_layers*num_directions, #batch, #hidden_size],
						    if not provided, set them to zero 
			return:
				memorys: [#len, #batch, #numdirections * hidden_size]
				(hn, cn): final state of the last layer, both of them have the shape
						  [#num_layers*num_directions, #batch, #hidden_size]
		'''
		if self.batch_first:
			input = input.transpose(1, 0)
		if hidden_state is None:
			h = torch.zeros(self.num_layers*self.num_direction, input.shape[1], self.hidden_size)
			c = h.clone()
			hidden_state = (h, c)
		if self.bidirectional:
			return self.run_double_direction(input, hidden_state)
		else:
			return self.run_single_direction(input, hidden_state)
