# Copyright (c) 2018 Aria-K-Alethia@github.com
# Description:
#   implementation of layer normalization
#   paper: https://arxiv.org/abs/1607.06450v1

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
	"""
	"""
	def __init__(self, size, eps = 1e-6):
		super(LayerNorm, self).__init__()
		self.size = size
		self.eps = eps
		self.a = nn.Parameter(torch.ones(size))
		self.b = nn.Parameter(torch.zeros(size))
	def forward(self, x):
		mean = x.mean(-1, keepdim = True)
		std = x.std(-1, keepdim = True)
		return self.a * (x - mean) /  (std + self.eps) + self.b
