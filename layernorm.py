# Copyright (c) 2018 Aria-K-Alethia@github.com
# Description:
#   implementation of layer normalization
#   paper: https://arxiv.org/abs/1607.06450v1

class LayerNorm(nn.Module):
	"""
	"""
	def __init__(self, size, eps = 1e-6):
		super(LayerNorm, self).__init__()
		self.size = size
		self.eps = eps
