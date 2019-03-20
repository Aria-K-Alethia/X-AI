'''	
	Copyright (c) 2019 Aria-K-Alethia@github.com
	
	Description:
		embedding class supporting multiple features
		and customized merge method	
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''
import torch
import torch.nn as nn
from embedding import Embedding
from mlp import MLP

class MultiFeatureEmbedding(nn.Module):
	"""
		overview:
			for nlp, support multiple feature
			and customized merge methods
		params:
			vocab_sizes: vocab size for each feature
			embedding_dims: embedding dim for each feature
			merge_methods: merge method for each feature
							currently support 'cat', 'sum'
			out_method: support 'none', 'linear', 'mlp'
			out_dim: output dimension
	"""
	def __init__(self, vocab_sizes,
				embedding_dims, merge_methods,
				padding_indices, fix_embedding,
				out_method = 'none',
				out_dim = None):
		super(MultiFeatureEmbedding, self).__init__()
		self._vocab_sizes = vocab_sizes
		self._embedding_dims = embedding_dims
		self._n_feature = len(vocab_sizes)
		self._merge_methods = merge_methods
		self._padding_indices = padding_indices
		self._fix_embedding = fix_embedding
		self.emb_list = nn.ModuleList(Embedding(
									vocab_size, embedding_dim,
									padding_idx, fix_embedding)
									for vocab_size, embedding_dim,
										padding_idx in
									zip(vocab_sizes,
										embedding_dims, padding_indices))
		self._out_method = out_method
		self._emb_out_dim = sum(dim for index, dim in
								enumerate(embedding_dims)
								if merge_methods[index] == 'cat')
		if out_method == 'none':
			self._out_dim = self._emb_out_dim
		elif out_method == 'linear':
			self._out_dim = out_dim
			self.out_module = nn.Linear(self._emb_out_dim, self._out_dim)
		else:
			self._out_dim = out_dim
			self.out_module = MLP(self._emb_out_dim,
								[int(self._emb_out_dim / 2), self._out_dim],
								['prelu', 'prelu'])


	def forward(self, x):
		'''
			overview:
				forward method
			params:
				x: [#batch, #len, #n_feature]
		'''
		# get embedding for each feature
		outs = []
		for emb, f in enumerate(zip(self.emb_list, x.split(1, -1))):
			outs.append(emb(f.squeeze(-1)))
		# merge
		# cat
		cat = [emb for index, emb in enumerate(outs)\
					if self.merge_methods[index] == 'cat']
		out = torch.cat(cat, -1)
		# sum
		sum_ = [emb for index, emb in enumerate(outs)\
					if self.merge_methods[index] == 'sum']
		[out.add_(item) for item in sum_]
		if self.output_method != 'none':
			out = self.out_module(out)
		return out

	@property
	def vocab_sizes(self):
		return self._vocab_sizes
	@property
	def embedding_dims(self):
		return self._embedding_dims
	@property
	def n_feature(self):
		return self._n_feature
	@property
	def merge_methods(self):
		return self._merge_methods
	@property
	def fix_embedding(self):
		return self._fix_embedding

	@property
	def output_method(self):
		return self._out_method
	
	@property
	def embedding_output_dim(self):
		return self._emb_out_dim

	@property
	def output_dim(self):
		return self._out_dim
	
	