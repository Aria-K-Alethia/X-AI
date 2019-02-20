'''
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		customized embedding class	
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''
import torch
import torch.nn as nn

class Embedding(nn.Module):
	"""
        overview:
			customized embedding class
			could be altered to support
			linguistic feature
		params:
			fix_embedding: if fix, gradient will
						   not be computed for embedding
	"""
	def __init__(self, vocab_size, embedding_dim,
				padding_idx, fix_embedding):
		super(Embedding, self).__init__()
		self._embedding_dim = embedding_dim
		self._vocab_size = vocab_size
		self._padding_idx = padding_idx
		self.embedding = nn.Embedding(vocab_size,
						embedding_dim, padding_idx = padding_idx)
		if fix_embedding:
			self.embedding.weight.requires_grad = False
	def forward(self, source):
		'''
			overview:
				forward method of embedding
				compute the embedding of input
			params:
				source: should be torch.LongTensor
			return:
				[-1, embedding_dim]
		'''
		return self.embedding(source)

	def load_pretrained_vectors(self, emb_file):
		'''
			overview:
				load pretrained vectors for this embedding
			params:
				emb_file: *.pt, save the embedding matrix
		'''
		pretrained = torch.load(emb_file)
		pre_vocab_size, pre_vec_size = pretrained.shape()
		assert pre_vocab_size == self.vocab_size,\
				"pretrained embedding's vocab \
				size %d should equal to embedding's\
				vocab size %d" % (pre_vocab_size, self.vocab_size)
		if self.embedding_dim > pre_vec_size:
			self.weight.data[:, :pre_vec_size] = pretrained
		elif self.embedding_dim < pre_vec_size:
			self.weight.data.copy_(pretrained[:, :self.embedding_dim])
		else:
			self.weight.data.copy_(pretrained)
	@property
	def embedding_dim(self):
		return self._embedding_dim
	@property
	def vocab_size(self):
		return self._vocab_size
	@property
	def padding_idx(self):
		return self._padding_idx
	@property
	def weight(self):
		return self.embedding.weight
	
	
	

