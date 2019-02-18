'''	
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		vocabulary for text processing
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

from collections import Counter

class Vocab(object):
	"""
		overview:
        	simple vocab for text processing
        params:
        	specials: special symbols for this vocab, like <unk>
			min_freq: minimal frequency
			max_size: maximal size
		NOTE:
			special symbols are not counted in vocab size
	"""
	def __init__(self, specials,
				min_freq = 5, max_size = 5000):
		super(Vocab, self).__init__()
		self._max_size = max_size
		self._min_freq = min_freq
		self._specials = specials
		self._itos = list()
		self._stoi = dict()
		self._freq = Counter()
		# process special symbols
		self._itos.extend(specials)
		self._stoi.update({s: i for i, s in enumerate(specials)})
		self._size = len(self._stoi)

	def extend(self, tokens):
		'''
			overview:
				add tokens in vocab
			params:
				tokens: iterable
		'''
		self._freq.update(tokens)
		index = len(self._itos)
		for token in tokens:
			if not token in self._stoi \
			   and self._freq[token] >= self._min_freq \
			   and self._size < self._max_size:
			   self._stoi[token] = index
			   self._itos.append(token)
			   index += 1
			   self._size += 1

	@property
	def size(self):
		return self._size
	
	@property
	def max_size(self):
		return self._max_size

	@property
	def min_freq(self):
		return self._min_freq

	@property
	def specials(self):
		return self._specials

	@property
	def itos(self):
		return self._itos

	@property
	def stoi(self):
		return self._stoi

	@property
	def freq(self):
		return self._freq
	



			