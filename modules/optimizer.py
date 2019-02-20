'''
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		custimized optimizer to adapt various optmization strategies
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Optimizer(object):
	"""
		overview:
			doing optimization, grad clip,
			schedule learning rate
		params:
			optimizer: optimizer of pytorch
			learning_rate: init learning rate
			learning_rate_decay_fn: compute a scaling factor to scale lr
			max_grad_norm: for grad clip
	"""
	def __init__(self, optimizer, learning_rate,
		learning_rate_decay_fn = None, max_grad_norm = None):
		super(Optimizer, self).__init__()
		self._optimizer = optimizer
		self._learning_rate = learning_rate
		self._learning_rate_decay_fn = learning_rate_decay_fn
		self._max_grad_norm = max_grad_norm
		self._train_step = 1
		self._decay_step = 1
		self._last_loss = 10

	def state_dict(self):
		return {
			'learning_rate': self._learning_rate,
			'max_grad_norm': self._max_grad_norm,
			'decay_step': self._decay_step,
			'last_loss': self._last_loss
		}

	def decay_learning_rate(self):
		if self._learning_rate_decay_fn:
			scale = self._learning_rate_decay_fn()
			self._learning_rate *= scale
		self._decay_step += 1 # update this value every time

	def step(self, loss):
		if loss > self._last_loss:
			self.decay_learning_rate()
		self._last_loss = loss
		for group in self._optimizer.param_groups:
			group['lr'] = self._learning_rate
			if self._max_grad_norm is not None and self._max_grad_norm > 0:
				clip_grad_norm_(group['params'], self._max_grad_norm)
		self._optimizer.step()
		self._train_step += 1

	@property
	def train_step(self):
		return self._train_step

	@property
	def decay_step(self):
		return self._decay_step

	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def max_grad_norm(self):
		return self._max_grad_norm
	
	@property
	def last_loss(self):
		return self._last_loss

	@last_loss.setter
	def last_loss(self, value):
		self._last_loss = value
	

