'''	
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		penalty in beam search
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

import torch
from functools import partial

class Penalty(object):
	"""
		overview:
			penalty in beam search
			support length and coverage
			penalties
		params:
			length_penalty: length penalty category
			coverage_penalty:
						  coverage penalty category
			alpha, beta: parameter, for google penalty
	"""
	def __init__(self, length_penalty,
				coverage_penalty,
				alpha = None, beta = None):
		super(Penalty, self).__init__()
		self.length_penalty_cate = length_penalty
		self.coverage_penalty_cate = coverage_penalty
		self.alpha = alpha
		self.beta = beta
		self.length_penalty = \
					self.build_length_penalty()
		self.coverage_penalty = \
					self.build_coverage_penalty()

	def build_length_penalty(self):
		cate = self.length_penalty_cate
		if cate == 'google':
			return partial(
				self.google_length_penalty,
				alpha = self.alpha)
		elif cate == 'average':
			return self.average_length_penalty
		else:
			return lambda scores, length: scores

	def build_coverage_penalty(self):
		cate = self.coverage_penalty_cate
		if cate == 'google':
			return partial(
				self.google_coverage_penalty,
				beta = self.beta)
		elif cate == 'summary':
			return partial(
				self.summary_coverage_penalty,
				beta = self.beta)
		else:
			return lambda coverage:\
					torch.zeros(coverage.shape[0])

	def google_length_penalty(self, scores,
							lengths, alpha = 0):
		factor = (5 + lengths)**alpha \
				/ (5 + 1)**alpha
		return scores / factor

	def average_length_penalty(self, scores, lengths):
		return scores / lengths

	def google_coverage_penalty(self, coverage,
								beta = 0):
		return beta * -torch.min(coverage, \
			torch.ones(*coverage.shape)\
			.type_as(coverage))\
			.log().sum(1).to(coverage.device)

	def summary_coverage_penalty(self, coverage,
								beta = 0):
		return beta * \
			(torch.max(\
			coverage, coverage.clone().fill_(1)\
			).sum(1) - coverage.shape[1])
