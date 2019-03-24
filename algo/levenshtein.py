'''	
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		Levenshtein Edit Distance
		Support customized cost
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

class LevenshteinEditDistance(object):
	"""
		overview:
			Lev edit distance
	"""
	def __init__(self):
		super(LevenshteinEditDistance, self).__init__()
		self.del_cost = self._del_cost
		self.ins_cost = self._ins_cost
		self.sub_cost = self._sub_cost

	def _del_cost(self, c):
		return 1

	def _ins_cost(self, c):
		return 1

	def _sub_cost(self, x, y):
		return 2 if x != y else 0

	def edit_distance(self, source, target):
		table = [[0 for i in range(len(target) + 1)] for j in range(len(source) + 1)]
		for i in range(1, len(source) + 1):
			table[i][0] = table[i-1][0] + self.del_cost(source[i - 1])
		for i in range(1, len(target) + 1):
			table[0][i] = table[0][i-1] + self.ins_cost(target[i - 1])
		for i in range(1, len(source) + 1):
			for j in range(1, len(target) + 1):
				table[i][j] = min(table[i][j-1] + self.ins_cost(target[j-1]),
								  table[i-1][j] + self.del_cost(source[i-1]),
								  table[i-1][j-1] + self.sub_cost(source[i-1], target[j-1]))
		return table[len(source)][len(target)]


if __name__ == '__main__':
	lev = LevenshteinEditDistance()
	print(lev.edit_distance('intention', 'execution'))
	x = input()