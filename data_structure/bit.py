'''
	Copyright (c) 2019 Aria-K-Alethia@github.com / xindetai@Beihang University

	Description:
		Binary Indexed Tree(BIT)
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

class BIT(object):
	"""

	"""
	def __init__(self, array, size = None):
		super(BIT, self).__init__()
		if size is None:
			size = len(array)
		assert len(array) == size, "size %d is not equal to len(array) %d" % (size, len(array))
		self.size = size
		self.bit = [0 for _ in range(self.size + 1)]
		_ = [self.add(i, v) for i, v in enumerate(array)]

	def __len__(self):
		return self.size

	def add(self, index, value):
		'''
			overview:
				add value to a[index]
		'''
		index += 1
		while(index <= self.size):
			self.bit[index] += value
			index += index & -index

	def sum(self, index):
		'''
			overview:
				summation of a[0]...a[index-1]
		'''
		s = 0
		while(index > 0):
			s += self.bit[index]
			index -= index & -index
		return s

class TwoDimBIT(object):
	"""

	"""
	def __init__(self, matrix, size = None):
		super(TwoDimBIT, self).__init__()
		if size is None:
			size = (len(matrix), len(matrix[0]))
		assert size[0] == len(matrix) and size[1] == len(matrix[0]),\
			"size %s do not equal to matrix size (%d, %d)"\
				% (size, len(matrix), len(matrix[0]))
		self._size = size
		self.bit = [[0 for i in range(self.size[1]+1)] for j in range(self.size[0]+1)]
		_ = [self.add(x, y, v) for x, line in enumerate(matrix) for y, v in enumerate(line)]

	@property
	def size(self):
		return self._size
	
	def add(self, x, y, v):
		'''
			overview:
				add v to matrix[x][y]
		'''
		x += 1
		y += 1
		i, j = x, y
		while i <= self.size[0]:
			while j <= self.size[1]:
				self.bit[i][j] += v
				j += j & -j
			j = y
			i += i & -i

	def sum(self, x, y):
		'''
			overview:
				sum over x line, each line sum y elem
		'''
		if x == 0 and y == 0:
			return 0
		s = 0
		i, j = x, y
		while i > 0:
			while j > 0:
				s += self.bit[i][j]
				j -= j & -j
			j = y
			i -= i & -i
		return s
		

def test_bit():
	import random
	temp = [random.randint(0, 1000) for i in range(1000)]
	bit = BIT(temp)
	for i in range(100):
		a = random.randint(1, 1000)
		b = random.randint(0, 100)
		assert sum(temp[:a]) == bit.sum(a)
		bit.add(a-1, b)
		temp[a-1] += b
		assert sum(temp[:a]) == bit.sum(a)
	assert 0 == bit.sum(0)
	assert temp[0] == bit.sum(1)
	assert sum(temp) == bit.sum(1000)

def test_twodimbit():
	import random
	x, y = random.randint(50, 100), random.randint(50, 100)
	mat = [[random.randint(-1000, 1000) for j in range(y)] for i in range(x)]
	bit = TwoDimBIT(mat)
	for i in range(100):
		a, b, v = random.randint(0, x - 1), random.randint(0, y - 1), random.randint(-1000, 1000)
		a = b = 1
		assert bit.sum(a+1, b+1) == sum(sum(line[:b+1]) for line in mat[:a+1])
		bit.add(a, b, v)
		mat[a][b] += v
		assert bit.sum(a+1, b+1) == sum(sum(line[:b+1]) for line in mat[:a+1])
	assert bit.sum(0, 0) == 0
	assert mat[0][0] == bit.sum(1, 1)
	assert bit.sum(x, y) == sum(sum(line) for line in mat)

if __name__ == '__main__':
	test_bit()
	test_twodimbit()