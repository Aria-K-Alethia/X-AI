'''
	Copyright (c) 2019 Aria-K-Alethia@github.com / xindetai@Beihang University

	Description:
		segment tree for range minimal query(rmq)
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

class RmqST(object):
	"""

	"""
	def __init__(self, array, size = None):
		super(RmqST, self).__init__()
		if size is None:
			size = len(array)
		assert len(array) == size, "size %d is not equal to len(array) %d" % (size, len(array))
		self.size = 1
		while(self.size < size): self.size *= 2
		self.__inf = float('inf')
		self.tree = [self.__inf for _ in range(self.size * 2 - 1)]
		_ = [self.update(i, v) for i, v in enumerate(array)]

	def __len__(self):
		return self.size

	def update(self, index, value):
		index += self.size - 1
		self.tree[index] = value
		while(index > 0):
			index = (index - 1) // 2
			self.tree[index] = min(self.tree[index*2 + 1], self.tree[index*2 + 2])

	def _query(self, s, t, index, l, r):
		if s >= r or t <= l: return self.__inf
		if s <= l and t >= r: return self.tree[index]
		else:
			lv = self._query(s, t, index*2 + 1, l, (l+r) // 2)
			rv = self._query(s, t, index*2 + 2, (l+r) // 2, r)
			return min(lv, rv)

	def query(self, s, t):
		'''
			overview:
				return minimal value between [s, t)
		'''
		return self._query(s, t, 0, 0, self.size)

def test():
	import random
	temp = [random.randint(0, 1000) for i in range(1000)]
	rmq = RmqST(temp)
	for i in range(100):
		a, b = random.randint(0, 1000), random.randint(0, 1000)
		if a > b:
			a, b = b, a
		if a == b:
			assert rmq.query(a, b) == float('inf')
		else:
			assert rmq.query(a, b) == min(temp[a:b])
	assert rmq.query(100, 101) == temp[100]
	assert rmq.query(0, len(temp)) == min(temp)
	print(rmq.query(100, 100))

if __name__ == '__main__':
	test()
	

