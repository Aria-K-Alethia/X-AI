'''
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		Trie for string matching
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

class Trie(object):
	"""
		overview:
			Trie for string matching
	"""
	def __init__(self):
		super(Trie, self).__init__()
		self.tree = {}
		self._store = 'store'

	@property
	def store(self):
		return self._store
	

	def add(self, string):
		'''
			overview:
				add string in Trie
		'''
		tree = self.tree
		if not string:
			return
		for c in string:
			if c in tree:
				tree = tree[c]
			else:
				tree[c] = {}
				tree = tree[c]
		if self.store not in tree:
			tree[self.store] = True

	def match_all(self, string):
		'''
			overview:
				match all char in string
			return:
				True: if all char is matched
				False: else
		'''
		tree = self.tree
		for c in string:
			if c in tree:
				tree = tree[c]
			else:
				return False
		if self.store in tree:
			return True

	def match_prefix(self, string):
		'''
			overview:
				match prefix of string
			return:
				if prefix of string matched
				return the matched prefix
				and existing info
				else return None
		'''
		tree = self.tree
		out = ''
		if string[0] not in tree:
			return None
		for c in string:
			if c in tree:
				tree = tree[c]
				out += c
			else:
				break
		if self.store in tree:
			return out, True
		else:
			return out, False

	def match_list(self, string_list):
		'''
			overview:
				match a list
			return:
				the maximal index
				if no match, return -1
		'''
		index = -1
		if not string_list:
			return index
		tree = self.tree
		while index+1 < len(string_list):
			temp = string_list[index + 1]
			for c in temp:
				if c in tree:
					tree = tree[c]
				else:
					break
			else:
				index += 1
				continue
			break
		return index

def test():
	trie = Trie()
	strings = ['apple', 'app', 'pear', '']
	[trie.add(string) for string in strings]
	assert trie.match_all('apple')
	assert trie.match_all('pear')
	assert not trie.match_all('')
	assert trie.match_prefix('app') == ('app', True)
	assert trie.match_prefix('peara') == ('pear', True)
	assert trie.match_prefix('appl') == ('appl', False)
	assert trie.match_list(['10', ',', '000']) == 2

if __name__ == '__main__':
	test()
