'''	
	Copyright (c) 2019 Aria-K-Alethia@github.com

	Description:
		EDS class to manipulate EDS
	Licence:
		MIT

	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.

'''
import re

class Graph(object):
	"""
		Naive Graph Class, with nodes and edges
	"""
	def __init__(self, nodes, edges):
		self.nodes = nodes
		self.edges = edges


class Node(object):
	"""

	"""
	def __init__(self, name, lemma, span, word, extra):
		super(Node, self).__init__()
		self.name = name
		self.lemma = lemma
		self.span = span
		self.word = word
		self.extra = extra
		
	def __str__(self):
		return '{}:{}<{}:{}>({}){{{}}}'.format(
			self.name, self.lemma, self.span[0],
			self.span[1], self.word, self.extra)
	def __repr__(self):
		return self.__str__()

class EDS(Graph):
	"""
		Elementary Dependency Structures
	"""
	def __init__(self, top, nodes, edges, string):
		super(EDS, self).__init__(nodes, edges)
		self.top = top
		self.string = string

	@staticmethod
	def parse_string(string):
		'''
			overview:
				parsing a eds string representation
				into a EDS instance
			params:
				string: eds string
			return:
				EDS instance
		'''
		string = string.strip(' \n{}')
		top, nodes, edges = None, [], []
		# top
		topend = string.index(':')
		top = string[:topend]
		string = string[topend+1:].strip()
		# nodes and edges
		strings = string.split('\n')
		pattern1 = re.compile(r'(.*):(.*)<(\d+):(\d+)>(.*)\{(.*)\}\[(.*)\]')
		pattern2 = re.compile(r'(.*):(.*)<(\d+):(\d+)>(.*)\[(.*)\]')
		for string in strings:
			if not string:
				continue
			string = string.strip()
			if '{' in string:
				pattern = pattern1
				extra_flag = True
			else:
				pattern = pattern2
				extra_flag = False
			m = pattern.match(string)
			name, lemma = m.group(1), m.group(2)
			temp = re.split('[/_]', lemma)
			temp = [item for item in temp if item]
			lemma = temp[0]
			span = (m.group(3), m.group(4))
			word = m.group(5).strip('"()')
			word = word if word else None
			if extra_flag:
				extra = m.group(6)
				espos = 7
			else:
				extra = None
				espos = 6
			node = Node(name, lemma, span, word, extra)
			nodes.append(node)
			es = m.group(espos).split(',')
			start = name
			for e in es:
				if not e:
					continue
				temp = e.strip().split()
				role = temp[0].strip()
				end = temp[1].strip()
				edges.append((start, role, end))
		return top, nodes, edges

	def __str__(self):
		return self.string

	def __repr__(self):
		return self.__str__()
