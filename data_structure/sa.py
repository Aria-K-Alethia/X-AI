'''
	Copyright (c) 2019 Aria-K-Alethia@github.com / xindetai@Beihang University

	Description:
		assistant function for building suffix array and lcp
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

def construct_sa(s):
	inf = float('inf')
	n = len(s)
	sa = [i for i in range(n+1)]
	rank = [ord(s[i]) for i in range(n)]
	rank.append(-inf)
	tmp = [0 for i in range(n+1)]
	k = 1

	def sa_cmp(index):
		a = rank[index]
		b = rank[index + k] if index + k <= n else -inf
		return (a, b)

	while k <= n:
		sa.sort(key = sa_cmp)
		tmp[sa[0]] = 0
		for i in range(1, len(sa)):
			tmp[sa[i]] = tmp[sa[i-1]] + (1 if sa_cmp(sa[i-1]) != sa_cmp(sa[i]) else 0)
		for i in range(len(sa)):
			rank[i] = tmp[i]
		k *= 2

	return sa

def construct_lcp(s, sa):
	n = len(s)
	rank = [0 for i in range(n+1)]
	lcp = [0 for i in range(n)]
	for i in range(n+1):
		rank[sa[i]] = i
	h = 0
	for i, c in enumerate(s):
		k = sa[rank[i] - 1]
		if h > 0: h -= 1
		while i+h < n and k+h < n and s[i+h] == s[k+h]:
			h += 1
		lcp[rank[i] - 1] = h
	return lcp

def construct_sa_lcp(s):
	inf = float('inf')
	n = len(s)
	sa = [i for i in range(n+1)]
	rank = [ord(s[i]) for i in range(n)]
	rank.append(-inf)
	tmp = [0 for i in range(n+1)]
	k = 1

	def sa_cmp(index):
		a = rank[index]
		b = rank[index + k] if index + k <= n else -inf
		return (a, b)

	while k <= n:
		sa.sort(key = sa_cmp)
		tmp[sa[0]] = 0
		for i in range(1, len(sa)):
			tmp[sa[i]] = tmp[sa[i-1]] + (1 if sa_cmp(sa[i-1]) != sa_cmp(sa[i]) else 0)
		for i in range(len(sa)):
			rank[i] = tmp[i]
		k *= 2

	lcp = [0 for i in range(n)]
	for i in range(n+1):
		rank[sa[i]] = i
	h = 0
	for i, c in enumerate(s):
		k = sa[rank[i] - 1]
		if h > 0: h -= 1
		while i+h < n and k+h < n and s[i+h] == s[k+h]:
			h += 1
		lcp[rank[i] - 1] = h
	return sa, lcp

if __name__ == '__main__':
	s = "abracadabra"
	sa, lcp = construct_sa_lcp(s)
	for a, l in zip(sa, lcp):
		print(a, l, s[a:])
	print(len(sa), len(lcp))
	print(sa[-1], s[sa[-1]:])
	sa2 = construct_sa(s)
	lcp2 = construct_lcp(s,sa2)
	assert all(a == b for a, b in zip(sa, sa2))
	assert all(a == b for a, b in zip(lcp, lcp2))

