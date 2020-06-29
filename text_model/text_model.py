'''
    Copyright (c) 2020 Aria-K-Alethia@github.com

    Description:
        text model class for generating text
    Licence:
        MIT
    THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
    Any use of this code should display all the info above.
'''

import random
from math import log2

class TextModel(object):
    """
    """
    def __init__(self, wordlist, weights):
        super(TextModel, self).__init__()
        self.wordlist = wordlist
        temp = sum(weights)
        self.prior = [weight / temp for weight in weights]

    def entropy(self):
        H = 0
        for p in self.prior:
            if p != 0:
                H += p * log2(p)
        return -H

    def random_text(self, size):
        assert size > 0
        out = random.choices(self.wordlist, weights=self.prior, k=size)
        return ''.join(out)

    def kl_divergence(self, textmodel):
        m1 = dict(zip(self.wordlist, self.prior))
        m2 = dict(zip(textmodel.wordlist, textmodel.prior))
        kl = 0
        for x in m1:
            p1 = m1[x]
            p2 = m2[x]
            kl += p1 * log2(p1 / p2)
        return kl

if __name__ == '__main__':
    model = TextModel(['a', 'b', 'c', 'd'], weights=[1,2,3,4])
    text = model.random_text(1000)
    from collections import Counter
    print(Counter(text))
