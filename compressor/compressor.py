'''
    Copyright (c) 2020 Aria-K-Alethia@github.com

    Description:
        text conpressor using huffman code
    Licence:
        MIT
    THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
    Any use of this code should display all the info above.
'''
from collections import Counter
from time import time as stdtime

class TreeNode(object):
    """
    """
    def __init__(self, value, freq):
        super(TreeNode, self).__init__()
        self.value = value
        self.left = None
        self.right = None
        self.freq = freq

class Compressor(object):
    """
    """
    def __init__(self):
        super(Compressor, self).__init__()
        self.word2code = None
        self.tree = None

    def _fit(self, word_freq):
        assert len(word_freq) > 0
        nodes = [TreeNode(word, freq) for word, freq in word_freq]
        nodes.sort(key=lambda x: x.freq)
        while len(nodes) > 1:
            left = nodes.pop(0)
            right = nodes.pop(0)
            new_node = TreeNode(None, left.freq + right.freq)
            new_node.left = left
            new_node.right = right
            if not nodes:
                break
            for i, node in enumerate(nodes):
                if node.freq > new_node.freq:
                    break
            nodes.insert(i, new_node)
        root = new_node if len(word_freq) > 1 else nodes[0]
        self._encode(root)

    def _encode(self, root):
        self.word2code = {}
        def recur(node, code):
            if node.value is not None:
                self.word2code[node.value] = code
            if node.left is not None:
                recur(node.left, code+'0')
            if node.right is not None:
                recur(node.right, code+'1')
        if root.left is None and root.right is None:
            self.word2code[root.value] = '0'
        else:
            recur(root, '')

    def fit_textmodel(self, textmodel):
        word_freq = list(zip(textmodel.wordlist, textmodel.prior))
        self._fit(word_freq)

    def fit_text(self, text):
        c = Counter(text)
        word_freq = c.most_common()
        self._fit(word_freq)

    def compress(self, text):
        assert self.word2code is not None
        begin = stdtime()
        out = []
        for c in text:
            out.append(self.word2code[c])

        out = ''.join(out)
        compress_rate = len(out) / len(text)
        time = stdtime() - begin
        print('Compress time: {:.2f}s, compression rate: {}'.format(time, compress_rate))
        return out

    def fit_compress(self, text):
        self.fit_text(text)
        return self.compress(text)

if __name__ == '__main__':
    from text_model import TextModel
    model = TextModel(['a', 'b', 'c', 'd'], [1, 5, 3, 7])
    print(model.entropy())
    c1 = Compressor()
    c2 = Compressor()
    c1.fit_textmodel(model)
    text = model.random_text(1000)
    out1 = c1.compress(text)
    print(c1.word2code)
    out2 = c2.fit_compress(text)
    print(c2.word2code)
