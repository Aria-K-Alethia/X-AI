# Copyright (c) 2018 Aria-K-Alethia@github.com
# Description:
#   An abstract embedding class

import numpy as np
import torch
from collections import defaultdict
from vocab import PAD

class Embedding(object):
    """

    """
    def __init__(self, size, dim, word2idx):
        '''
            overview:
                An abstract Embedding class containing the following functions:
                1. create embedding table.
                2. init embedding.
                3. load pretrained embedding from other file, e.g. GloVe
                4. dump embedding tensor to different format, e.g. numpy pytorch
            params:
                size: word count
                dim: embedding length
                word2idx: a dict, which maps word to its index
        '''
        self._size = size
        self._dim = dim
        self._word2idx = word2idx
        self._embedding = None
    def init_embedding(self, method="uniform"):
        '''
            overview:
                using method to init the embedding
            params:
                method: can be uniform
        '''
        if(method == "uniform"):
            self._embedding = np.random.uniform(low = -0.1, high = 0.1, size = (self._size,self._dim))
            # PAD should be zero
            self._embedding[self._word2idx[PAD]] = 0
            self._embedding = np.asarray(self._embedding,dtype=np.float32)
    def load_pretrained(self,file,typ="glove"):
        '''
            overview:
                loading pretrained embedding from file.
            params:
                file: the embedding file path
                type: the type of embedding file, can be glove or word2vec
            NOTE:
                this method would only load those embedding in the wordlist
        '''
        print("loading embedding from %s..." % file)
        count = 0
        record = defaultdict(int)
        with open(file, 'r') as infile:
            if(typ == 'word2vec'):
                # consume the first line of word2vec file
                _ = infile.readline()
            for line in infile:
                #format: <WORD> <EMBEDDING>
                temp = line.strip().split(' ')
                word,vec = temp[0],temp[1:]
                #word and vec should already obtained
                vec = list(map(float,vec))
                if(len(vec) < self._dim):
                    raise Exception("The embedding dimension in the file should"+
                        " be equal or greater than original dimension"+
                        "vec len: %d, dim: %d" % (len(vec), self._dim))
                elif(len(vec) > self._dim):
                    print("Warning: the embedding dimension in the file is "+
                        "greater than original dimension, the exceeded part "+
                        "will be truncated")
                vec = np.asarray(vec[:self._dim])
                for w in [word, word.upper(), word.lower()]:
                    if(w in self._word2idx):
                        self._embedding[self._word2idx[w]] = vec
                        if(record[w] == 0):
                            count += 1
                        record[w] += 1
                        break
        print("Totally read %d embedding from file" % count)
        print("Embedding size: %d, Rest: %d" % (self._size, self._size - count))
        print("load pretrained vector done")
    @property
    def embedding(self):
        return self._embedding
    @property
    def size(self):
        return self._size
    @property
    def dim(self):
        return self._dim
    def to_file(self,file,typ="pytorch"):
        ''' 
            overview:
                dump the embedding to file
            params:
                file: file path
                typ: can be pytorch or numpy, which correspond to torch.save
                      and numpy.save
        '''
        if(typ == "pytorch"):
            temp = torch.tensor(self._embedding)
            torch.save(temp,file)
        elif(typ == "numpy"):
            np.save(file,self._embedding)

