# Copyright (c) 2018 Aria-K-Alethia@github.com
# Description:
#   An abstract vocab class to support NLP task.

from collections import Counter

#define some dummy word, which are usually used in NLP tasks
SENTENCE_START = '<SOS>'
SENTENCE_END   = '<EOS>'
PAD            = '<PAD>'
UNK            = '<UNK>'
DUMMY_TOKEN_LIST = [PAD, UNK, SENTENCE_START, SENTENCE_END]

class Vocab(object):
    """
        This is a abstract class of vocabulary
        The designing goal is to reduce our work in NLP
    """
    def __init__(self,MAX_SIZE,keep_frequency=False):
        '''
            overview:
                init vocab
            params:
                MAX_SIZE: 
                    the max size of this vocab, doesn't contain the 4 dummy tokens
                keep_frequency:
                    if True, the word frequency would be stored,default False
            NOTE:
                the final max_size will be MAX_SIZE+4, cause we will add 4 dummy tokens
        '''
        self._DUMMY_TOKEN_LIST = DUMMY_TOKEN_LIST
        self._max_size = MAX_SIZE + len(DUMMY_TOKEN_LIST)
        self._size = 0
        self._frequency = {} if keep_frequency else None
        self._word2idx = {}
        self._idx2word = {}
        for token in DUMMY_TOKEN_LIST:
            self._word2idx[token] = self._size
            self._idx2word[self._size] = token
            self._size += 1
    def clear(self, MAX_SIZE = None):
        self._max_size = self._max_size if MAX_SIZE != None else MAX_SIZE + \
                        len(self._DUMMY_TOKEN_LIST)
        self._size = 0
        if(self._frequency != None):
            self._frequency.clear()
        self._word2idx.clear()
        self._idx2word.clear()
        for token in DUMMY_TOKEN_LIST:
            self._word2idx[token] = self._size
            self._idx2word[self._size] = token
            self._size += 1
    def frequency(self,word):
        return -1 if (self._frequency == None or word not in self._frequency) else self._frequency[word]
    def word_to_idx(self,word):
        if(word in self._word2idx):
            return self._word2idx[word]
        else:
            return self._word2idx[self.UNK]
    def idx_to_word(self,idx):
        return (None if idx >= self._size else self._idx2word[idx])
    def _add(self, word, freq = None):
        if(self._size >= self._max_size):
            print("WARNING: reaching the max size, can't add")
            return False
        self._word2idx[word] = self._size
        self._idx2word[self._size] = word
        self._size += 1
        if(self._frequency != None and freq != None):
            self._frequency[word] = freq
        elif(self._frequency != None and freq == None):
            print("WARNING: frequency set, but no freq to add")
        elif(self._frequency == None and freq != None):
            print("WARNING: frequency doesn't set, but having freq to add")
        else:
            pass
        return True

    def from_file(self,path,delimiter=' ',with_frequency=True):
        '''
            overview:
                load vocab from file.
                the line format of the file should be:
                <WORD><Delimiter><FREQUENCY>\n
                or if with_frquency = False:
                <WORD>\n
            params:
                path: file path
                with_frequency: default True
                delimiter: needed when with_frequency == True,default ' '
            NOTE:
                1. lines go beyond the max_size would be ignored
        '''
        infile = open(path,'r',encoding='utf8')
        print("Load vocab from: %s ..." % path)
        lines = infile.read().split('\n')
        infile.close()
        if(lines[-1] == ''):
            _ = lines.pop(-1)
        word_len = len(lines)
        print('Total word: %d' % word_len)
        word_count = 0
        for line in lines:
            if(self._size >= self._max_size): break
            if(with_frequency):
                temp = line.split(delimiter)
                word,freq = temp[0],int(temp[1])
            else:
                word = line
                freq = None
            self._add(word,freq)
            word_count += 1
        print("Read: %d word, Ignored: %d word" % (word_count, word_len - word_count))
        print("done")
    def to_file(self,path,with_frequency=True):
        ''' 
            overview:
                dump the current vocab to file
            params:
                path: file path
        '''
        print("open the file %s..." % path)
        outfile = open(path,'w',encoding='utf8')
        for key in self._word2idx.keys():
            if(key in self._DUMMY_TOKEN_LIST): continue
            if(with_frequency and self._frequency == None):
                raise Exception("Can't output frequency, " +
                    "this vocab has no frequency dict!")
            if(with_frequency):
                outstring = key+" "+str(self._frequency[key])+"\n"
            else:
                outstring = key+"\n"
            outfile.write(outstring)
        outfile.close()
        print("Totally write %d words" % self._size)
        print("done")
    def make_from_corpus(self, target, mode='string', range = (100, 80000), tfidf = None):
        '''
            overview:
                make vocab from corpus.
            params:
                target: the target string or file
                mode: must be string or file
                range: freq range to save
                tfidf: sklearn.feature_extraction.text.TfidfVectorizer
            NOTE:
                This method would be illegal if you have loaded
                other vocab from file i.e. self.size > 4
                In this case you should call vocab.clear()
                to clear this vocab
            
        '''
        assert mode in ['string', 'file']
        assert len(self._word2idx) == len(self._DUMMY_TOKEN_LIST)
        word_count = 0
        max_freq, min_freq = 0, 0
        counter = Counter()
        if(mode == 'string'):
            counter.update(target.strip().split(' '))
        elif(mode == file):
            try:
                file = open(target, 'r', encoding = 'utf8')
            except Exception as e:
                raise e
            for line in file:
                counter.update(line.strip().split(' '))
        else:
            raise ValueError("No such mode")
        max_freq = -1
        min_freq = float('inf')
        if tfidf is not None:
            vocab = tfidf.vocabulary_
            idf = tfidf.idf_
            idf = (idf - idf.min()) / (idf.max() - idf.min())
            max_freq = counter.most_common(1)[0][1]
            for key in counter:
                if key in vocab:
                    counter[key] = (counter[key] - 1) / (max_freq - 1)
                    counter[key] *= idf[vocab[key]] * 1e8
                else:
                    counter[key] = 0
            max_freq = counter.most_common(1)[0][1]
            min_freq = float('inf')
            for w, f in counter.most_common():
                if not self._add(w, f):
                    break
                if min_freq > f: min_freq = f
                word_count += 1
        else: 
            for w, f in sorted(counter.items(), key = lambda x: -x[1]):
                if not range[0] <= f <= range[1]:
                    continue
                if max_freq < f: max_freq = f
                if min_freq > f: min_freq = f
                if not self._add(w,f):
                    break
                word_count += 1
        print("total: %d words, read: %d words, ignored: %d words" % (
                                                    len(counter), word_count,
                                                    len(counter) - word_count))
        print("max frequency: %d, min frequency: %d" % (max_freq, min_freq))
        print("done")
    @property
    def size(self):
        return self._size
    @property
    def maxsize(self):
        return self._max_size
    @property
    def PAD(self):
        return PAD
    @property
    def UNK(self):
        return UNK
    @property
    def SOS(self):
        return SENTENCE_START
    @property
    def EOS(self):
        return SENTENCE_END
    @property
    def PAD_INDEX(self):
        return self._word2idx[PAD]
    @property
    def UNK_INDEX(self):
        return self._word2idx[UNK]
    @property
    def SOS_INDEX(self):
        return self._word2idx[SOS]
    @property
    def EOS_INDEX(self):
        return self._word2idx[EOS]
