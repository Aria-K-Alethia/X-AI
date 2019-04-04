# X-NLP
A Customized NLP toolkit.
The code in this repo is designed for NLP or at least Machine Learning Task.  

## Function
### Algorithm
Algorithms used in NLP

- penalty, used in beam search, see the paper <<*Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*>>
- Layer Norm
- EditDistance, useful for many NLP tasks, like spelling correction

### Modules

Useful modules which could be adopted to your code easily

- multi-layer LSTM
- embedding, support pretrained embedding
- optimizer, simple wrapper of pytorch's optimizer, support learning rate schedule strategy
- mlp, multi-layer perceptron, support customized layer number and activation function
- multi-feature embedding, support embedding with multiple source(like linguistic feature), with customized merge method
- highway, highway network layer implementation of the paper <<*Highway Network*>>
### Preprocess

Common tools that you may use in text processing.

- Vocab, simple but effective vocabulary class

### Misc

- EDS, used to process eds, also includes a method to parse text repr.
- utils, includes various functions used in NLP, Machine Learning and Deep Learning.

## TODO

- [ ] LSTM CELL
- [ ] Transformer
- [ ] word2vec

## Requirements

The code is implemented with Python3, some of the code(not all) may need to use the other packages.

- Python3
- Pytorch >= 0.4
- nltk
- numpy

## Licence
This code is under MIT Licence.
Any use of this code should display all the info above; the user agrees to assume all liability for the use of this code.
