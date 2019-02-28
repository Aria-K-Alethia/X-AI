# Copyright (c) 2018 Aria-K-Alethia@github.com
# Some useful function in DL
## most of them can only be used in pytorch

import torch
import torch.nn as nn
import nltk
import pickle


################## Routine #######################
def pickle_load(path):
    logging.info('pickle load from: %s' % (path))
    with open(path, 'rb') as infile:
        data = pickle.load(infile)
    return data

def pickle_dump(data, path):
    logging.info('pickle dump to: %s' % (path))
    with open(path, 'wb') as outfile:
        pickle.dump(data, outfile)

def dict_to_namedtuple(dict_):
    key = list(dict_.keys())
    values = list(dict_.values())
    Option = namedtuple('Option', key)
    option = Option(*values)
    return option


################ Text Processing ##################
def contain_words(sent1, sent2):
    temp1 = nltk.word_tokenize(sent1)
    temp2 = nltk.word_tokenize(sent2)
    out = list(filter(lambda x: x in temp2, temp1))
    return out

def pad_sent(sent, length, PAD):
    '''
        pad sent to length, use PAD
        NOTE:
            len(sent) <= length
    '''
    assert len(sent) <= length, "sent length: %d, length: %d" % (len(sent), length)
    padlen = length - len(sent)
    return sent + [PAD]*padlen

def sent2idx(sent, vocab):
    return [vocab.word_to_idx(w) for w in sent]

def add_sent_tag(sent, SOS, EOS):
    return [SOS] + sent + [EOS]

def add_para_tag(para, SOS, EOS):
    '''
        add SOS and EOS to para.
        note the para could contain multiple sentence
        so multiple SOS and EOS are needed
    '''
    punk = ['!','.','?']
    # add SOS and EOS in begining and ending, respectively
    para = para + [EOS]
    # find each word in punk, add EOS and SOS after it
    '''
    i = 0
    while(i < len(para) - 2):
        if(para[i] in punk):
            para.insert(i+1, EOS)
            #para.insert(i+2, SOS)
        i += 1
    '''
    return para

    

def clean_raw_data(string):
    '''
        overview:
            clean the english raw data in string
    '''
    string = string.strip().lower()
    string = ' '.join(nltk.word_tokenize(string))
    return string


################# Machine Learniing #################

def F_score(truth, pred):
    '''
        overview:
            compute the F-score between truth and prediction
            for multi-class, return the average f-score for all classes.
        params:
            truth: torch.LongTensor or list [#length]
            pred: torch.LongTensor or list [#length]
        return:
            f-score: scalar
    '''
    if isinstance(truth, list):
        t = torch.LongTensor(truth)
    if isinstance(pred, list):
        p = torch.LongTensor(pred)
    # make sure truth and pred is longtensor
    t = t.long().squeeze()
    p = p.long().squeeze()
    assert t.dim() == p.dim()
    assert t.shape[0] == p.shape[0]
    # get the label number
    temp = set(t.tolist())
    fscore = 0 
    for label in temp:
        # get TP FP FN for this label
        TP = float((t[p == label] == label).sum())
        FP = float((t[p == label] != label).sum())
        FN = float((t[p != label] == label).sum())
        precision = 1 if TP + FP == 0 else TP / (TP + FP) 
        recall = 1 if TP + FN == 0 else TP / (TP + FN) 
        fscore += 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)
    return fscore / len(temp)

################# Pytorch Deep Learning #############

def get_device(gpu):
    return torch.device('cuda' if gpu else 'cpu')

def bottle(t):
    return t.view(-1,t.shape[-1])

def merge_bidirection_state(self, state):
    '''
        overview:
            merge the state of bi-directional RNN to one direction
            support LSTM & GRU
        params:
            state: torch.tensor [#num_layers * #num_directions, #batch, #hidden_size]
                   could be a list for LSTM
        return:
            torch.tensor [#num_layers, #batch, #hidden_size * #num_directions]
            could be a list for LSTM
    '''
    if isinstance(state, list):
        ret = [torch.cat([item[0:item.size(0):2], item[1:item.size(0):2]], 2) for item in state]
    else:
        ret = torch.cat([state[0:state.size(0):2], state[1:state.size(0):2]], 2)
    return ret


def build_rnn(rnn_type, **kwargs):
    return getattr(nn, rnn_type)(**kwargs)

def tally_parameters(model):
    '''
        overview:
            tally the parameters in the model
        params:
            model: a pytorch model
    '''
    params = list(model.parameters())
    n_params = len(params)
    n_elems = sum([p.nelement() for p in params])
    print('* number of parameters: %e *' % n_params)
    print('* number of elements: %e *' % n_elems)

def align_tensors(tensors, max_length, dim, device = None):
    '''
        overview:
            align all tensor in tensors to max length at dim
        params:
            tensors: list, containing torch.tensor to be aligned
            max_length: the max length
            dim: on which dim to align
        return:
            list of tensors with the same length in dim
        NOTE:
            other dimension of the tensor in tensors should be equal
    '''
    device = device if device != None else torch.device('cpu')
    temp = tensors[0]
    dims = [i for i in temp.shape]
    for i, t in enumerate(tensors):
        dims[dim] = max_length - t.shape[dim]
        if(dims[dim] == 0):
            continue
        dummy = torch.zeros(dims, dtype = t.dtype, device = device)
        tensors[i] = torch.cat([t, dummy], dim = dim)
    return tensors

def mask_sequence(lengths,max_length = None):
    ''' 
        overview:
            generating a mask matrix,
            according to the lengths
        params:
            lengths: tensor, each elem indicates the length of the example, should be a one dim tensor
            max_length: max length of the examples, if not given, it would be generated according to the list
        return:
            mask: [#batch_size, #max_len]
                  for each row in mask, row[0:example_len] = 1,
                  row[example_len:] = 0
    '''
    assert lengths.dim() == 1
    device = lengths.device
    max_len = max_length if max_length != None else lengths.max()
    mask = torch.arange(max_len)\
            .type_as(lengths)\
            .repeat(lengths.numel(),1)\
            .lt(lengths.view(-1,1))\
            .to(device)
    return mask

def tile(tensor, count, dim = 0):
    '''
        overview:
            tile a tensor on dim
            count times
            this method is usually
            used in beam search
        example:
            >>> temp = torch.randn(2, 2)
            >>> temp2 = tile(temp, 5)
            >>> temp2 = temp2.view(2, 5, 2)
            >>> for i in range(2):
                    assert all(item.equal(temp[i]) for item in temp2[i])
    '''
    dims = list(range(tensor.dim()))
    # if dim is not 0, transpose it to dim 0
    if dim != 0:
        dims[0], dims[dim] = dims[dim], dims[0]
        tensor = tensor.permute(*dims).contiguous()
    new_shape = list(tensor.shape)
    new_shape[0] *= count
    tensor = tensor.unsqueeze(1)
    shape = [1 if i != 1 else count for i in range(tensor.dim())]
    tensor = tensor.repeat(*shape).view(*new_shape)
    # when dim != 0, we need to recover it
    if dim != 0:
        tensor = tensor.permute(*dims).contiguous()
    return tensor

def set_random_seed(seed, gpu = False):
    if seed > 0:
        torch.manual_seed(seed)
        random.seed(seed)
        if gpu:
            torch.cuda.manual_seed(seed)