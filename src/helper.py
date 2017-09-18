#coding:utf8
from collections import namedtuple
from copy import deepcopy
import numpy
import random

random.seed(1234)

Config = namedtuple('parameters',
        ['vocab_size', 'embedding_dim', 'position_size','position_dim','word_input_size','sent_input_size',
        'word_GRU_hidden_units','sent_GRU_hidden_units','pretrained_embedding'])
    
class Document():
    def __init__(self,content, label, summary):
        self.content = content
        self.label = label
        self.summary = summary
class Dataset():
    def __init__(self, data_list):
        self._data = data_list
    def __len__(self):
        return len(self._data)
        
    def __call__(self, batch_size, shuffle = True):
        max_len = len(self)
        if shuffle:
            random.shuffle(self._data)
        batchs = [self._data[index:index + batch_size] for index in range(0, max_len, batch_size)]
        return batchs
    def __getitem__(self, index):
        return self._data[index]
class DataLoader():
    def __init__(self, dataset, batch_size = 1, shuffle = True):
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size
    def __iter__(self):
        return iter(self.dataset(self.batch_size, self.shuffle))

def prepare_data(doc, word2id):
    data = deepcopy(doc.content)
    max_len = -1
    for sent in data:
        words = sent.strip().split()
        max_len = max(max_len, len(words))
    sent_list = []
     
    for sent in data:
        words = sent.strip().split()
        sent = [word2id[word] if word in word2id else 1 for word in words]
        sent += [0 for _ in range(max_len - len(sent))]
        sent_list.append(sent)
    
    sent_array = numpy.array(sent_list)
    label_array = numpy.array(doc.label)

    return sent_array, label_array

def test(doc, probs, id):
    probs = [prob[0] for prob in probs]
    predict = [1 if prob >= 0.5 else 0 for prob in probs]
    
    index = range(len(probs))
    probs = zip(probs,index)
    probs.sort(key = lambda x: x[0], reverse = True)

    num_of_sents = min(3, len(probs))
    summary_index = [probs[i][1] for i in range(num_of_sents)]
    summary_index.sort()
    hyp = [doc.content[i] for i in summary_index]

    ref = doc.summary

    with open('../result/ref/ref.' + str(id) + '.summary', 'w') as f:
        f.write('\n'.join(ref))
    with open('../result/hyp/hyp.' + str(id) + '.summary', 'w') as f:
        f.write('\n'.join(hyp))
    
    return hyp,doc.label, predict
if __name__ == '__main__':
    pass
