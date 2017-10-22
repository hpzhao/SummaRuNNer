#!/usr/bin/env python
#coding:utf8
import argparse
import logging
import torch
import sys
import cPickle as pkl
import torch.nn as nn
from helper import Config
from helper import Dataset
from helper import DataLoader
from helper import prepare_data
from helper import test
from model import SummaRuNNer
from torch.autograd import Variable

torch.manual_seed(233)
logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--emb_file', type=str, default='../data/embedding.pkl')
parser.add_argument('--test_file', type=str, default='../data/test.pkl')
parser.add_argument('--model_file', type=str, default='../model/summary.model')
parser.add_argument('--hidden', type=int, default=200)
args = parser.parse_args()

logging.info('generate config')

pretrained_embedding = pkl.load(open(args.emb_file))
config = Config(
        vocab_size = pretrained_embedding.shape[0],
        embedding_dim = pretrained_embedding.shape[1],
        position_size = 500,
        position_dim = 50,
        word_input_size = 100,
        sent_input_size = 2 * args.hidden,
        word_GRU_hidden_units = args.hidden,
        sent_GRU_hidden_units = args.hidden,
        pretrained_embedding = pretrained_embedding)

word2id = pkl.load(open('../data/word2id.pkl'))

logging.info('loadding test dataset')
test_dataset = pkl.load(open(args.test_file))
test_loader = DataLoader(test_dataset, shuffle = False)

net = SummaRuNNer(config).cuda()
net.load_state_dict(torch.load(args.model_file))

for index, docs in enumerate(test_loader):
    doc = docs[0]
    x, y = prepare_data(doc, word2id)
    sents = Variable(torch.from_numpy(x)).cuda()
    outputs = net(sents)
    hyp, gold, predict = test(doc, outputs.data.tolist(), index)
