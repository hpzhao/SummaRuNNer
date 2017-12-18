#!/usr/bin/env python
#coding:utf8
import argparse
import logging
import torch
import argparse
import sys
import random
import cPickle as pkl
import torch.nn as nn
from helper import Config
from helper import Dataset
from helper import DataLoader
from helper import prepare_data
from model import SummaRuNNer
from torch.autograd import Variable

torch.manual_seed(233)
logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--emb_file', type=str, default='../data/embedding.pkl')
parser.add_argument('--train_file', type=str, default='../data/train.pkl')
parser.add_argument('--validation_file', type=str, default='../data/validation.pkl')
parser.add_argument('--model_file', type=str, default='../model/summary.model')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

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

logging.info('loadding train dataset')
train_dataset = pkl.load(open(args.train_file))
train_loader = DataLoader(train_dataset)

logging.info('loadding validation dataset')
validation_dataset = pkl.load(open(args.validation_file))
validation_loader = DataLoader(validation_dataset, shuffle = False)

net = SummaRuNNer(config)
net.cuda()

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# training
loss_sum = 0
min_eval_loss = float('Inf')
for epoch in range(args.epochs):
    for step, docs in enumerate(train_loader):
        doc = docs[0]
        x, y = prepare_data(doc, word2id)
        sents = Variable(torch.from_numpy(x)).cuda()
        labels = Variable(torch.from_numpy(y)).cuda()
        labels = labels.float()
        # Forward + Backward + Optimize  
        outputs = net(sents)
        #print outputs
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss_sum += loss.data[0]
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm(net.parameters(), 1e-4)
        optimizer.step()
        if step % 1000 == 0 and step != 0: 
            logging.info('Epoch ' + str(epoch) + ' Loss: ' + str(loss_sum / 1000.0))
            loss_sum = 0
        if step % 10000 == 0 and step != 0:
            eval_loss = 0
            for step, docs in enumerate(validation_loader):
                doc = docs[0]
                x, y = prepare_data(doc, word2id)
                sents = Variable(torch.from_numpy(x)).cuda()
                labels = Variable(torch.from_numpy(y)).cuda()
                labels = labels.float()
                outputs = net(sents)
                loss = criterion(outputs, labels)
                eval_loss += loss.data[0]
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                torch.save(net.state_dict(), args.model_file)
                logging.info('epoch ' + str(epoch) + ' Loss in validation: ' + str(eval_loss * 1.0 / len(validation_dataset)))
