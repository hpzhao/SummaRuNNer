#!/usr/bin/env python
#coding:utf8
import argparse
import torch
from utils import Vocab,Dataset
from tqdm import tqdm
from glob import glob
from time import time
PAD_IDX = 0
UNK_IDX = 1
PAD_TOKEN = 'PAD_TOKEN'
UNK_TOKEN = 'UNK_TOKEN'

def build_vocab(args):
    f = open(args.embed)
    embed_dim = int(next(f).split()[1])

    word2id = {}
    id2word = {}
    
    word2id[PAD_TOKEN] = PAD_IDX
    word2id[UNK_TOKEN] = UNK_IDX
    id2word[PAD_IDX] = PAD_TOKEN
    id2word[UNK_IDX] = UNK_TOKEN
    
    embed_list = []
    # fill PAD and UNK vector
    embed_list.append([0 for _ in range(embed_dim)])
    embed_list.append([0 for _ in range(embed_dim)])
    
    # build Vocab

    for line in f:
        tokens = line.split()
        word = tokens[0]
        vector = [float(num) for num in tokens[1:]]
        
        embed_list.append(vector)
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    
    embed = torch.FloatTensor(embed_list)
    vocab = Vocab(embed, word2id, id2word)
    torch.save(vocab,args.vocab)

def build_dataset(args):
    examples = []
    for f in tqdm(glob(args.source_dir)):
        parts = open(f).read().split('\n\n')
        try:
            entities = { line.strip().split(':')[0]:line.strip().split(':')[1] for line in parts[-1].split('\n')}
        except:
            continue
        sents,labels,summaries = [],[],[]
        # content
        for line in parts[1].strip().split('\n'):
            content, label = line.split('\t\t\t')
            tokens = content.strip().split()
            for i,token in enumerate(tokens):
                if token in entities:
                    tokens[i] = entities[token]
            label = '1' if label == '1' else '0'
            sents.append(' '.join(tokens))
            labels.append(label)
        # summary
        for line in parts[2].strip().split('\n'):
            tokens = line.strip().split()
            for i, token in enumerate(tokens):
                if token in entities:
                    tokens[i] = entities[token]
            line = ' '.join(tokens).replace('*','')
            summaries.append(line)
        ex = {'doc':'\n'.join(sents),'labels':'\n'.join(labels),'summaries':'\n'.join(summaries)}
        examples.append(ex)
    dataset = Dataset(examples)
    torch.save(dataset,args.target_dir)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-embed', type=str, default='data/100.w2v')
    parser.add_argument('-vocab', type=str, default='data/vocab.pt')
    parser.add_argument('-source_dir', type=str, default='data/neuralsum/dailymail/test/*')
    parser.add_argument('-target_dir', type=str, default='data/test.pt')
    parser.add_argument('-build_vocab',action='store_true')

    args = parser.parse_args()
    
    if args.build_vocab:
        build_vocab(args)
    else:
        build_dataset(args)
