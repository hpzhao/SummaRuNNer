#!/usr/bin/env python
#coding:utf8
import sys
import os
import argparse
import cPickle as pkl
from helper import Document
from helper import Dataset

parser = argparse.ArgumentParser()

parser.add_argument('-source',type=str,help='path/to/dailymail')
parser.add_argument('-target',type=str,help='path/to/save/data')
args = parser.parse_args()

def process_raw_text(source_path, target_path, prefix):
    files = os.listdir(source_path)
    dataset_list = []
    
    for num , f in enumerate(files):
        file_path = os.path.join(source_path, f)
        parts = open(file_path).read().split('\n\n')
        try:
            entities = {line.strip().split(':')[0]:line.strip().split(':')[1]
                        for line in parts[-1].split('\n')}
            sents = []
            labels = []
            summary = []
            # content
            for index, line in enumerate(parts[1].strip().split('\n')):
                if prefix == 'train' and index >= 100: break
                content, label = line.split('\t\t\t')
                tokens = content.strip().split()
                for i,token in enumerate(tokens):
                    if token in entities:
                        tokens[i] = entities[token]
                
                label = 1 if label == '1' else 0 
                if prefix == 'train' and len(tokens) > 50:
                    sents.append(' '.join(tokens[:50]))
                else:
                    sents.append(' '.join(tokens))
                
                labels.append(label)
                corpus.write(' '.join(tokens) + '\n')
            # summary
            for line in parts[2].strip().split('\n'):
                tokens = line.strip().split()
                for i, token in enumerate(tokens):
                    if token in entities:
                        tokens[i] = entities[token]
                line = ' '.join(tokens)
                line = line.replace('*','')
                summary.append(line)
            doc = Document(sents, labels, summary)
            dataset_list.append(doc)
        except:
            continue
    dataset = Dataset(dataset_list)
    pkl.dump(dataset, open(target_path + prefix + '.pkl', 'w'))

if __name__ == '__main__':
    source_path = args.source
    target_path = args.target
    
    process_raw_text(os.path.join(source_path, 'training'), target_path, 'train')
    process_raw_text(os.path.join(source_path, 'test'), target_path, 'test')
    process_raw_text(os.path.join(source_path, 'validation'), target_path, 'val')
