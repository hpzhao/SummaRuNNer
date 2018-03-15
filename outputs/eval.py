#!/usr/bin/env python
#coding:utf8
import os
from pyrouge import Rouge155

def remove_broken_files():
    error_id = []
    for f in os.listdir('ref'):
        try:
            open('ref/' + f).read().decode('utf8')
        except:
            error_id.append(f)
    for f in os.listdir('hyp'):
        try:
            open('hyp/' + f).read().decode('utf8')
        except:
            error_id.append(f)
    error_set = set(error_id)
    for f in error_set:
        os.remove('ref/' + f)
        os.remove('hyp/' + f)

def rouge():
    r = Rouge155()
    r.home_dir = '.'
    r.system_dir = 'hyp'
    r.model_dir =  'ref'

    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '#ID#.txt'

    command = '-e /users2/hpzhao/project/nlp-metrics/ROUGE-1.5.5/data -a -c 95 -m -n 2 -b 75'
    output = r.convert_and_evaluate(rouge_args=command)
    print output

if __name__ == '__main__':
    remove_broken_files()
    rouge()
