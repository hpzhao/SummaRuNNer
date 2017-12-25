#!/bin/bash
curTime=$(date "+%Y%m%d")
python train.py -train_file ../data/train.pkl -validation_file ../data/validation.pkl -model_file ../model/summary.model
