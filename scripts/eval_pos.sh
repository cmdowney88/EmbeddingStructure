#!/bin/sh
source ~/miniconda3/bin/activate $1

CONFIG=$2

python -u src/eval_finetune.py --config $CONFIG
