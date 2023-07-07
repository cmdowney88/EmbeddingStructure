#!/bin/sh
VENVNAME=$1
DEVICES=$2
CONFIG=$3

source ~/miniconda3/bin/activate $VENVNAME

CUDA_VISIBLE_DEVICES=$DEVICES python -u src/eval_finetune.py --config $CONFIG