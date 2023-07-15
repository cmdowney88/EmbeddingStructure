#!/bin/sh
PATH_TO_CONDA=$1
VENVNAME=$2
DEVICES=$3
CONFIG=$4

source ${PATH_TO_CONDA}/miniconda3/bin/activate $VENVNAME

CUDA_VISIBLE_DEVICES=$DEVICES python -u src/eval_finetune.py --config $CONFIG
