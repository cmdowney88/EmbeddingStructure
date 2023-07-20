#!/bin/sh
PATH_TO_CONDA=$1
VENVNAME=$2
DEVICES=$3
CONFIG=$4

echo $PATH_TO_CONDA
echo $VENVNAME
echo $DEVICES
echo $CONFIG
echo ${PATH_TO_CONDA}/miniconda3/bin/activate $VENVNAME

source ~/anaconda3/etc/profile.d/conda.sh
conda activate txlm

CUDA_VISIBLE_DEVICES=$DEVICES python -u src/eval_finetune.py --config $CONFIG
