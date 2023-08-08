#!/bin/sh
source ~/miniconda3/bin/activate txlm

CONFIG=$1

python -u tools/train_spm.py --config configs/${CONFIG}
