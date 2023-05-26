#!/bin/sh
source ~/miniconda3/bin/activate txlm

CONFIG=$1

python -u tools/sample_and_merge_langs.py --config configs/${CONFIG}

python -u tools/train_spm.py --config configs/${CONFIG}
