#!/bin/sh
source ~/miniconda3/bin/activate txlm

CONFIG=$1

python -u tools/sample_and_merge_langs.py --config configs/${CONFIG}
