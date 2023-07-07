#!/bin/sh

source ~/miniconda3/bin/activate txlm

python -u src/mlm.py --config $1
