#!/bin/sh
source ~/miniconda3/bin/activate txlm

python -u tools/make_shards.py $@
