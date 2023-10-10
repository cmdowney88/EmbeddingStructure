#!/bin/sh
PATH_TO_CONDA=$1
VENVNAME=$2
CONFIG=$3

source ${PATH_TO_CONDA}/miniconda3/bin/activate $VENVNAME

python -u tools/sample_and_merge_langs.py --config configs/${CONFIG}
