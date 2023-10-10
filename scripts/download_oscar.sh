#!/bin/sh
PATH_TO_CONDA=$1
VENVNAME=$2
LANGUAGE=$3
HF_AUTH_TOKEN=$4
OUTPUT_FILE=$5

source ${PATH_TO_CONDA}/miniconda3/bin/activate $VENVNAME

python -u tools/download_oscar.py $LANGUAGE $HF_AUTH_TOKEN $OUTPUT_FILE

