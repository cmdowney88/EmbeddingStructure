#!/bin/sh
source ~/miniconda3/bin/activate txlm

LANGUAGE=$1
HF_AUTH_TOKEN=$2
OUTPUT_FILE=$3

python -u tools/download_oscar.py $LANGUAGE $HF_AUTH_TOKEN $OUTPUT_FILE

