import math
import os
import random
import sys

from pathlib import Path

input_file = sys.argv[1]
train = float(sys.argv[2])
dev = float(sys.argv[3])

assert (0 < train < 1)
assert (0 < dev < 1)
assert (0 < train + dev <= 1)

random.seed(1)

file_path = Path(input_file)
file_stem = file_path.stem
output_dir = file_path.parent

lines = [line.strip() for line in open(input_file, 'r') if line != '']
random.shuffle(lines)

train_length = math.ceil(len(lines) * train)
dev_length = math.ceil(len(lines) * dev)

train_lines = lines[:train_length]
dev_lines = lines[train_length:train_length+dev_length]
test_lines = lines[train_length+dev_length:] if train_length + dev_length < len(lines) else None

with open(os.path.join(output_dir, file_stem + '_train.txt'), 'w') as fout:
    for line in train_lines:
        print(line, file=fout)

with open(os.path.join(output_dir, file_stem + '_dev.txt'), 'w') as fout:
    for line in dev_lines:
        print(line, file=fout)

if test_lines:
    with open(os.path.join(output_dir, file_stem + '_test.txt'), 'w') as fout:
        for line in test_lines:
            print(line, file=fout)
