import math
import os
import random
import sys

from pathlib import Path

input_file = sys.argv[1]
dev = int(sys.argv[2])
test = int(sys.argv[3])

random.seed(1)

file_path = Path(input_file)
file_stem = file_path.stem
output_dir = file_path.parent

lines = [line.strip() for line in open(input_file, 'r') if line != '']
original_length = len(lines)
random.shuffle(lines)

devtest_length = dev+test
devtest_lines = lines[-devtest_length:]
train_lines = lines[:-devtest_length]

assert len(train_lines) + len(devtest_lines) == original_length

dev_lines = devtest_lines[:dev]
test_lines = devtest_lines[dev:]
assert len(dev_lines) == dev
assert len(test_lines) == test

with open(os.path.join(output_dir, file_stem + '_train.txt'), 'w') as fout:
    for line in train_lines:
        print(line, file=fout)

with open(os.path.join(output_dir, file_stem + '_dev.txt'), 'w') as fout:
    for line in dev_lines:
        print(line, file=fout)

with open(os.path.join(output_dir, file_stem + '_test.txt'), 'w') as fout:
    for line in test_lines:
        print(line, file=fout)
