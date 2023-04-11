import argparse
import os
import random
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Break a large data file into a specified number of shards"
    )
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('num_shards', type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--shuffle_first', action='store_true')
    args = parser.parse_args()

    lines = [
        line.strip()
        for line in open(args.input_file, 'r') if line.strip() != ''
    ]
    num_lines = len(lines)
    if args.shuffle_first:
        random.seed(args.seed)
        random.shuffle(lines)


    quotient, remainder = divmod(num_lines, args.num_shards)
    lower_elements = [quotient for i in range(args.num_shards - remainder)]
    higher_elements = [quotient + 1 for i in range(remainder)]
    shard_lengths = lower_elements + higher_elements

    os.makedirs(args.output_dir, exist_ok=True)

    for shard_num, shard_length in zip(range(args.num_shards), shard_lengths):
        filestem = Path(args.input_file).stem
        filename = os.path.join(
            args.output_dir, f'{filestem}_shard{shard_num+1}.txt'
        )
        start_index = sum(shard_lengths[:shard_num])
        with open(filename, 'w') as fout:
            for line in lines[start_index:start_index+shard_length]:
                print(line, file=fout)
