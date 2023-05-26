import argparse
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create version of text file without blank lines")
    parser.add_argument(
        '--input_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True
    )
    args = parser.parse_args()

    file_lines = [line.strip() for line in open(args.input_file, 'r') if line.strip() != '']
    file_lines = [re.sub(r' ([,.;:?!\)\]»“])', r'\1', line) for line in file_lines]
    file_lines = [re.sub(r'([\(\[«„]) ', r'\1', line) for line in file_lines]

    # Print merged lines to output file
    with open(args.output_file, 'w+') as fout:
        for line in file_lines:
            print(line, file=fout)

