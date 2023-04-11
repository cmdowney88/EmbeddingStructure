import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge text files, removing duplicate and blank lines"
    )
    parser.add_argument('--input_files', nargs='*', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    # Read in lines from each input file in turn, discarding empty lines
    all_lines = []
    for input_file in args.input_files:
        file_lines = [
            line.strip() for line in open(input_file, 'r') if line.strip() != ''
        ]
        all_lines += file_lines

    # Remove duplicate lines by converting to a set and back
    all_lines = list(set(all_lines))

    # Print merged lines to output file
    with open(args.output_file, 'w+') as fout:
        for line in all_lines:
            print(line, file=fout)
