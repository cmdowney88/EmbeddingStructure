import argparse

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

    # Print merged lines to output file
    with open(args.output_file, 'w+') as fout:
        for line in file_lines:
            print(line, file=fout)

    
