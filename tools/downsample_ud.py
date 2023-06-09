import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reduce the size of a Universal Dependencies CONNLU file"
    )
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--output_size', type=int, required=True)
    parser.add_argument('--just_print_length', action='store_true')
    args = parser.parse_args()

    random.seed(1)

    file_lines = [line.strip() for line in open(args.input_file, 'r')]

    sentences = []
    current_sentence = []
    for line in file_lines:
        if line == '' and len(current_sentence) > 0:
            sentences.append(current_sentence)
            current_sentence = []
        elif line == '':
            continue
        else:
            current_sentence.append(line)

    if args.just_print_length:
        print(len(sentences))
        exit(0)

    random.shuffle(sentences)

    with open(args.output_file, 'w+') as fout:
        for sentence in sentences[:args.output_size]:
            for line in sentence:
                print(line, file=fout)
            print('', file=fout)
