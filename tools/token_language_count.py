import argparse
import json
import numpy as np
import pandas as pd
from transformers import XLMRobertaTokenizer

def get_languages_index(lang2file):
    index2lang = list(lang2file.keys())
    lang2index = dict()
    for i in range(len(index2lang)):
        language = index2lang[i]
        lang2index[language] = i
        
    return index2lang, lang2index

def get_counts_array(tokenizer, index2lang, lang2file, max_lines):
    counts = np.zeros(shape=(len(tokenizer), len(index2lang)), dtype=int)
    for lang_index in range(len(index2lang)):
        language = index2lang[lang_index]
        input_file = lang2file[language]
        with open(input_file, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
        num_lines = min(len(lines), max_lines)
        for i in range(num_lines):
            line = lines[i]
            token_indices = tokenizer.encode(line)
            for token_index in token_indices:
                counts[token_index, lang_index] += 1
    return counts
            
def get_index2token(tokenizer):
    vocab = tokenizer.get_vocab()
    index2token = np.empty(shape=tokenizer.vocab_size, dtype=object)
    for token in vocab:
        index = vocab[token]
        index2token[index] = token
    return index2token

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Count tokens for each language in training files"
    )
    parser.add_argument('--language_files_json', type=str, required=True)
    parser.add_argument('--output_csv_file', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--max_vocab_lines', type=int, default=100000)
    args = parser.parse_args()
    
    with open(args.language_files_json, 'r') as reader:
        lang2file = json.load(reader)
    
    tokenizer = XLMRobertaTokenizer(vocab_file=args.vocab_file)
    index2token = get_index2token(tokenizer)
    
    index2lang, lang2index = get_languages_index(lang2file)
    
    counts = get_counts_array(tokenizer, index2lang, lang2file, args.max_vocab_lines)
            
    counts_df = pd.DataFrame(data=counts, index=index2token, columns=index2lang)
    counts_df.to_csv(args.output_csv_file)