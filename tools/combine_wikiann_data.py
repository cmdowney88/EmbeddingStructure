import argparse
import json
import os
from sklearn.model_selection import train_test_split

zero_shot_threshold = 1792
min_dev = 256
min_test = 512

dev_size = 0.05
test_size = 0.1

seed = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--langs', nargs='+', type=str)
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()
    
    if args.output_folder is None:
        args.output_folder = args.input_folder
    
    # get file list
    files = os.listdir(args.input_folder)
    lang_files = [file for file in files for lang in args.langs if f'_{lang}_' in file]
    
    # get data from files
    data = []
    for file in lang_files:
        file_path = os.path.join(args.input_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    
    # train, dev, test split
    if len(data) < zero_shot_threshold:
        # save data instances list to json
        outfile = f'wikiann_{args.dataset_name}_test.json'
        with open(os.path.join(args.output_folder, outfile), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    else:
        splits = dict()
        ts = max(test_size, min_test/len(data))
        ds = max(dev_size, min_dev/len(data))
        tr, splits['test'] = train_test_split(data, test_size=ts, random_state=seed)
        splits['train'], splits['dev'] = train_test_split(tr, test_size=ds/(1 - ts), random_state=seed)
        
        print(f'\ttrain\t{len(splits["train"])}\t{len(splits["train"])/len(data)}')
        print(f'\tdev\t{len(splits["dev"])}\t{len(splits["dev"])/len(data)}')
        print(f'\ttest\t{len(splits["test"])}\t{len(splits["test"])/len(data)}')
        
        # save train, dev and test to separate json files
        for name in splits:
            outfile = f'wikiann_{args.dataset_name}_{name}.json'
            with open(os.path.join(args.output_folder, outfile), 'w', encoding='utf-8') as writer:
                json.dump(splits[name], writer, ensure_ascii=False)