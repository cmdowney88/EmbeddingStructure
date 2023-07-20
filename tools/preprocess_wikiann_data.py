import argparse
from collections import defaultdict
import os
import json
from sklearn.model_selection import train_test_split

tag2int = {'O'     : 0,
           'B-PER' : 1,
           'I-PER' : 2,
           'B-ORG' : 3,
           'I-ORG' : 4,
           'B-LOC' : 5,
           'I-LOC' : 6}

zero_shot_threshold = 1792
min_dev = 256
min_test = 512

dev_size = 0.05
test_size = 0.1

seed = 1

if __name__ == '__main__':
    #get list of input files
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()
    
    all_train = []
    all_dev = []
    
    #loop over input files
    for infile in os.listdir(args.input_folder):
        lang = infile[8:-4]
        
        #split text into groups of lines
        filepath = os.path.join(args.input_folder, infile)
        with open(filepath, 'r', encoding='utf-8') as reader:
            text = reader.read()
        line_groups = text.split('\n\n')
        for i in range(len(line_groups)):
            line_groups[i] = line_groups[i].split('\n')
            for j in range(len(line_groups[i])): #iterate over lines in line group
                line_groups[i][j] = line_groups[i][j].split()
        
        # remove lines without content
        line_groups = [x for x in line_groups if len(x[0]) > 1]
        
        #print number of line groups
        print(f'{lang}\t{len(line_groups)}')
        
        #empty list for data instances
        instances = []
        
        #loop over line groups
        for group in line_groups:
            #make dictionary of lists containing tokens, ner_tags, langs, spans
            instance = defaultdict(lambda: [])
            instance['tokens'] = [line[0] for line in group]
            instance['ner_tags'] = [tag2int[line[-1]] for line in group]
            instance['langs'] = [lang for x in range(len(group))]
            instance['spans'] = [line[-1].split('-')[-1]+': '+line[1] for line in group if line[-1][0] == 'B']
            
            #add dictionary to data instances list
            instances.append(instance)
            
        #train, dev, test split
        if len(instances) < zero_shot_threshold:
            #save data instances list to json
            outfile = f'{infile.split(".")[0]}_test.json'
            with open(os.path.join(args.output_folder, outfile), 'w', encoding='utf-8') as writer:
                json.dump(instances, writer, ensure_ascii=False)
        else:
            splits = dict()
            ts = max(test_size, min_test/len(instances))
            ds = max(dev_size, min_dev/len(instances))
            tr, splits['test'] = train_test_split(instances, test_size=ts, random_state=seed)
            splits['train'], splits['dev'] = train_test_split(tr, test_size=ds/(1 - ts))
            
            print(f'\ttrain\t{len(splits["train"])}\t{len(splits["train"])/len(instances)}')
            print(f'\tdev\t{len(splits["dev"])}\t{len(splits["dev"])/len(instances)}')
            print(f'\ttest\t{len(splits["test"])}\t{len(splits["test"])/len(instances)}')
            
            all_train.extend(splits['train'])
            all_dev.extend(splits['dev'])
            
            #save train, dev and test to separate json files
            for name in splits:
                outfile = f'{infile.split(".")[0]}_{name}.json'
                with open(os.path.join(args.output_folder, outfile), 'w', encoding='utf-8') as writer:
                    json.dump(splits[name], writer, ensure_ascii=False)
    
    print(f'\n{args.dataset_name}\ttrain\t{len(all_train)}')
    print(f'{args.dataset_name}\tdev\t{len(all_dev)}')
    
    all_tr_file = f'wikiann-{args.dataset_name}_train.json'
    with open(os.path.join(args.output_folder, all_tr_file), 'w', encoding='utf-8') as writer:
        json.dump(all_train, writer, ensure_ascii=False)
    all_dev_file = f'wikiann-{args.dataset_name}_dev.json'
    with open(os.path.join(args.output_folder, all_dev_file), 'w', encoding='utf-8') as writer:
        json.dump(all_dev, writer, ensure_ascii=False)