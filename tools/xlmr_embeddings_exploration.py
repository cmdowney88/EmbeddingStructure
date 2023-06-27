from collections import defaultdict
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def hex2dec(hex_str):
    """Convert Unicode hexadecimal string to base-10 int."""
    return int(hex_str, 16)

def get_ord2script(scriptfile):
    """Return dictionary (key: Unicode decimal, val: script of corresponding
    character according to Unicode documentation)"""
    with open(scriptfile, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        
    ord2script = dict()
    
    for line in lines:
        if line[0] != '#':
            items = line.split()
            if len(items) > 0:
                
                script = items[2]
                encoding = items[0]
                
                if '..' in encoding:
                    start_stop = encoding.split('..')
                    start = hex2dec(start_stop[0])
                    stop = hex2dec(start_stop[1])
                    
                    for dec_encoding in range(start, stop+1):
                        ord2script[dec_encoding] = script
                        
                else:
                    dec_encoding = hex2dec(encoding)
                    ord2script[dec_encoding] = script
                    
    return ord2script

def top_script(token, ord2script):
    """Return most-used script within token (str), using ord2script (dict)
    to retrieve the script of each char in token."""
    script_counts = defaultdict(lambda: 0)

    for character in token:
        try:
            script = ord2script[ord(character)]
        except KeyError:
            script = 'UNK'
        script_counts[script] += 1
    
    return max(script_counts, key=lambda x: script_counts[x])

def plot_pca(coords, index2label, label_offset=0.01, fontsize='x-small', mean_all=None):
    """Plot principal components. coords (numpy array of floats) gives x-coordinates
    in column 0 and y-coordinates in column 1. index2label (dict) has indices
    of coordinates as keys and name of corresponding label for plot as values."""
    sns.scatterplot(x=coords[:,0], y=coords[:,1])
    
    for i in range(len(index2label)):
        x_coord = coords[i,0] + label_offset
        y_coord = coords[i,1] + label_offset
        label = index2label[i]
        plt.text(x_coord, y_coord, label, fontsize=fontsize)
        
    if mean_all is not None:
        plt.text(mean_all[0,0], mean_all[0,1], 'X', color='red', fontweight='extra bold')
        
    plt.show()

if __name__ == '__main__':
    #get XLM-R embeddings
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
    embeddings = model.get_input_embeddings().weight
    #print(embeddings[0,0])
    
    #get mean and standard deviation of all embeddings
    std_and_mean = torch.std_mean(embeddings, dim=0)
    all_embeddings_std = std_and_mean[0]
    all_embeddings_mean = std_and_mean[1]
    
    #convert embedding IDs to tokens
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    id2token = tokenizer.convert_ids_to_tokens([x for x in range(len(embeddings))])
    
    #convert Unicode decimal to character's script
    ord2script = get_ord2script('unicode_scripts_for_embeddings_exploration.txt')
    
    #get script for each token in XLM-R's embeddings
    script2ids = defaultdict(lambda: []) #key: script, value: list of token indices
    for i in range(len(id2token)):
        token = id2token[i]
        token = token[1:] if token[0] == '▁' and len(token) > 1 else token
        script = top_script(token, ord2script)
        script2ids[script].append(i)
    
    print('Number of tokens in each script')
    for script in script2ids:
        print(f'{script} tokens:\t{len(script2ids[script])}')
        
    print(f'\nTokens with unknown script:\n{[id2token[x] for x in script2ids["UNK"]]}')
    
    #get mean and standard deviation of embeddings for each script
    index2script = list(script2ids.keys()) #locations of scripts in script_stds and script_means
    script_stds = list() #list of numpy arrays
    script_means = list() #list of numpy arrays
    for script in index2script:
        script_embed_list = [embeddings[x] for x in script2ids[script]]
        script_embeddings = torch.stack(script_embed_list, dim=0)
        std_and_mean = torch.std_mean(script_embeddings, dim=0)
        script_stds.append(std_and_mean[0].detach().numpy())
        script_means.append(std_and_mean[1].detach().numpy())
    script_stds = np.array(script_stds)
    script_means = np.array(script_means)
    
    #fit PCA to all embeddings
    rs = 1 #26062023
    pca = PCA(n_components=2, random_state=rs)
    pca.fit(embeddings.detach().numpy())
    
    #principal components for script mean embeddings
    reduced = pca.transform(script_means)
    
    #principal components for mean of all embeddings
    mean_all = pca.transform(all_embeddings_mean.detach().numpy().reshape(1, -1))
    print(mean_all)
    
    #plot mean principal components
    plot_pca(reduced, index2script, label_offset=.025)
    
    #plot cluster of scripts (exclude Cyrillic, Braille, Common, Latin)
    skip_scripts = ['Cyrillic', 'Braille', 'Common', 'Latin']
    skip_indices = [index2script.index(x) for x in skip_scripts]
    not_all_means_pc = np.delete(reduced, skip_indices, axis=0)
    not_all_scripts = [x for x in index2script if x not in skip_scripts]
    plot_pca(not_all_means_pc, not_all_scripts, label_offset=.01)
    
    #plot most frequent scripts
    most_common_scripts = [x for x in script2ids if len(script2ids[x]) > 4000]
    frequent_scripts = most_common_scripts + ['Katakana', 'Hiragana']
    frequent_script_indices = [index2script.index(x) for x in frequent_scripts]
    frequent_script_means_pc = reduced[frequent_script_indices]
    plot_pca(frequent_script_means_pc, frequent_scripts, label_offset=.025, 
             fontsize='small')