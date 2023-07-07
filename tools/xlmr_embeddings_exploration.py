from collections import defaultdict
from matplotlib import colormaps
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForMaskedLM, XLMRobertaTokenizer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
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

def top_script(token, ord2script, preferred_scripts=None):
    """Return most-used script within token (str), using ord2script (dict)
    to retrieve the script of each char in token. If there are multiple options
    for the script with the most characters within token, scripts in 
    preferred_scripts (iterable) will be chosen over other options."""
    script_counts = defaultdict(lambda: 0)

    for character in token:
        try:
            script = ord2script[ord(character)]
        except KeyError:
            script = 'UNK'
        script_counts[script] += 1
        
    if preferred_scripts == None:
        return max(script_counts, key=lambda x: script_counts[x])
    
    return max(script_counts, key=lambda x: (script_counts[x], x in preferred_scripts))

def get_script2ids(id2token, ord2script):
    """Return dict with scripts as keys and lists of token IDs as values. 
    id2token (dict) IDs as keys and tokens as values. ord2script (dict) has 
    Unicode decimals as keys and scripts of corresponding characters as values."""
    script2ids = defaultdict(lambda: []) # key: script, value: list of token indices
    for i in range(len(id2token)):
        token = id2token[i]
        token = token[1:] if token[0] == '▁' and len(token) > 1 else token
        script = top_script(token, ord2script, preferred_scripts={'Latin', 'Cyrillic'})
        script2ids[script].append(i)
    return script2ids

def display_script_tokens(script2ids, id2token, embeddings_name):
    """Display number of tokens for each script. Also display all tokens with unknown
    script."""
    print(f'{embeddings_name}: number of tokens in each script')
    for script in script2ids:
        print(f'\t{script} tokens:\t{len(script2ids[script])}')
        
    print(f'\n{embeddings_name} tokens with unknown script:\n{[id2token[x] for x in script2ids["UNK"]]}')
    
def get_script_stds_means(index2script, script2ids, embeddings): 
    """Return numpy array of standard deviations for each script and numpy array 
    of means for each script."""
    script_stds = list() # list of numpy arrays
    script_means = list() # list of numpy arrays
    for script in index2script:
        script_embed_list = [embeddings[x] for x in script2ids[script]]
        script_embeddings = torch.stack(script_embed_list, dim=0)
        std_and_mean = torch.std_mean(script_embeddings, dim=0)
        script_stds.append(std_and_mean[0].detach().numpy())
        script_means.append(std_and_mean[1].detach().numpy())
    return np.array(script_stds), np.array(script_means)

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
    
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Source: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    #  Using a special case to obtain the eigenvalues of this
    #  two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    #  Calculating the standard deviation of x from
    #  the squareroot of the variance and multiplying
    #  with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    #  calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
def plot_pca_ellipses(pc_all, scripts, script2ids, colormap='tab10', means=None,
                      mean_color='red'):
    """Plot confidence ellipses for principal components of embeddings for each
    script in scripts (list). pc_all is a 2D array of principal components of
    embeddings. script2ids (dict) has scripts from the scripts list as keys and 
    corresponding indices in pc_all as values."""
    fig, ax = plt.subplots()
    colors = colormaps[colormap].colors
    for i in range(len(scripts)):
        script = scripts[i]
        color = colors[i]
        if len(script2ids[script]) > 2:
            script_pc = pc_all[script2ids[script]]
            x = script_pc[:,0]
            y = script_pc[:,1]
            ax.scatter(x, y, s=0.5, color=color, label=script)
            ax.axvline(c='grey', lw=1)
            ax.axhline(c='grey', lw=1)
            confidence_ellipse(x, y, ax, edgecolor=color)
            if means != None:
                ax.scatter(means[i][0], means[i][1], color=mean_color)
    plt.legend(loc='upper right')
    plt.show()
    
def plot_word_position_scatters(id2token, pc_all, index2script, script2ids):
    """Make scatter plots of embedding principal components with hue determined
    by whether tokens are word-initial or word-medial. One scatter plot shows
    results for all scripts. There is also a separate scatter plot for each script."""
    # plot all scripts
    word_position = np.array(['Initial' if x[0] == '▁' else 'Medial' for x in id2token])
    sns.scatterplot(x=pc_all[:,0], y=pc_all[:,1], hue=word_position).set_title('All scripts')
    plt.show()
    
    # make plots for each script
    for script in index2script:
        script_ids = script2ids[script]
        pc_script = pc_all[script_ids]
        script_underscores = word_position[script_ids]
        sns.scatterplot(x=pc_script[:,0], y=pc_script[:,1], hue=script_underscores).set_title(script)
        plt.show()
    
def plot_base_embeddings(embeddings, ord2script, pca):
    """Plot principal components of XLM-R base embeddings."""
    
    # get id2token for base embeddings
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    id2token = tokenizer.convert_ids_to_tokens([x for x in range(len(embeddings))])
    
    # get script for each token in embeddings
    script2ids = get_script2ids(id2token, ord2script)
    display_script_tokens(script2ids, id2token, 'base')
    
    # get mean and standard deviation of all embeddings
    std_and_mean = torch.std_mean(embeddings, dim=0)
    all_embeddings_mean = std_and_mean[1]
    
    # get mean and standard deviation for each script
    index2script = [x for x in script2ids.keys() if len(script2ids[x]) > 0]
    script_stds, script_means = get_script_stds_means(index2script, script2ids, embeddings)
    
    # principal components for script mean embeddings
    reduced = pca.transform(script_means)
    
    # principal components for mean of all embeddings
    mean_all = pca.transform(all_embeddings_mean.detach().numpy().reshape(1, -1))
    print(f'Principal components of mean of base embeddings: {mean_all}')
    
    # plot mean principal components
    plot_pca(reduced, index2script, label_offset=.025)
    
    # plot mean principal components for cluster of scripts (exclude Cyrillic, Braille, Common, Latin)
    skip_scripts = [x for x in ['Cyrillic', 'Braille', 'Common', 'Latin'] if x in index2script]
    skip_indices = [index2script.index(x) for x in skip_scripts]
    not_all_means_pc = np.delete(reduced, skip_indices, axis=0)
    not_all_scripts = [x for x in index2script if x not in skip_scripts]
    plot_pca(not_all_means_pc, not_all_scripts, label_offset=.01)
    
    # plot mean principal components for most frequent scripts
    most_common_scripts = [x for x in index2script if len(script2ids[x]) > 4000]
    frequent_scripts = most_common_scripts + [x for x in ['Katakana', 'Hiragana'] if x in index2script]
    frequent_script_indices = [index2script.index(x) for x in frequent_scripts]
    frequent_script_means_pc = reduced[frequent_script_indices]
    plot_pca(frequent_script_means_pc, frequent_scripts, label_offset=.025, 
             fontsize='small')
    
    # plot confidence ellipses for most frequent scripts
    pc_all = pca.transform(embeddings.detach().numpy())
    plot_pca_ellipses(pc_all, frequent_scripts, script2ids, colormap='tab20')
    
    # plot confidence ellipses for expected Uralic scripts
    plot_pca_ellipses(pc_all, ['Common', 'Latin', 'Cyrillic'], script2ids)
    
    # plot confidence ellipses for scripts used for same language or groups of closely-related languages
    dravidian_scripts = ['Kannada', 'Malayalam', 'Tamil', 'Telugu']
    indoaryan_scripts = ['Bengali', 'Devanagari', 'Gujarati', 'Gurmukhi', 
                         'Oriya', 'Sinhala']
    semitic_scripts = ['Arabic', 'Hebrew', 'Syriac']
    taikaidai_scripts = ['Lao', 'Thai']
    japanese_scripts = ['Hiragana', 'Katakana']
    chinese_scripts = ['Bopomofo', 'Han']
    for scripts in (dravidian_scripts, indoaryan_scripts, semitic_scripts, 
                    taikaidai_scripts, japanese_scripts, chinese_scripts):
        plot_pca_ellipses(pc_all, scripts, script2ids)
    
    # plot embedding principal components with hue determined by presence/absence
    # of a leading underscore
    plot_word_position_scatters(id2token, pc_all, index2script, script2ids)
    
def plot_new_embeddings(model_path, tokenizer_path, pca, embeddings_name): 
    """Plot principal components of embeddings using model at model_path and 
    tokenizer at tokenizer_path."""
    
    # get embeddings
    model = AutoModelForMaskedLM.from_pretrained(model_path) 
    embeddings = model.get_input_embeddings().weight
    
    # get id2token for embeddings
    tokenizer = XLMRobertaTokenizer(vocab_file=tokenizer_path)
    id2token = tokenizer.convert_ids_to_tokens([x for x in range(len(embeddings))])
    
    # get script for each token in embeddings
    script2ids = get_script2ids(id2token, ord2script)
    display_script_tokens(script2ids, id2token, embeddings_name)
    
    # get mean and standard deviation of all embeddings
    std_and_mean = torch.std_mean(embeddings, dim=0)
    all_embeddings_mean = std_and_mean[1]
    
    # get mean and standard deviation for each script
    index2script = [x for x in script2ids.keys() if len(script2ids[x]) > 0]
    script_stds, script_means = get_script_stds_means(index2script, script2ids, embeddings)
    
    # principal components for script mean embeddings
    reduced = pca.transform(script_means)
    
    # principal components for mean of all embeddings
    mean_all = pca.transform(all_embeddings_mean.detach().numpy().reshape(1, -1))
    print(f'Principal components of mean of {embeddings_name} embeddings: {mean_all}')
    
    # plot mean principal components
    plot_pca(reduced, index2script, label_offset=.025)
    
    # plot standard deviation ellipses for scripts with more than 2 tokens
    pc_all = pca.transform(embeddings.detach().numpy())
    ellipse_scripts = [x for x in script2ids if len(script2ids[x]) > 2]
    plot_pca_ellipses(pc_all, ellipse_scripts, script2ids)
    
    # plot embedding principal components with hue determined by presence/absence
    # of a leading underscore
    plot_word_position_scatters(id2token, pc_all, index2script, script2ids)
    
if __name__ == '__main__':
    # convert Unicode decimal to character's script
    ord2script = get_ord2script('unicode_scripts_for_embeddings_exploration.txt')
    
    # get base XLM-R embeddings
    base_model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
    base_embeddings = base_model.get_input_embeddings().weight
    
    # fit PCA to base embeddings
    pca = PCA(n_components=2, random_state=1)
    pca.fit(base_embeddings.detach().numpy())
    
    plot_base_embeddings(base_embeddings, ord2script, pca)
    
    uralic_model_path = './models/checkpoint-200000' 
    uralic_tokenizer_path = './tokenizers/uralic_spm_32k.model'
    
    plot_new_embeddings(uralic_model_path, uralic_tokenizer_path, pca, 'Uralic')