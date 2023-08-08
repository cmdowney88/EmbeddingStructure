import json
import os
import torch

from transformers import (
    WEIGHTS_NAME, RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaTokenizerFast, BertConfig,
    BertModel, BertForMaskedLM, BertTokenizerFast, XLMRobertaConfig, XLMRobertaModel,
    XLMRobertaForMaskedLM, XLMRobertaTokenizerFast, XLMRobertaTokenizer
)

MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaTokenizerFast),
    "bert": (BertConfig, BertModel, BertForMaskedLM, BertTokenizerFast),
    'xlmr': (XLMRobertaConfig, XLMRobertaModel, XLMRobertaForMaskedLM, XLMRobertaTokenizerFast)
}

_model_names = [
    'bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large', 'xlm-roberta-base',
    'xlm-roberta-large', 'bert-base-multilingual-cased'
]

_lang_choices = {
    'pos': ['et', 'fi', 'hu', 'no', 'ru', 'kpv', 'krl', 'mdf', 'myv', 'olo', 'sme', 'sms'],
    'ppl': ['en', 'de', 'fr', 'ru', 'es', 'it', 'ja', 'zh-cn', 'zh-tw',\
    'pl', 'uk', 'nl', 'sv', 'pt', 'sr', 'hu', 'ca', 'cs', 'fi', 'ar',\
    'ko', 'fa', 'no', 'vi', 'he', 'id', 'ro', 'tr', 'bg', 'et', 'ms',\
    'da', 'sk', 'hr', 'el', 'lt', 'sl', 'th', 'hi', 'lv', 'tl'],
    'bpc': ['en', 'de', 'fr', 'ru', 'es', 'it', 'ja', 'zh-cn', 'zh-tw',\
    'pl', 'uk', 'nl', 'sv', 'pt', 'sr', 'hu', 'ca', 'cs', 'fi', 'ar',\
    'ko', 'fa', 'no', 'vi', 'he', 'id', 'ro', 'tr', 'bg', 'et', 'ms',\
    'da', 'sk', 'hr', 'el', 'lt', 'sl', 'th', 'hi', 'lv', 'tl'],
    'ner': ['et', 'fi', 'fiu-vro', 'hu', 'koi', 'komi', 'kv', 'mari', 'mdf', 
            'mhr', 'mrj', 'myv', 'no', 'ru', 'se', 'udm', 'vep']
}

_task_choices = ['pos', 'ppl', 'ner']

_label_spaces = {
    #pos documentation: https://universaldependencies.org/u/pos/index.html
    'UPOS': ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', \
        'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',\
        'X'],
    #ner documentation: https://huggingface.co/datasets/wikiann
    'NER': [x for x in range(7)]
    }

_xlmr_special_tokens = ['<s>', '</s>', '<unk>', '<pad>', '<mask>']


#loads tokenizer and model based on given model type and name
def load_hf_model(model_type, model_name, task='ppl', random_weights=False, tokenizer_path=None):
    config_class, model_class, lm_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(
        model_name,
        cache_dir=None,
    )

    if tokenizer_path in ['xlm-roberta-base', 'xlm-roberta-large']:
        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_path,
            do_lower_case=False,
            cache_dir=None,
        )
    elif tokenizer_path:
        tokenizer = XLMRobertaTokenizerFast(vocab_file=tokenizer_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(
            model_name,
            do_lower_case=False,
            cache_dir=None,
        )

    if task == 'ppl':
        model_class = lm_class

    if random_weights:
        model = model_class(config=config)
    else:
        #tokenizer.max_len_single_sentence will give max length of tokenizer
        model = model_class.from_pretrained(
            model_name,
            config=config,
            cache_dir=None,
        )
    return model, tokenizer


def _load_word_level_ud(file_path, task):
    dataset = []

    example_sent = []
    example_labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if len(line) == 0:
                # Make sure not to append blank example after more than one blank line
                if len(example_sent) > 0:
                    dataset.append((example_sent, example_labels))
                example_sent = []
                example_labels = []
            else:
                idx, word, lemma, upos, xpos, morph_feats, head, dep_rel, deps, misc = line.split('\t')
                #skip mulitpart phrases since each part is additionally
                # annotated (weird for tokenization but ¯\_(ツ)_/¯)
                if '-' in idx:
                    continue
                # if idx == 1: assert len(example) == 0

                #using upos for part of speech task instead of xpos
                if task == 'pos':
                    label = upos

                example_sent.append(word)
                example_labels.append(label)

    if len(example_sent) > 0:
        dataset.append((example_sent, example_labels))
    return dataset


def _load_ud_text(file_path):
    dataset = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# text = '):
                line = line.replace('# text = ', '').strip()
                dataset.append(line)
            else:
                continue

    return dataset


def load_ud_splits(data_path, lang, splits=['train', 'dev', 'test'], task='pos'):
    ud_files = os.listdir(data_path)
    split_data = {}
    for split_name in splits:
        split_file = [u for u in ud_files if u.startswith(lang + '_') and f'-{split_name}' in u]
        assert len(split_file) == 1
        split_file = split_file[0]
        split_path = os.path.join(data_path, split_file)
        if task == 'ppl':
            split_data[split_name] = _load_ud_text(split_path)
        else:
            split_data[split_name] = _load_word_level_ud(split_path, task)

    return split_data if len(splits) > 1 else split_data[splits[0]]

def _load_ner_data(file_path):
    # Return tokens and labels
    dataset = []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for instance in data:
        if len(instance['tokens']) > 0:
            dataset.append((instance['tokens'], instance['ner_tags']))
        
    return dataset


def load_ner_splits(data_path, lang, splits=['train', 'dev', 'test']):
    ner_files = os.listdir(data_path)
    split_data = {}
    for split_name in splits:
        split_file = [x for x in ner_files if f'_{lang}_' in x and f'_{split_name}' in x]
        assert len(split_file) == 1
        split_file = split_file[0]
        split_path = os.path.join(data_path, split_file)
        
        split_data[split_name] = _load_ner_data(split_path)

    return split_data if len(splits) > 1 else split_data[splits[0]]


# averaging outputs for subword units
def _consolidate_features(features, alignment):
    consolidated_features = []
    for a in alignment:
        c = features[a]
        c = torch.mean(c, dim=0).unsqueeze(0)
        assert c.shape[0] == 1
        consolidated_features.append(c)
    return consolidated_features


#EOF
