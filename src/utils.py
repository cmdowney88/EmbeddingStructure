import os 
import torch

from transformers import (
    WEIGHTS_NAME,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    BertConfig,
    BertModel,
    BertForMaskedLM,
    BertTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaForMaskedLM,
    XLMRobertaTokenizerFast,
    T5Config,
    T5EncoderModel,
    T5TokenizerFast,

)

MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaTokenizerFast),
    "bert": (BertConfig, BertModel, BertForMaskedLM, BertTokenizerFast),
    'xlmr': (XLMRobertaConfig, XLMRobertaModel, XLMRobertaForMaskedLM, XLMRobertaTokenizerFast),
    't5': (T5Config, T5EncoderModel, None, T5TokenizerFast)
}

_model_names = ['bert-base-cased', 'bert-large-cased', 'roberta-base', 'roberta-large', 'xlm-roberta-base', 'xlm-roberta-large', 'bert-base-multilingual-cased', 't5-base']

_lang_choices = {
    'pos': ['af', 'ar', 'bg', 'ca', 'cs', 'cy', 'da', 'de', 'el',\
    'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'ga', 'gd', 'gl', 'fy',\
    'he', 'hi', 'hr', 'hu', 'hy', 'hyw', 'id', 'is', 'it', 'ja', 'ko',\
    'la', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl',\
    'sr', 'sv', 'ta', 'tr', 'uk', 'ur', 'vi', 'zh'],
    'ppl': ['en', 'de', 'fr', 'ru', 'es', 'it', 'ja', 'zh-cn', 'zh-tw',\
    'pl', 'uk', 'nl', 'sv', 'pt', 'sr', 'hu', 'ca', 'cs', 'fi', 'ar',\
    'ko', 'fa', 'no', 'vi', 'he', 'id', 'ro', 'tr', 'bg', 'et', 'ms',\
    'da', 'sk', 'hr', 'el', 'lt', 'sl', 'th', 'hi', 'lv', 'tl'],
    'bpc': ['en', 'de', 'fr', 'ru', 'es', 'it', 'ja', 'zh-cn', 'zh-tw',\
    'pl', 'uk', 'nl', 'sv', 'pt', 'sr', 'hu', 'ca', 'cs', 'fi', 'ar',\
    'ko', 'fa', 'no', 'vi', 'he', 'id', 'ro', 'tr', 'bg', 'et', 'ms',\
    'da', 'sk', 'hr', 'el', 'lt', 'sl', 'th', 'hi', 'lv', 'tl'],
}

_task_choices = ['pos', 'ppl']

_label_spaces = {
    #pos documentation: https://universaldependencies.org/u/pos/index.html
    'UPOS':['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', \
        'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',\
        'X'],
    }

#loads tokenizer and model based on given model type and name
def load_hf_model(model_type, model_name, task='ppl', random_weights=False):
    config_class, model_class, lm_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(
        model_name,
        cache_dir=None,
    )

    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        do_lower_case=False,
        cache_dir=None,
    )

    if task == 'ppl': model_class = lm_class
    
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
            if line.startswith('#'): continue
            if len(line) == 0:
                dataset.append((example_sent, example_labels))
                example_sent = []
                example_labels = []
            else:
                idx, word, lemma, upos, xpos, morph_feats, head, dep_rel, deps, misc = line.split('\t')
                #skip mulitpart phrases since each part is additionally
                # annotated (weird for tokenization but ¯\_(ツ)_/¯)
                if '-' in idx: continue
                if idx == 1: assert len(example) == 0

                #using upos for part of speech task instead of xpos
                if task == 'pos': label = upos

                example_sent.append(word)
                example_labels.append(label)

    if len(example_sent) > 0: dataset.append((example_sent, example_labels))
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

def load_ud_datasets(data_path, lang, task='pos'):
    ud_files = os.listdir(data_path)
    train_file = [u for u in ud_files if u.startswith(lang+'_') and '-train' in u]
    valid_file = [u for u in ud_files if u.startswith(lang+'_') and '-dev' in u]
    test_file = [u for u in ud_files if u.startswith(lang+'_') and '-test' in u]
    assert len(train_file) == 1 and len(valid_file) == 1 and len(test_file) == 1
    
    train_file = train_file[0]
    valid_file = valid_file[0]
    test_file = test_file[0]

    train_path = os.path.join(data_path, train_file)
    valid_path = os.path.join(data_path, valid_file)
    test_path = os.path.join(data_path, test_file)

    if task == 'ppl':
        train_data = _load_ud_text(train_path)
        valid_data = _load_ud_text(valid_path)
        test_data = _load_ud_text(test_path)
    else:
        train_data = _load_word_level_ud(train_path, task)
        valid_data = _load_word_level_ud(valid_path, task)
        test_data = _load_word_level_ud(test_path, task)

    return train_data, valid_data, test_data

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