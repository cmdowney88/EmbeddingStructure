import argparse
import torch
from torch.nn import functional as F
import os
import random
import math
import numpy as np
from tqdm import tqdm

# :(((
import tensorflow_datasets as tfds

from utils import MODEL_CLASSES, _lang_choices, _task_choices,\
     _model_names, _label_spaces, load_hf_model, load_ud_datasets,\
     _consolidate_features

class Probe(torch.nn.Module):
    def __init__(self, input_dim, output_dim, probe_type='linear'):
        super(Probe, self).__init__()

        self.hidden = None
        self.proj = torch.nn.Linear(input_dim, output_dim)

        if probe_type == 'nonlinear':
            self.hidden = torch.nn.Linear(input_dim, input_dim)

    def forward(self, input_ids):
        if self.hidden:
            input_ids = torch.nn.ReLU(self.hidden(input_ids))
        output = self.proj(input_ids)
        output = F.softmax(output, dim=-1)
        return output

def preprocess_ud_word_level(model, tokenizer, dataset, label_space, layer_id=-1):
    processed_dataset = []
    
    for sentence, labels in tqdm(dataset):
        #tokenize
        #do this manually word by word because huggingface 
        #can't get word alignments to track correctly
        #https://github.com/huggingface/transformers/issues/9637
        tokens = []
        alignment = []
        alignment_id = 0 #no offset since we add cls later
        for word in sentence:
            word_ids = tokenizer.encode(' '+word, add_special_tokens=False)
            tokens.extend(word_ids)
            word_alignments = [x for x in range(alignment_id, alignment_id+len(word_ids))]
            alignment_id = alignment_id+len(word_ids)
            alignment.append(word_alignments)

        #handling overflow
        feats = []
        for i in range(0, len(tokens), 510):
            token_ex = tokens[i:i+510]
            if tokenizer.cls_token_id != None:
                token_ex = [tokenizer.cls_token_id]+token_ex+[tokenizer.sep_token_id]
            else:
                token_ex = token_ex+[tokenizer.eos_token_id]
            input_ids = torch.LongTensor(token_ex).to('cuda:0')
            #run through model
            with torch.no_grad():
                model.eval()
                output = model(input_ids.unsqueeze(0), output_hidden_states=True, return_dict=True)

                feats_ex = output['hidden_states'][layer_id].squeeze().to('cpu')
                #removing cls and sep tokens
                if tokenizer.cls_token_id != None: feats_ex = feats_ex[1:feats_ex.shape[0]-1]
                else: feats_ex = feats_ex[:feats_ex.shape[0]-1]

                feats.append(feats_ex)
        feats = torch.cat(feats, dim=0)

        #align xlmr outputs to words
        feats = _consolidate_features(feats, alignment)
        assert len(feats) == len(labels)

        examples = list(zip(feats, labels))

        #process labels into tensor
        #'_' means an unlabeled word in UD, so skip these
        examples = [(f, torch.LongTensor([label_space.index(l)])) if l in label_space else (f, torch.LongTensor([-100]))  for f, l in examples]
        #examples = [(f, torch.LongTensor([label_space.index(l)])) for f, l in examples]
        processed_dataset.extend(examples)

    return processed_dataset

def train_probe(model, data, valid_data, criterion, optimizer, bsz=1, epochs=1, patience=-1, best_ckpt_path='./model.ckpt'):
    model.train()
    model.to('cuda:0')

    max_valid_acc = 0
    patience_step = 0
    #training epochs
    for epoch_id in range(0, epochs):
        random.shuffle(data)

        train_loss = 0.

        #go over training data
        for i in range(0, len(data), bsz):
            if i+bsz > len(data): batch = data[i:]
            else: batch = data[i:i+bsz]

            input_ids, labels = zip(*batch)
            input_ids = torch.cat(input_ids, dim=0).to('cuda:0')
            labels = torch.cat(labels, dim=0).to('cuda:0')

            optimizer.zero_grad()
            output = model.forward(input_ids)

            loss = criterion(output, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        valid_acc = evaluate_probe(model, valid_data, bsz=bsz)

        #use train loss to see if we need to stop
        if max_valid_acc > valid_acc:
            if patience_step == patience: break 
            else: patience_step += 1
        else:
            max_valid_acc = valid_acc
            #checkpoint model
            torch.save(model.state_dict(), best_ckpt_path)
            patience_step = 0

    return model

def evaluate_probe(model, data, bsz=1):
    model.eval()
    model.to('cuda:0')

    num_correct = 0.
    total_num = len(data)

    for i in range(0, len(data), bsz):
        if i+bsz > len(data): batch = data[i:]
        else: batch = data[i:i+bsz]

        input_ids, labels = zip(*batch)
        input_ids = torch.cat(input_ids, dim=0).to('cuda:0')
        labels = torch.cat(labels, dim=0).to('cuda:0')

        with torch.no_grad():   
            output = model.forward(input_ids)
        _, preds = torch.topk(output, k=1, dim=-1)
        batch_correct = torch.eq(labels.squeeze(), preds.squeeze()).sum().item()
        num_correct += batch_correct
        
    return num_correct/total_num

#task expects loose .conllu files for train an valid for the given language in the data_path dir
def pos(model, tokenizer, data_path, ckpt_path, lang='en'):

    #load UD train and eval data for probing on given langauge
    train_text_data, valid_text_data, test_text_data = load_ud_datasets(data_path, lang, task='pos')

    #DEBUGGING
    #train_data = train_data[:1000]
    pos_labels = _label_spaces['UPOS']

    #get majority sense baseline
    label_counts = []
    words = {}
    for _ in range(len(pos_labels)): label_counts.append(0.)
    for sent, labels in train_text_data:
        for w, l in zip(sent, labels):
            if l == '_': continue
            label_counts[pos_labels.index(l)] += 1
            if w not in words: words[w] = [0. for _ in range(len(pos_labels))]
            words[w][pos_labels.index(l)] += 1

    print('majority label baseline: {}'.format((max(label_counts)/sum(label_counts))*100))

    for w in words: words[w] = pos_labels[np.argmax(words[w])]
    per_word_maj_correct = 0.
    per_word_maj_total = 0.
    for sent, labels in test_text_data:
        for w, l in zip(sent, labels):
            if w in words and words[w] == l:
                per_word_maj_correct += 1
            per_word_maj_total += 1
    print('per word majority baseline: {}'.format((per_word_maj_correct/per_word_maj_total)*100))


    #preprocessing can take in a hidden layer id if we want to probe inside model
    train_data = preprocess_ud_word_level(model, tokenizer, train_text_data, pos_labels)
    valid_data = preprocess_ud_word_level(model, tokenizer, valid_text_data, pos_labels)
    test_data = preprocess_ud_word_level(model, tokenizer, test_text_data, pos_labels)

    random_seeds = [1,2,3,4,5]
    scores = []
    for rand_x in random_seeds:
        #set random seeds for model init, data shuffling
        torch.cuda.manual_seed(rand_x)
        torch.cuda.manual_seed_all(rand_x)   
        np.random.seed(rand_x)
        random.seed(rand_x)

        #create probe model
        feat_dim = train_data[0][0].shape[-1]
        probe = Probe(feat_dim, len(pos_labels))

        #load criterion and optimizer
        probe_lr = 0.001 #lr from Liu et al., 2019
        probe_bsz = 256

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(probe.parameters(), lr=probe_lr)

        #train probe model
        probe = train_probe(probe, train_data, valid_data, criterion, optimizer, bsz=probe_bsz, epochs=50, patience=3, best_ckpt_path=ckpt_path)
        
        #load best checkpoint for model state
        probe.load_state_dict(torch.load(ckpt_path))

        #evaluate on probe model
        acc = evaluate_probe(probe, test_data, bsz=probe_bsz)
        acc = acc*100
        scores.append(acc)
        print(acc)

        #reinitalize the model for each trial if using randomly initialized encoder
        if args.rand_weights:
            model, tokenizer = load_hf_model(args.model_class, args.model_name, task=args.task, random_weights=args.rand_weights)
            model.cuda()
            model.eval()

            train_data = preprocess_ud_word_level(model, tokenizer, train_text_data, pos_labels)
            valid_data = preprocess_ud_word_level(model, tokenizer, valid_text_data, pos_labels)
            test_data = preprocess_ud_word_level(model, tokenizer, test_text_data, pos_labels)

    mean = sum(scores)/len(scores)
    print('mean score = {}'.format(mean))
    var = sum([(s-mean)**2 for s in scores])/len(scores)
    #print('score variance = {}'.format(var))
    std_dev = math.sqrt(var)
    print('standard deviation = {}'.format(std_dev))
    #std_err = math.sqrt(var)/math.sqrt(len(scores))
    #print('standard error = {}'.format(std_err))

LABEL_PAD_IDX = -100
#not first subword == ## (bert, mbert); 
def _mask_inputs(input_ids, tokenizer, word_ids, mlm_probability=0.15):
    mask_idx = tokenizer.mask_token_id
    special_token_ids = [tokenizer.pad_token_id, mask_idx, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    vocab_size = len(tokenizer)

    tensor_shape = input_ids.shape
    labels = torch.empty(tensor_shape, dtype=torch.long).fill_(LABEL_PAD_IDX)
    masked_inputs = torch.empty(tensor_shape, dtype=torch.long).fill_(tokenizer.pad_token_id)

    i = 0
    masked_chars_count = 0
    curr_id = -1
    prob=0.9 #set to not mask until we see a word
    for j in range(0, tensor_shape[1]):
        original_idx = input_ids[i,j].item()
        #doing word piece masking istead of whole word masking
        if word_ids[j] != None and word_ids[j] > curr_id: 
            prob = random.uniform(0, 1)
            curr_id = word_ids[j]
        if original_idx in special_token_ids:
            prob=0.9 #do NOT mask special tokens
        #mask out 80% of the time
        if prob <= (mlm_probability*0.8):
            masked_inputs[i,j] = mask_idx
            labels[i,j] = original_idx
            #tracking for bpc
            token = tokenizer.decode([original_idx])
            if token.startswith('##'): token = token.replace('##', '', 1)
            masked_chars_count += len(token)
        #replace with random word piece 10% (and 10% of time, do nothing)
        elif prob <= (mlm_probability*0.9):
            masked_inputs[i,j] = random.randrange(0, len(tokenizer))
            labels[i,j] = original_idx
            #tracking for bpc
            token = tokenizer.decode([original_idx])
            if token.startswith('##'): token = token.replace('##', '', 1)
            masked_chars_count += len(token)
        #for not masked words
        else:
            masked_inputs[i,j] = original_idx

    return masked_inputs, labels, masked_chars_count

def _calculate_ppl(model, tokenizer, data):

    num_tokens = 0
    num_chars = 0
    total_loss = 0.

    for example in tqdm(data):
        #extract input
        texts = example['text'].numpy().decode("utf-8").split('\n')

        for text in texts:
            text = text.strip()
            input_dict = tokenizer(text, truncation=True)
            

            word_ids = [input_dict.token_to_word(i) for i in range(0, len(input_dict['input_ids']))]

            input_ids = torch.LongTensor(input_dict['input_ids']).unsqueeze(dim=0)
            attention_mask = torch.LongTensor(input_dict['attention_mask']).unsqueeze(dim=0)
            input_ids, labels, masked_chars_count = _mask_inputs(input_ids, tokenizer, word_ids)

            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                sample_size = labels[labels != -100].shape[0]
                #skip examples with no masked tokens
                if sample_size > 0:
                    output = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = output.loss.item()
                    total_loss += loss*sample_size #model returns mean loss
                    num_tokens += sample_size
                    num_chars += masked_chars_count

    nll = total_loss/num_tokens
    ppl = math.exp(nll)
    bpc = nll*(num_tokens/num_chars)
    return (ppl, bpc)

def ppl(model, tokenizer, lang='en'):
    #load wiki40b eval data for ppl eval on given langauge
    #lm_data = tfds.load('wiki40b/{}'.format(lang), split='validation', data_dir="/checkpoint/tblevins/data/wiki40b/")

    #ppl, bpc = _calculate_ppl(model, tokenizer, lm_data)
    #print('{} dev ppl = {}'.format(lang, ppl))
    #print('{} dev bpc = {}'.format(lang, bpc))

    lm_data = tfds.load('wiki40b/{}'.format(lang), split='test[:3000]', data_dir="/checkpoint/tblevins/data/wiki40b/")

    ppl, bpc = _calculate_ppl(model, tokenizer, lm_data)
    print('{} test ppl = {}'.format(lang, ppl))
    print('{} test bpc = {}'.format(lang, bpc))

def set_up_and_run_task(args):

    #load huggingface model, tokenizer
    #TODO: make the model being loading an argument

    model, tokenizer = load_hf_model(args.model_class, args.model_name, task=args.task, random_weights=args.rand_weights)
    model.cuda()
    model.eval()
    if args.task == 'pos':
        pos(model, tokenizer, args.dataset_path, args.checkpoint_path, lang=args.lang)
    else:
        ppl(model, tokenizer, lang=args.lang)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #allow languages from any eval task in args
    all_choices = []
    for _, v in _lang_choices.items(): all_choices.extend(v)

    # Required parameters
    parser.add_argument(
        "--checkpoint_path", default='./model_best.ckpt', type=str, help="Path to where the best model ckpt is saved"
    )
    parser.add_argument(
        "--dataset_path", default=None, type=str, help="Path the dataset to use for evaluation (not needed for perplexity eval)."
    )
    parser.add_argument(
        "--lang", default=None, type=str, choices=all_choices, required=True, help="Language on which to evaluate the given FairSeq XLMR checkpoint (if not given for ppl, evaluates jointly on all languages in dataset; otherwise required)."
    )
    parser.add_argument(
        "--task", default=None, type=str, choices=_task_choices, required=True, help="Task on which to evaluate the given FairSeq XLMR checkpoint"
    )
    parser.add_argument(
        "--model_name", default='bert-base-cased', choices=_model_names
    )
    parser.add_argument(
        "--model_class", default='bert', choices=list(MODEL_CLASSES.keys())
    )
    parser.add_argument(
        "--rand-weights", action='store_true'
    )
    parser.add_argument(
        "--rand_seed", default=1, type=int
    )

    args = parser.parse_args()

    #ensure that given lang matches given task
    assert args.lang in _lang_choices[args.task]

    #set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)   
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    set_up_and_run_task(args)

#EOF