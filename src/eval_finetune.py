import argparse
import copy
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from tqdm import tqdm

from utils import (
    MODEL_CLASSES, _consolidate_features, _label_spaces, _lang_choices, _model_names, _task_choices,
    load_hf_model, load_ud_splits
)


class Tagger(torch.nn.Module):
    """
    Tagger class for conducting classification with a pre-trained model. Takes in the pre-trained
    model as the `encoder` argument on initialization. The tagger/classification-head itself
    consists of a single linear layer + softmax
    """
    def __init__(self, encoder, output_dim):
        super(Tagger, self).__init__()

        self.encoder = copy.deepcopy(encoder)
        input_dim = self.encoder.encoder.layer[-1].output.dense.out_features

        self.proj = torch.nn.Linear(input_dim, output_dim)

    def forward(self, input_ids, alignments):
        # run through encoder
        output = self.encoder(input_ids).last_hidden_state

        # get single rep for each word
        feats = []
        for i in range(output.shape[0]):
            feats.extend(_consolidate_features(output[i], alignments[i]))
        output = torch.cat(feats, dim=0)

        output = self.proj(output)
        output = F.softmax(output, dim=-1)
        return output


def preprocess_ud_word_level(tokenizer, dataset, label_space, layer_id=-1):
    """
    Pre-process the text data from a UD dataset using a huggingface tokenizer, keeping track of the
    mapping between word and subword tokenizations. Break up sentences in the dataset that are
    longer than 511 tokens
    """
    processed_dataset = []

    num_subwords = 0.
    num_words = 0.
    for sentence, labels in tqdm(dataset):
        # tokenize manually word by word because huggingface can't get word
        # alignments to track correctly
        # https://github.com/huggingface/transformers/issues/9637
        tokens = []
        if tokenizer.cls_token_id != None:
            tokens = [tokenizer.cls_token_id]
        alignment = []
        label_subset = []
        alignment_id = 1  #offset for cls token
        for word, label in zip(sentence, labels):
            word_ids = tokenizer.encode(' ' + word, add_special_tokens=False)
            if len(tokens) + len(word_ids) > 511:
                # add example to dataset
                if tokenizer.cls_token_id != None:
                    tokens += [tokenizer.sep_token_id]
                else:
                    tokens += [tokenizer.eos_token_id]

                input_ids = torch.LongTensor(tokens)
                labels = [
                    torch.LongTensor([label_space.index(l)])
                    if l in label_space else torch.LongTensor([-100]) for l in label_subset
                ]
                processed_dataset.append((input_ids, alignment, labels))

                # reset
                tokens = []
                if tokenizer.cls_token_id != None:
                    tokens = [tokenizer.cls_token_id]
                alignment = []
                label_subset = []
                alignment_id = 1  #offset for cls token

            tokens.extend(word_ids)
            word_alignments = [x for x in range(alignment_id, alignment_id + len(word_ids))]
            alignment_id = alignment_id + len(word_ids)
            alignment.append(word_alignments)
            label_subset.append(label)

            num_subwords += len(word_ids)
            num_words += 1

        if tokenizer.cls_token_id != None:
            tokens += [tokenizer.sep_token_id]
        else:
            tokens += [tokenizer.eos_token_id]

        input_ids = torch.LongTensor(tokens)
        # process labels into tensor
        # labels = [torch.LongTensor([label_space.index(l)]) for l in labels]
        # filtering out "unlabeled" examples with "_"
        labels = [
            torch.LongTensor([label_space.index(l)])
            if l in label_space else torch.LongTensor([-100]) for l in label_subset
        ]
        processed_dataset.append((input_ids, alignment, labels))

    # 1.183 for en train
    # print('Avg. number of subword pieces per word = {}'.format(num_subwords / num_words))
    return processed_dataset


def _pad_to_len(seq, length, pad_idx):
    s_len = seq.shape[-1]
    if s_len < length:
        pad_tensor = torch.LongTensor([pad_idx] * (length - s_len))
        seq = torch.cat([seq, pad_tensor], dim=-1)
    return seq


def batchify(data, pad_idx):
    input_ids, alignments, labels = zip(*data)

    max_length = max([x.shape[-1] for x in input_ids])
    input_ids = torch.stack([_pad_to_len(x, max_length, pad_idx) for x in input_ids], dim=0)

    flat_labels = []
    for l in labels:
        flat_labels.extend(l)
    labels = torch.cat(flat_labels, dim=0)

    return input_ids, alignments, labels


def train_model(
    model,
    data,
    valid_data,
    criterion,
    optimizer,
    pad_idx,
    bsz=1,
    gradient_accumulation=1,
    epochs=1,
    max_train_examples=math.inf,
    eval_every=2,
    patience=0,
    model_dir='./eval_checkpoints'
):
    model.train()
    model.to('cuda:0')

    max_valid_acc = 0
    patience_step = 0
    accumulation_counter = 0

    total_examples = min(len(data), max_train_examples)
    if total_examples < len(data):
        random.shuffle(data)
        data = data[:total_examples]

    # training epochs
    for epoch_id in tqdm(range(0, epochs), desc='training loop', total=epochs, unit='epoch'):
        random.shuffle(data)

        # TODO: make this and eval work with new data format (and add feature
        # handling to forward()) go over training data
        for i in range(0, total_examples, bsz):
            if i + bsz > len(data):
                batch_data = data[i:]
            else:
                batch_data = data[i:i + bsz]
            input_ids, alignments, labels = batchify(batch_data, pad_idx)

            input_ids = input_ids.to('cuda:0')
            labels = labels.to('cuda:0')

            output = model.forward(input_ids, alignments)
            loss = criterion(output, labels)
            loss.backward()
            accumulation_counter += 1
            if accumulation_counter >= gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_counter = 0

        if ((epoch_id + 1) % eval_every) == 0:
            valid_acc = evaluate_model(model, valid_data, pad_idx, bsz=bsz)

            # use train loss to see if we need to stop
            if max_valid_acc > valid_acc:
                if patience_step == patience:
                    break
                else:
                    patience_step += 1
            else:
                max_valid_acc = valid_acc
                # checkpoint model
                best_model_path = os.path.join(model_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                patience_step = 0

    # output a marker that training is completed, as well as the number of epochs to convergence;
    # this is so trials need not be repeated if a job is preempted
    num_epochs_to_convergence = (epoch_id + 1 - (patience_step * eval_every))
    checkpoint_control_path = os.path.join(model_dir, "checkpoint_control.yml")
    with open(checkpoint_control_path, 'w') as fout:
        print("training_complete: True", file=fout)
        print(f"num_epochs_to_convergence: {num_epochs_to_convergence}", file=fout)

    # return model, number of training epochs for best ckpt
    return model, num_epochs_to_convergence


def evaluate_model(model, data, pad_idx, bsz=1):
    model.eval()
    model.to('cuda:0')

    num_correct = 0.
    total_num = 0.

    for i in range(0, len(data), bsz):
        if i + bsz > len(data):
            batch_data = data[i:]
        else:
            batch_data = data[i:i + bsz]
        input_ids, alignments, labels = batchify(batch_data, pad_idx)

        input_ids = input_ids.to('cuda:0')
        labels = labels.to('cuda:0')

        with torch.no_grad():
            output = model.forward(input_ids, alignments)
        _, preds = torch.topk(output, k=1, dim=-1)

        batch_correct = torch.eq(labels.squeeze(), preds.squeeze()).sum().item()
        num_correct += batch_correct
        total_num += labels.shape[-1]

    return num_correct / total_num


def majority_label_baseline(text_data, label_set):
    """
    Return the majority-label baseline for a text dataset of sentences and their respective
    word-wise labels. Majority-label baseline is the accuracy achieved if a model predicts only the
    most common label
    """
    num_labels = len(label_set)
    label_counts = []
    words = {}
    for _ in range(num_labels):
        label_counts.append(0.)
    for sent, labels in text_data:
        for w, l in zip(sent, labels):
            if l == '_':
                continue
            label_counts[label_set.index(l)] += 1
            if w not in words:
                words[w] = [0. for _ in range(num_labels)]
            words[w][label_set.index(l)] += 1

    return (max(label_counts) / sum(label_counts) * 100, words)


def per_word_majority_baseline(text_data, word_label_count, label_set):
    """
    Return the per-word majority-label baseline for a text dataset of sentences and their respective
    word-wise labels. Per-word majority-label baseline is the accuracy achieved if a model predicts
    the most common label for each token
    """
    words = word_label_count
    for w in words:
        words[w] = label_set[np.argmax(words[w])]
    per_word_maj_correct = 0.
    per_word_maj_total = 0.
    for sent, labels in text_data:
        for w, l in zip(sent, labels):
            if w in words and words[w] == l:
                per_word_maj_correct += 1
            per_word_maj_total += 1

    return per_word_maj_correct / per_word_maj_total * 100


def mean_stdev(values):
    """
    Return the mean and standard deviation of the input list of values
    """
    mean = sum(values) / len(values)
    var = sum([(v - mean)**2 for v in values]) / len(values)
    std_dev = math.sqrt(var)
    return mean, std_dev


# task expects loose .conllu files for train an valid for the given language in
# the data_path dir
def pos(
    args,
    model,
    tokenizer,
    data_path,
    ckpt_path,
    lang='en',
    max_epochs=50,
    max_train_examples=math.inf
):
    """
    Conduct fine-tuning and evaluation for a POS tagging task. Fine-tune the model on four different
    random seeds per training set. If `args.zero_shot_transfer` is true and
    `args.transfer_source` is specified, conduct zero-shot transfer. Zero-shot transfer assumes that
    only test set(s) are available for the language(s) being evaluated. If doing zero-shot,
    fine-tune the model on the training/dev sets available for `args.transfer_source`, then loop
    over the specified language test sets at eval time
    """
    pos_labels = _label_spaces['UPOS']
    is_zero_shot = getattr(args, 'zero_shot_transfer', False)
    has_transfer_source = getattr(args, 'transfer_source', False)
    do_zero_shot = is_zero_shot and has_transfer_source

    # If doing zero-shot transfer, load the train and dev sets for `args.transfer_source` and load
    # the test set for each of the languages in `args.langs`. Pre-process the UD data
    if do_zero_shot:
        train_valid_data = load_ud_splits(
            data_path, args.transfer_source, splits=['train', 'dev'], task='pos'
        )
        train_text_data = train_valid_data['train']
        valid_text_data = train_valid_data['dev']
        test_text_data = {
            lg: load_ud_splits(data_path, lg, splits=['test'], task='pos')
            for lg in args.langs
        }

        train_data = preprocess_ud_word_level(tokenizer, train_text_data, pos_labels)
        valid_data = preprocess_ud_word_level(tokenizer, valid_text_data, pos_labels)
        test_data = {
            lg: preprocess_ud_word_level(tokenizer, data, pos_labels)
            for lg, data in test_text_data.items()
        }

        scores = defaultdict(list)
    # If not doing zero-shot transfer, pre-process the train, dev, and test sets for the language
    # in question. Also get and log the majority label baselines
    else:
        print("########")
        print(f"Beginning POS evaluation for {lang}")
        # load UD train and eval data for probing on given langauge
        ud_splits = load_ud_splits(data_path, lang, task='pos')
        train_text_data, valid_text_data, test_text_data = [
            ud_splits[x] for x in ['train', 'dev', 'test']
        ]

        # Get majority sense baseline
        majority_baseline, words = majority_label_baseline(train_text_data, pos_labels)
        print(f"majority label baseline: {round(majority_baseline, 2)}")

        per_word_baseline = per_word_majority_baseline(test_text_data, words, pos_labels)
        print(f"per word majority baseline: {round(per_word_baseline, 2)}")

        # preprocessing can take in a hidden layer id if we want to probe inside model
        train_data = preprocess_ud_word_level(tokenizer, train_text_data, pos_labels)
        valid_data = preprocess_ud_word_level(tokenizer, valid_text_data, pos_labels)
        test_data = preprocess_ud_word_level(tokenizer, test_text_data, pos_labels)

        scores = []

    random_seeds = [1, 2, 3, 4]
    epochs = []
    for rand_x in random_seeds:
        # set random seeds for model init, data shuffling
        torch.cuda.manual_seed(rand_x)
        torch.cuda.manual_seed_all(rand_x)
        np.random.seed(rand_x)
        random.seed(rand_x)

        # look for a control file to determine if a trial has already been done, e.g. by a
        # submitted job that was pre-empted before completing all trials
        lang_name = args.transfer_source if do_zero_shot else lang
        model_dir = os.path.join(ckpt_path, lang_name, str(rand_x))
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_control_path = os.path.join(model_dir, "checkpoint_control.yml")
        checkpoint_control_exists = os.path.isfile(checkpoint_control_path)
        if checkpoint_control_exists:
            with open(checkpoint_control_path, 'r') as fin:
                control_dict = yaml.load(fin, Loader=yaml.Loader)
                training_complete = control_dict['training_complete']
        else:
            control_dict = None
            training_complete = False

        # create probe model
        tagger = Tagger(model, len(pos_labels))
        tagger_bsz = args.batch_size

        # only do training if we can't retrieve the existing checkpoint
        if not training_complete:
            # load criterion and optimizer
            tagger_lr = 0.000005
            gradient_accumulation = getattr(args, 'gradient_accumulation', 1)

            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            optimizer = torch.optim.Adam(tagger.parameters(), lr=tagger_lr)

            # train classification model
            tagger, num_epochs = train_model(
                tagger,
                train_data,
                valid_data,
                criterion,
                optimizer,
                tokenizer.pad_token_id,
                bsz=tagger_bsz,
                gradient_accumulation=gradient_accumulation,
                epochs=max_epochs,
                max_train_examples=max_train_examples,
                patience=args.patience,
                model_dir=model_dir
            )
        else:
            print(
                f"{lang_name} random seed {rand_x} previously trained; reading checkpoint", file=sys.stderr
            )
            num_epochs = control_dict['num_epochs_to_convergence']

        epochs.append(num_epochs * 1.0)

        # load best checkpoint for model state
        best_model_path = os.path.join(model_dir, "best_model.pt")
        tagger.load_state_dict(torch.load(best_model_path))

        print(f"random seed: {rand_x}")
        print(f"num epochs: {num_epochs}")

        # evaluate classificaiton model
        if do_zero_shot:
            for lg in args.langs:
                acc = evaluate_model(tagger, test_data[lg], tokenizer.pad_token_id, bsz=tagger_bsz)
                acc = acc * 100
                scores[lg].append(acc)
                print(f"{lg} accuracy: {round(acc, 2)}")
            print("----")
        else:
            acc = evaluate_model(tagger, test_data, tokenizer.pad_token_id, bsz=tagger_bsz)
            acc = acc * 100
            scores.append(acc)
            print(f"accuracy: {round(acc, 2)}")
            print("----")

        # reinitalize the model for each trial if using randomly initialized encoder
        if args.random_weights:
            model, tokenizer = load_hf_model(
                args.model_class,
                args.model_name,
                task=args.task,
                random_weights=args.random_weights
            )
            model.cuda()
            model.eval()

            if do_zero_shot:
                train_data = preprocess_ud_word_level(tokenizer, train_text_data, pos_labels)
                valid_data = preprocess_ud_word_level(tokenizer, valid_text_data, pos_labels)
                test_data = {
                    lg: preprocess_ud_word_level(tokenizer, data, pos_labels)
                    for lg, data in test_text_data.items()
                }
            else:
                train_data = preprocess_ud_word_level(tokenizer, train_text_data, pos_labels)
                valid_data = preprocess_ud_word_level(tokenizer, valid_text_data, pos_labels)
                test_data = preprocess_ud_word_level(tokenizer, test_text_data, pos_labels)

    print("all trials finished")
    mean_epochs, epochs_stdev = mean_stdev(epochs)
    print(f"mean epochs: {round(mean_epochs, 2)}")

    if do_zero_shot:
        for lg in args.langs:
            mean_score, score_stdev = mean_stdev(scores[lg])
            print(f"{lg} mean score: {round(mean_score, 2)}")
            print(f"{lg} standard deviation: {round(score_stdev, 2)}")
    else:
        mean_score, score_stdev = mean_stdev(scores)
        print(f"mean score: {round(mean_score, 2)}")
        print(f"standard deviation: {round(score_stdev, 2)}")


def set_up_pos(args):
    """
    For each language being evaluated, set up POS tagging task by loading a huggingface
    model/tokenizer then calling the `pos` function
    """
    # Loop over languages to eval
    for lang in args.langs:
        # load huggingface model, tokenizer
        model, tokenizer = load_hf_model(
            args.model_class,
            args.model_name,
            task=args.task,
            random_weights=args.random_weights,
            tokenizer_path=args.tokenizer_path
        )
        model.cuda()
        model.eval()
        # Do pos task
        pos(
            args,
            model,
            tokenizer,
            args.dataset_path,
            args.checkpoint_path,
            lang=lang,
            max_epochs=args.epochs,
            max_train_examples=args.max_train_examples
        )


def set_up_pos_zero_shot(args):
    """
    Set up POS tagging task by loading a huggingface model/tokenizer then calling the `pos`
    function. Unlike `set_up_pos`, the loop over evaluation languages occurs inside the `pos`
    function in the zero-shot case, since the model is fine-tuned only on `transfer-source`
    """
    # load huggingface model, tokenizer
    model, tokenizer = load_hf_model(
        args.model_class,
        args.model_name,
        task=args.task,
        random_weights=args.random_weights,
        tokenizer_path=args.tokenizer_path
    )
    model.cuda()
    model.eval()
    # Do pos task
    pos(
        args,
        model,
        tokenizer,
        args.dataset_path,
        args.checkpoint_path,
        max_epochs=args.epochs,
        max_train_examples=args.max_train_examples
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # allow languages from any eval task in args
    all_choices = []
    for _, v in _lang_choices.items():
        all_choices.extend(v)

    # required parameters
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config_dict = vars(args)
    with open(args.config, 'r') as config_file:
        config_dict.update(yaml.load(config_file, Loader=yaml.Loader))

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # set max_train_examples to infinity it is is null or absent in config
    if not getattr(args, 'max_train_examples', False):
        args.max_train_examples = math.inf

    args.tokenizer_path = getattr(args, 'tokenizer_path', None)

    # ensure that given lang matches given task
    for lang in args.langs:
        assert lang in _lang_choices[args.task]

    print(f"Pytorch version: {torch.__version__}")
    print(f"Pytorch CUDA version: {torch.version.cuda}")
    print(f"GPUs available: {torch.cuda.is_available()}")

    # set random seeds
    torch.manual_seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Normal POS eval assumes a train, dev, and test set for each language being evaluated.
    # Zero-shot POS assumes a test set is available for each language in question, and that
    # train/dev sets are availalble for `args.transfer_source`
    if args.zero_shot_transfer and args.transfer_source:
        print(
            f"Doing zero-shot transfer from source {args.transfer_source} to languages {', '.join(args.langs)}"
        )
        set_up_pos_zero_shot(args)
    else:
        set_up_pos(args)

#EOF
