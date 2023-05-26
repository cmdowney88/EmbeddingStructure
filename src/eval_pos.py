import argparse
import torch
from torch.nn import functional as F
import os
import random
import math
import numpy as np
from tqdm import tqdm
import copy

from utils import MODEL_CLASSES, _lang_choices, _task_choices,\
     _model_names, _label_spaces, load_hf_model, load_ud_datasets,\
     _consolidate_features


class Tagger(torch.nn.Module):
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
                    if l in label_space else torch.LongTensor([-100])
                    for l in label_subset
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
            word_alignments = [
                x for x in range(alignment_id, alignment_id + len(word_ids))
            ]
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
            if l in label_space else torch.LongTensor([-100])
            for l in label_subset
        ]
        processed_dataset.append((input_ids, alignment, labels))

    # 1.183 for en train
    print(
        'Avg. number of subword pieces per word = {}'.format(
            num_subwords / num_words
        )
    )
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
    input_ids = torch.stack(
        [_pad_to_len(x, max_length, pad_idx) for x in input_ids], dim=0
    )

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
    epochs=1,
    patience=0,
    best_ckpt_path='./model_best.ckpt'
):
    model.train()
    model.to('cuda:0')

    max_valid_acc = 0
    patience_step = 0
    # training epochs
    for epoch_id in range(0, epochs):
        random.shuffle(data)

        train_loss = 0.

        # TODO: make this and eval work with new data format (and add feature
        # handling to forward()) go over training data
        for i in range(0, len(data), bsz):
            if i + bsz > len(data):
                batch_data = data[i:]
            else:
                batch_data = data[i:i + bsz]
            input_ids, alignments, labels = batchify(batch_data, pad_idx)

            input_ids = input_ids.to('cuda:0')
            labels = labels.to('cuda:0')

            optimizer.zero_grad()
            output = model.forward(input_ids, alignments)

            loss = criterion(output, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

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
            torch.save(model.state_dict(), best_ckpt_path)
            patience_step = 0

    # return model, number of training epochs for best ckpt
    return model, (epoch_id + 1 - patience_step)


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


# task expects loose .conllu files for train an valid for the given language in
# the data_path dir
def pos(model, tokenizer, data_path, ckpt_path, lang='en', max_epochs=50):

    # load UD train and eval data for probing on given langauge
    train_text_data, valid_text_data, test_text_data = load_ud_datasets(
        data_path, lang, task='pos'
    )

    #DEBUGGING
    #train_data = train_data[:100]
    pos_labels = _label_spaces['UPOS']

    # Get majority sense baseline
    label_counts = []
    words = {}
    for _ in range(len(pos_labels)):
        label_counts.append(0.)
    for sent, labels in train_text_data:
        for w, l in zip(sent, labels):
            if l == '_':
                continue
            label_counts[pos_labels.index(l)] += 1
            if w not in words:
                words[w] = [0. for _ in range(len(pos_labels))]
            words[w][pos_labels.index(l)] += 1

    print(
        'majority label baseline: {}'.format(
            (max(label_counts) / sum(label_counts)) * 100
        )
    )

    for w in words:
        words[w] = pos_labels[np.argmax(words[w])]
    per_word_maj_correct = 0.
    per_word_maj_total = 0.
    for sent, labels in test_text_data:
        for w, l in zip(sent, labels):
            if w in words and words[w] == l:
                per_word_maj_correct += 1
            per_word_maj_total += 1
    print(
        'per word majority baseline: {}'.format(
            (per_word_maj_correct / per_word_maj_total) * 100
        )
    )

    # preprocessing can take in a hidden layer id if we want to probe inside model
    train_data = preprocess_ud_word_level(
        tokenizer, train_text_data, pos_labels
    )
    valid_data = preprocess_ud_word_level(
        tokenizer, valid_text_data, pos_labels
    )
    test_data = preprocess_ud_word_level(tokenizer, test_text_data, pos_labels)

    random_seeds = [1, 2, 3, 4, 5]
    scores = []
    epochs = []
    for rand_x in random_seeds:
        # set random seeds for model init, data shuffling
        torch.cuda.manual_seed(rand_x)
        torch.cuda.manual_seed_all(rand_x)
        np.random.seed(rand_x)
        random.seed(rand_x)

        # create probe model
        tagger = Tagger(model, len(pos_labels))

        _batch_sizes = {
            'bert-base-cased': 16,  #32, 
            'bert-base-multilingual-cased': 16,  #32,
            'bert-large-cased': 8,
            'roberta-base': 16,  #32, 
            'roberta-large': 8,
            'xlm-roberta-base': 16,  #32, 
            'xlm-roberta-large': 8,
            't5-base': 16
        }

        # load criterion and optimizer
        tagger_lr = 0.000005
        tagger_bsz = _batch_sizes[args.model_name]
        patience = 3

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(tagger.parameters(), lr=tagger_lr)

        # train probe model
        tagger, num_epochs = train_model(
            tagger,
            train_data,
            valid_data,
            criterion,
            optimizer,
            tokenizer.pad_token_id,
            bsz=tagger_bsz,
            epochs=max_epochs,
            patience=patience,
            best_ckpt_path=ckpt_path
        )

        # load best checkpoint for model state
        tagger.load_state_dict(torch.load(ckpt_path))

        # evaluate on probe model
        acc = evaluate_model(
            tagger, test_data, tokenizer.pad_token_id, bsz=tagger_bsz
        )
        acc = acc * 100
        scores.append(acc)
        epochs.append(num_epochs * 1.0)
        print(acc, num_epochs)

        # reinitalize the model for each trial if using randomly initialized encoder
        if args.rand_weights:
            model, tokenizer = load_hf_model(
                args.model_class,
                args.model_name,
                task=args.task,
                random_weights=args.rand_weights
            )
            model.cuda()
            model.eval()

            train_data = preprocess_ud_word_level(
                tokenizer, train_text_data, pos_labels
            )
            valid_data = preprocess_ud_word_level(
                tokenizer, valid_text_data, pos_labels
            )
            test_data = preprocess_ud_word_level(
                tokenizer, test_text_data, pos_labels
            )

    mean = sum(scores) / len(scores)
    print('mean score = {}'.format(mean))
    var = sum([(s - mean)**2 for s in scores]) / len(scores)
    #print('score variance = {}'.format(var))
    std_dev = math.sqrt(var)
    print('standard deviation = {}'.format(std_dev))
    #std_err = math.sqrt(var)/math.sqrt(len(scores))
    #print('standard error = {}'.format(std_err))
    mean_epochs = sum(epochs) / len(epochs)
    print('mean epochs = {}'.format(mean_epochs))


def set_up_and_run_task(args):

    # load huggingface model, tokenizer
    # TODO: make the model being loading an argument

    model, tokenizer = load_hf_model(
        args.model_class,
        args.model_name,
        task=args.task,
        random_weights=args.rand_weights
    )
    model.cuda()
    model.eval()
    # Do pos task
    pos(
        model,
        tokenizer,
        args.dataset_path,
        args.checkpoint_path,
        lang=args.lang,
        max_epochs=args.epochs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # allow languages from any eval task in args
    all_choices = []
    for _, v in _lang_choices.items():
        all_choices.extend(v)

    # required parameters
    parser.add_argument(
        "--checkpoint_path",
        default='./model_best.ckpt',
        type=str,
        help="Path to where the best model ckpt is saved"
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        help=
        "Path the dataset to use for evaluation (not needed for perplexity eval)."
    )
    parser.add_argument(
        "--lang",
        default=None,
        type=str,
        choices=all_choices,
        required=True,
        help=
        "Language on which to evaluate the given FairSeq XLMR checkpoint (if not given for ppl, evaluates jointly on all languages in dataset; otherwise required)."
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        choices=_task_choices,
        required=True,
        help="Task on which to evaluate the given FairSeq XLMR checkpoint"
    )
    parser.add_argument(
        "--model_name", default='bert-base-cased', choices=_model_names
    )
    parser.add_argument(
        "--model_class", default='bert', choices=list(MODEL_CLASSES.keys())
    )
    parser.add_argument("--rand-weights", action='store_true')
    parser.add_argument("--rand_seed", default=42, type=int)
    parser.add_argument("--epochs", default=50, type=int)

    args = parser.parse_args()

    # ensure that given lang matches given task
    assert args.lang in _lang_choices[args.task]

    # set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_up_and_run_task(args)

#EOF
