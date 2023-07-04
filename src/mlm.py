import argparse
import yaml

import torch
from transformers import (
    AutoTokenizer, XLMRobertaTokenizer, AutoModelForMaskedLM,
    LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer,
    TrainingArguments
)

from data import ShardedTextDataset
from utils import _xlmr_special_tokens

# read training configurations from YAML file
parser = argparse.ArgumentParser(
    description="Finetune XLM-R model on raw text corpora"
)
parser.add_argument('--config', type=str)
args = parser.parse_args()
config_dict = vars(args)
with open(args.config, 'r') as config_file:
    config_dict.update(yaml.load(config_file, Loader=yaml.Loader))

# load pretrained model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(args.hf_model)

# if using a new vocabulary, load in the vocab model, replace the embedding
# layer of the model, tie the new weights to the output, and reset vocab size
if getattr(args, 'new_vocab_file', False):
    # Get the vocab and embeddings from base XLM-R model
    old_tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    old_vocab = old_tokenizer.get_vocab()
    old_embeddings = model.get_input_embeddings().weight

    # read in the new tokenizer, initialize new embeddings
    new_tokenizer = XLMRobertaTokenizer(vocab_file=args.new_vocab_file)
    new_vocab = new_tokenizer.get_vocab()
    new_vocab_size = new_tokenizer.vocab_size
    
    if getattr(args, 'new_embedding_path', False):
        # hard-coding the pad token for now
        new_padding_index = new_vocab['<pad>']
        new_embedding_weights = torch.load(args.new_embedding_path)
        new_embeddings = torch.nn.Embedding.from_pretrained(new_embedding_weights, padding_idx=1)
    else:
        new_embeddings = torch.nn.Embedding(
            new_vocab_size, model.config.hidden_size
        )
        # set the embeddings for special tokens to be identical to XLM-R
        for special_token in _xlmr_special_tokens:
            old_token_index = old_vocab[special_token]
            new_token_index = new_vocab[special_token]
            new_embeddings.weight[new_token_index] = old_embeddings[old_token_index]

    # set the model's new embeddings, then tie weights to output layer
    model.set_input_embeddings(new_embeddings)
    model.tie_weights()
    model.config.vocab_size = new_vocab_size
# if the tokenizer/vocab path is different from the model path, load that vocab
elif getattr(args, 'vocab_file', False):
    tokenizer = XLMRobertaTokenizer(vocab_file=args.vocab_file)
# load the default tokenizer for the model path
else:
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

if getattr(args, 'freeze_main_model', False):
    for name, param in model.named_parameters():
        if name.startswith(args.model_freeze_prefix):
            param.requires_grad = False

# move model to GPU if available
device = torch.device('cuda')
model.to(device)

# prepare training data
if getattr(args, 'sharded_train_dataset', False):
    train_dataset = ShardedTextDataset(
        args.train_dataset_path, tokenizer, max_seq_length=args.max_seq_len
    )
else:
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.train_dataset_path,
        block_size=args.max_seq_len
    )

# prepare validation data
val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.val_dataset_path,
    block_size=args.max_seq_len
)

# initialize trainer class with training configs
training_args = TrainingArguments(
    seed=args.seed,
    data_seed=args.seed,
    log_level="info",
    num_train_epochs=args.training_epochs,
    learning_rate=float(args.learning_rate),
    per_device_train_batch_size=args.train_batch_size,
    logging_steps=args.logging_steps,
    evaluation_strategy=args.eval_strategy,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    save_total_limit=args.saved_checkpoints_limit,
    output_dir=args.checkpoints_directory,
    overwrite_output_dir=True,
    weight_decay=float(args.weight_decay),
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=float(args.warmup_ratio),
    warmup_steps=args.warmup_steps,
    auto_find_batch_size=args.auto_find_batch_size,
    group_by_length=args.group_by_length,
    gradient_checkpointing=args.gradient_checkpointing,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=args.fp16,
    fsdp=args.torch_distributed_training,
    full_determinism=args.full_determinism
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_masking_prob
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# start training
trainer.train()

# evaluate model
trainer.evaluate()
