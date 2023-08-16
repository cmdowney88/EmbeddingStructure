import argparse
import numpy as np
import os
import random
import sys
import yaml
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, XLMRobertaTokenizer, AutoModelForMaskedLM, LineByLineTextDataset,
    DataCollatorForLanguageModeling, EarlyStoppingCallback, Trainer, TrainerCallback, TrainerControl,
    TrainerState, TrainingArguments
)

from data import ShardedTextDataset
from utils import _xlmr_special_tokens


class CheckpointControlCallback(TrainerCallback):
    """
    """
    def __init__(self, trainer: Trainer, checkpoint_path: str):
        self.trainer = trainer
        self.checkpoint_path = checkpoint_path
        self.checkpoint_control_path = os.path.join(
            checkpoint_path, "checkpoint_control.yml"
        )

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        output_dir = self.checkpoint_path
        last_checkpoint_path = os.path.join(output_dir, f"checkpoint-{state.global_step}")
        with open(self.checkpoint_control_path, 'w') as fout:
            print("resume_from_checkpoint: True", file=fout)
            print(f"last_checkpoint: {last_checkpoint_path}", file=fout)

        train_dataset = self.trainer.train_dataset
        if isinstance(train_dataset, ShardedTextDataset):
            dataset_save_path = os.path.join(last_checkpoint_path, "train_dataset_state.yml")
            train_dataset.save(dataset_save_path)


class InitialFreezeCallback(TrainerCallback):
    """
    Callback to freeze some model parameters at the beginning of training. The parameter
    `model_freeze_prefix` controls which parameters to freeze (as a prefix of their name)
    """
    def __init__(self, trainer: Trainer, model_freeze_prefix: str):
        self.trainer = trainer
        self.model_freeze_prefix = model_freeze_prefix

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        for name, param in self.trainer.model.named_parameters():
            if name.startswith(self.model_freeze_prefix):
                param.requires_grad = False
        print(
            f"\nAll parameters with prefix {self.model_freeze_prefix} frozen", file=sys.stderr
        )

class UnfreezeCallback(TrainerCallback):
    """
    Callback to unfreeze the entire model at a certain point in training. The parameter
    `unfreeze_step_ratio` controls the point at which to unfreeze (as a ratio of the maximum
    training steps)
    """
    def __init__(self, trainer: Trainer, unfreeze_step_ratio: float):
        self.trainer = trainer
        self.unfreeze_step_ratio = unfreeze_step_ratio
        self.already_unfrozen = False

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        reached_unfreeze_step = state.global_step >= int(self.unfreeze_step_ratio * state.max_steps)
        if reached_unfreeze_step and not self.already_unfrozen:
            for param in self.trainer.model.parameters():
                param.requires_grad = True
            self.already_unfrozen = True
            print(
                f"\nAll model parameters unfrozen after global step {state.global_step}",
                file=sys.stderr
            )


if __name__ == "__main__":
    # read training configurations from YAML file
    parser = argparse.ArgumentParser(description="Finetune XLM-R model on raw text corpora")
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config_dict = vars(args)
    with open(args.config, 'r') as config_file:
        config_dict.update(yaml.load(config_file, Loader=yaml.Loader))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # establish or read a control file to determine whether to resume from a checkpoint
    os.makedirs(args.checkpoints_directory, exist_ok=True)
    checkpoint_control_path = os.path.join(args.checkpoints_directory, "checkpoint_control.yml")
    checkpoint_control_exists = os.path.isfile(checkpoint_control_path)
    if checkpoint_control_exists:
        with open(checkpoint_control_path, 'r') as fin:
            control_dict = yaml.load(fin, Loader=yaml.Loader)
        args.resume_from_checkpoint = control_dict['resume_from_checkpoint']
        args.control_dict = control_dict
        last_checkpoint_name = control_dict['last_checkpoint']
        last_checkpoint_name = Path(last_checkpoint_name).stem

        # resume_from_checkpoint==False is only okay if we're starting a new training
        # run; exit if the folder contains checkpoints not meant to be resumed
        if not args.resume_from_checkpoint:
            raise RuntimeError(
                f"Not resuming from checkpoints in {args.checkpoints_directory};"
                " control file indicates resuming should be blocked"
            )
        print(
           f"Checkpoint control file found; resuming training from {last_checkpoint_name}",
            file=sys.stderr
        )
    else:
        args.resume_from_checkpoint = False

    # load pretrained model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(args.hf_model)

    # if using a new vocabulary, load in the vocab model, replace the embedding
    # layer of the model, tie the new weights to the output, and reset vocab size
    if getattr(args, 'new_vocab_file', False):
        # Get the vocab and embeddings from base XLM-R model
        old_tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        old_vocab = old_tokenizer.get_vocab()
        old_embeddings = model.get_input_embeddings().weight.detach()

        # read in the new tokenizer, initialize new embeddings
        new_tokenizer = XLMRobertaTokenizer(vocab_file=args.new_vocab_file)
        tokenizer = new_tokenizer
        new_vocab = new_tokenizer.get_vocab()
        new_vocab_size = new_tokenizer.vocab_size

        if getattr(args, 'new_embedding_path', False):
            # hard-coding the pad token for now
            new_padding_index = new_vocab['<pad>']
            new_embedding_weights = torch.load(args.new_embedding_path)
            new_embeddings = torch.nn.Embedding.from_pretrained(
                new_embedding_weights, padding_idx=1
            )
            print(f"Loaded new embeddings from {args.new_embedding_path}", file=sys.stderr)
        else:
            new_embedding_weights = torch.nn.Embedding(
                new_vocab_size, model.config.hidden_size
            ).weight.detach()
            # set the embeddings for special tokens to be identical to XLM-R
            for special_token in _xlmr_special_tokens:
                old_token_index = old_vocab[special_token]
                new_token_index = new_vocab[special_token]
                new_embedding_weights[new_token_index] = old_embeddings[old_token_index]
            new_embeddings = torch.nn.Embedding.from_pretrained(
                new_embedding_weights, padding_idx=1
            )
            print("Initialized new embeddings randomly", file=sys.stderr)

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

    # for sanity, make sure all parameters require gradients initially;
    # this is mostly in response to new embeddings not having grads, but might as
    # well make sure everything is trainable at first
    for p in model.parameters():
        p.requires_grad = True

    # move model to GPU if available
    device = torch.device('cuda')
    model.to(device)

    # prepare training data
    if getattr(args, 'sharded_train_dataset', False):
        if args.resume_from_checkpoint:
            last_checkpoint_path = args.control_dict['last_checkpoint']
            checkpoint_dataset_path = os.path.join(last_checkpoint_path, "train_dataset_state.yml")
            train_dataset = ShardedTextDataset.from_saved(checkpoint_dataset_path, tokenizer)
            print(
                f"Loaded training dataset state from {last_checkpoint_path}",
                file=sys.stderr
            )
        else:
            train_dataset = ShardedTextDataset(
                args.train_dataset_path,
                tokenizer,
                max_seq_length=args.max_seq_len,
                shuffle_within_shard=False
            )
    else:
        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer, file_path=args.train_dataset_path, block_size=args.max_seq_len
        )

    # prepare validation data
    val_dataset = LineByLineTextDataset(
        tokenizer=tokenizer, file_path=args.val_dataset_path, block_size=args.max_seq_len
    )

    # Get default values if some training arguments missing
    args.training_epochs = getattr(args, 'training_epochs', 1.0)
    args.training_steps = getattr(args, 'training_steps', -1)
    args.logging_steps = getattr(args, 'logging_steps', 500)
    args.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)

    # initialize trainer class with training configs
    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        log_level="info",
        num_train_epochs=args.training_epochs,
        max_steps=args.training_steps,
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.eval_strategy,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.saved_checkpoints_limit,
        load_best_model_at_end=True,
        output_dir=args.checkpoints_directory,
        overwrite_output_dir=True,
        weight_decay=float(args.weight_decay),
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=float(args.warmup_ratio),
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        auto_find_batch_size=args.auto_find_batch_size,
        group_by_length=args.group_by_length,
        gradient_checkpointing=args.gradient_checkpointing,
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

    checkpoint_callback = CheckpointControlCallback(trainer, args.checkpoints_directory)
    trainer.add_callback(checkpoint_callback)

    if getattr(args, 'early_stopping_patience', False):
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        trainer.add_callback(early_stopping_callback)

    if getattr(args, 'freeze_main_model', False):
        freeze_callback = InitialFreezeCallback(trainer, args.model_freeze_prefix)
        trainer.add_callback(freeze_callback)

    # may want to unfreeze transformer blocks at a point during training; use this custom callback
    # to do so
    if getattr(args, 'freeze_main_model', False) and getattr(args, 'unfreeze_step_ratio', None):
        unfreeze_callback = UnfreezeCallback(trainer, args.unfreeze_step_ratio)
        trainer.add_callback(unfreeze_callback)

    # start training
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # when training is finished, block resuming from checkpoint
    with open(checkpoint_control_path, 'w') as fout:
        print("resume_from_checkpoint: False", file=fout)

    best_checkpoint_path = trainer.state.best_model_checkpoint
    print(f"Best checkpoint: {best_checkpoint_path}", file=sys.stderr)

    best_checkpoint_path = os.path.join(args.checkpoints_directory, 'best-checkpoint')
    trainer.save_model(best_checkpoint_path)
    trainer.save_state()

    # evaluate model
    trainer.evaluate()
