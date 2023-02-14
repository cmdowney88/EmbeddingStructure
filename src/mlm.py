import argparse
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments


# read training configurations from YAML file
parser = argparse.ArgumentParser(
    description="Finetune XLM-R model on raw text corpora"
)
parser.add_argument('--config', type=str)
args = parser.parse_args()
config_dict = vars(args)
with open(config_dict['config'], 'r') as config_file:
    config_dict.update(yaml.load(config_file, Loader=yaml.Loader))


# load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config_dict["hf_model"], max_len=config_dict["max_len"])
model = AutoModelForMaskedLM.from_pretrained(config_dict["hf_model"])

# move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# prepare training data
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=config_dict["train_dataset_path"],
    block_size=config_dict["dataset_block_size"],
)

# prepare validation data
val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=config_dict["val_dataset_path"],
    block_size=config_dict["dataset_block_size"],
)

# initialize trainer class with training configs
training_args = TrainingArguments(
    seed=config_dict["seed"],
    data_seed=config_dict["seed"],
    log_level="info",
    num_train_epochs=config_dict["training_epochs"],
    learning_rate=float(config_dict["learning_rate"]),
    per_device_train_batch_size=config_dict["train_batch_size"],
    evaluation_strategy=config_dict["eval_strategy"],
    per_device_eval_batch_size=config_dict["eval_batch_size"],
    eval_steps=config_dict["eval_steps"],
    save_steps=config_dict["save_steps"],
    save_total_limit=config_dict["saved_checkpoints_limit"],
    output_dir=config_dict["checkpoints_directory"],
    overwrite_output_dir=True,
    weight_decay=float(config_dict["weight_decay"]),
    lr_scheduler_type=config_dict["lr_scheduler_type"],
    warmup_ratio=float(config_dict["warmup_ratio"]),
    warmup_steps=config_dict["warmup_steps"],
    auto_find_batch_size=config_dict["auto_find_batch_size"],
    group_by_length=config_dict["group_by_length"],
    gradient_checkpointing=config_dict["gradient_checkpointing"],
    gradient_accumulation_steps=config_dict["gradient_accumulation_steps"],
    fp16=config_dict["fp16"],
    fsdp=config_dict["torch_distributed_training"],
    full_determinism=config_dict["full_determinism"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=config_dict["mlm_masking_prob"]
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
