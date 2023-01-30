import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments

# load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large', max_len=512)
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

# move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# prepare training data
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/Users/ibrahimsharafelden/workspace/TXLM/opus_data/data/georgian/CCMatrix_latest_mono_ka.txt",
    block_size=128,
)

# prepare validation data
val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/Users/ibrahimsharafelden/workspace/TXLM/opus_data/data/georgian/CCMatrix_latest_mono_ka.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# initialize trainer class
training_args = TrainingArguments(
    output_dir="./TXLM",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10000,
    save_total_limit=2,
    evaluation_strategy='steps',
    eval_steps=10000,
    seed=42,
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
