import argparse
import os
import random
import yaml

from pathlib import Path
import sentencepiece as spm

# Read training configurations from YAML file
parser = argparse.ArgumentParser(
    description="Train SentencePiece tokenizer on raw vocab"
)
parser.add_argument('--config', type=str)
args = parser.parse_args()
config_dict = vars(args)
with open(args.config, 'r') as config_file:
    config_dict.update(yaml.load(config_file, Loader=yaml.Loader))

output_dir = Path(args.output_path).parent
os.makedirs(output_dir, exist_ok=True)

random.seed(1)

spm.SentencePieceTrainer.Train(
    input=args.train_dataset_path,
    model_prefix=args.output_path,
    vocab_size=args.vocab_size,
    bos_id=0,
    pad_id=1,
    eos_id=2,
    unk_id=3,
    bos_piece='[CLS]',
    pad_piece='[PAD]',
    eos_piece='[SEP]',
    unk_piece='[UNK]',
    user_defined_symbols='[MASK]',
    model_type=args.tokenizer_model,
    character_coverage=0.999
)
