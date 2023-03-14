import argparse
import yaml

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing

# read training configurations from YAML file
parser = argparse.ArgumentParser(
    description="Train tokenizer on raw vocab"
)
parser.add_argument('--config', type=str)
args = parser.parse_args()
config_dict = vars(args)
with open(args.config, 'r') as config_file:
    config_dict.update(yaml.load(config_file, Loader=yaml.Loader))

# train a BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# normalization
normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.normalizer = normalizer

# pre-tokenization
tokenizer.pre_tokenizer = Whitespace()

# special tokens and tokenizer training parameters
trainer = BpeTrainer(vocab_size=args.tokenizer_vocab_size,
                     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# start training tokenizer
tokenizer.train(files=[args.train_dataset_path],
                trainer=trainer)

# add BERT special tokens during post-processing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

# save trained tokenizer
tokenizer.save(f"{args.tokenizer_save_dir}/tokenizer.json")
