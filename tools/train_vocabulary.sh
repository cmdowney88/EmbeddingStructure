CONFIG_FILE=$1

python -u tools/sample_and_merge_langs.py --config configs/train_vocab.yml

python -u tools/train_spm.py --config configs/train_vocab.yml
