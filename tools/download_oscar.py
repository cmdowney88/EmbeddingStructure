import sys
from datasets import load_dataset
from tqdm import tqdm

language = sys.argv[1]
auth_token = sys.argv[2]
output_file = sys.argv[3]

dataset = load_dataset(
    "oscar-corpus/OSCAR-2201",
    use_auth_token=auth_token,
    language=language,
    streaming=True,
    split="train"
)

with open(output_file, "w") as fout:
    for record in tqdm(dataset):
        print(record["text"], file=fout)
