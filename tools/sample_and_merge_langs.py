import argparse
import os
import random
import yaml

from tqdm import tqdm

# read training configurations from YAML file
parser = argparse.ArgumentParser(description="Train tokenizer on raw vocab")
parser.add_argument('--config', type=str)
args = parser.parse_args()
config_dict = vars(args)
with open(args.config, 'r') as config_file:
    config_dict.update(yaml.load(config_file, Loader=yaml.Loader))

random.seed(1)

datafiles = args.data_files
languages = list(datafiles.keys())
lang_data = {}
total_length = 0

for language in datafiles:
    lang_data[language] = {}
    lang_data[language]['text'] = []
    for filename in datafiles[language]['files']:
        filepath = os.path.join(datafiles[language]['directory'], filename)
        lines = [
            line.strip() for line in open(filepath, 'r') if line.strip() != ''
        ]
        lang_data[language]['text'] += lines
    lang_length = len(lang_data[language]['text'])
    lang_data[language]['length'] = lang_length
    total_length += lang_length

print(f"Sampling and merging {len(languages)} languages")
print(f"Total number of lines: {total_length}")
print(f"Sampling alpha: {args.alpha}")

orig_probs = {
    lang: (subdict['length'] / total_length)
    for lang, subdict in lang_data.items()
}

if getattr(args, 'composition_file', False):
    with open("language_composition.csv", 'w') as fout:
        print("language, lines, proportion", file=fout)
        for language, subdict in lang_data.items():
            print(f"{language}, {subdict['length']}, {orig_probs[language]}", file=fout)

print("\nOriginal probabilities")
for lang, prob in orig_probs.items():
    print(f"{lang}: {round(prob, 4)}")

exponiated_probs = {
    lang: prob**args.alpha
    for lang, prob in orig_probs.items()
}

z = sum(exponiated_probs.values())

sample_probs = {
    lang: exponiated_probs[lang] / z
    for lang in languages
}

print("\nUp-/Down-sampled probabilities")
for lang, prob in sample_probs.items():
    print(f"{lang}: {round(prob, 4)}")

lambdas = {
    lang: (1 / orig_probs[lang]) * sample_probs[lang]
    for lang in languages
}

sample_probs_list = [sample_probs[lang] for lang in languages]

if getattr(args, 'output_length', False):
    output_length = args.output_length
    print(f"\nSampling {output_length} lines to output file")
else:
    output_length = total_length

with open(args.combined_data_output, 'w') as fout:
    for i in tqdm(range(output_length)):
        sampled_lang = random.choices(languages, weights=sample_probs_list, k=1)[0]
        sampled_line = random.sample(lang_data[sampled_lang]['text'], k=1)[0]
        print(sampled_line, file=fout)
