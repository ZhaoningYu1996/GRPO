import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import csv

argparser = argparse.ArgumentParser(description='Get the maximum sequence length')
argparser.add_argument('--data_path', type=str, default='simplescaling/s1K', help='Path to save the processed data')
args = argparser.parse_args()

# Example for loading a dataset
dataset = load_dataset(args.data_path)
print(dataset)
texts = dataset['train']  # Adjust according to your dataset structure

print(f"Number of texts: {len(texts)}")

# tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")

max_seq_len = 0

seq_len_list = []

# Collect all text that has length less than 7500
cleaned_texts = []

# Calculate max sequence length
for i, text in enumerate(tqdm(texts, desc="Calculating max sequence length", total=len(texts))):
    # print(text)
    # print(text.keys())
    # print(stop)
    count = 0
    for key in text.keys():
        # if key in ['instruction', 'input', 'output']:
        if key in ['question']:
        # if key == 'solution':
        # if key == 'output':
            # max_input_len = 0
            # if key == 'input':
            #     for i in range(len(text[key])):
            #         tokens = tokenizer.encode(text[key][i], add_special_tokens=True)
            #         max_input_len = max(max_input_len, len(tokens))
            #     count += max_input_len
            # else:
            print(text[key])
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text[key]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            tokens = tokenizer.encode(prompt, add_special_tokens=True)  # Ensure special tokens are considered
            count += len(tokens)
        # else:
        #     print(f"Key: {key}")
        #     print(stop)
    max_seq_len = max(max_seq_len, count)
    seq_len_list.append(count)
    # if count <= 7500:
    #     cleaned_texts.append(text)

print(f"Maximum sequence length: {max_seq_len}")
