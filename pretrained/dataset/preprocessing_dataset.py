import os
import json
import argparse

import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from itertools import chain
from datasets import load_from_disk
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name",                type=str, required=True,  help="HuggingFace Hub model name",             default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--dataset",                   type=str, required=True,  help="Name / Path of the dataset to load.",    default="Project44/Dummy")
parser.add_argument("--subset",                    type=str, required=False, help="subset of the dataset to load.",         default=None)
parser.add_argument("--output_dataset_path",       type=str, required=True,  help="Path were the dataset will be saved",    default="./datasets/Dummy/")
parser.add_argument("--batch_size",                type=int, required=False, help="Batch size for group_texts",             default=100000)
parser.add_argument("--seed",                      type=int, required=False, help="Random Seed",                            default=42)
parser.add_argument("--preprocessing_num_workers", type=int, required=False, help="Number of threads to build the dataset", default=10)
args = parser.parse_args()

# dataset = load_dataset(args.dataset, args.subset)
dataset = load_from_disk(args.dataset)
train_dataset = dataset["train"]

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    model_max_length=2048,
)
tokenizer.add_special_tokens({'pad_token': '<pad>'})

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def add_end_of_sentence(example):
    example["text"] = example["text"] + tokenizer.eos_token
    return example

print("Add the EOS token")
train_dataset = train_dataset.map(add_end_of_sentence, num_proc=args.preprocessing_num_workers, batched=False)

def tokenize_function(example):
    return tokenizer(example["text"], return_special_tokens_mask=True)
    
print("Tokenize the dataset")
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    num_proc=args.preprocessing_num_workers,
    remove_columns=['text'],
    batched=True,
)

max_seq_length=2048

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


tokenized_datasets = tokenized_train_dataset.map(
    group_texts,
    batched=True,
    batch_size=args.batch_size,
    num_proc=args.preprocessing_num_workers,
    desc=f"Grouping texts in chunks of {max_seq_length}",
)

filtered_dataset = tokenized_datasets.filter(lambda example: [example for input_ids in example["input_ids"] if len(input_ids) >= 2048],
    batched=True,
    batch_size=args.batch_size,
    num_proc=args.preprocessing_num_workers,
    desc=f"Filtering texts",
)

# hf_dataset = hf_dataset.shuffle(args.seed)
filtered_dataset.save_to_disk(args.output_dataset_path)
print('Dataset saved.')
