import argparse
import datasets
import utils
import sys
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to tokenize.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model to use for tokenization.")
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="The starting index for the dataset.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="The number of samples to tokenize. Samples are taken from the start index.")
    parser.add_argument(
        "--split", type=str, default="test", help="The split of the dataset to use. Default is 'test'.")
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="The subset of the dataset to use. Default is None.")
    parser.add_argument(
        "--text_col", default="input", type=str, help="The column name for the text data. Default is 'input'.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the tokenized dataset.")
    
    return parser.parse_args()

def tokenize(example, tokenizer, col="input"):
    text = example[col]
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        max_length=sys.maxsize,
        return_attention_mask=True,
        return_tensors="pt",
    )
    example["input_ids"] = tokenized["input_ids"][0]
    example["attention_mask"] = tokenized["attention_mask"][0]
    example["tokenized_len"] = len(example["input_ids"])
    example["tokens"] = tokenizer.batch_decode(tokenized["input_ids"][0], skip_special_tokens=True)
    assert len(example["tokens"]) == len(example["input_ids"])
    example["input"] = text
    return example

def tokenize_with_progress(dataset, tokenizer, col):
    with tqdm(total=len(dataset), desc="Tokenizing") as pbar:
        def wrapped_tokenize(example):
            result = tokenize(example, tokenizer, col)
            pbar.update(1)
            return result
        return dataset.map(wrapped_tokenize)

def main(args):
    tokenizer = utils.load_tokenizer(args.model)
    dataset = utils.load_data(args.dataset, args.split, args.subset, args.start_index, args.num_samples)
    if args.dataset == "emozilla/govreport-test-tokenized":
        dataset = dataset.remove_columns(["input_ids", "attention_mask", "output", "tokenized_len"])
    print(f"Loaded {args.dataset} dataset with {len(dataset)} samples.")
    dataset = tokenize_with_progress(dataset, tokenizer, args.text_col)
    dataset = dataset.remove_columns([args.text_col]) if args.text_col != "input" else dataset
    path = os.path.join(args.save_dir, f"{args.start_index}_{args.num_samples}_{args.split}")
    if args.subset:
        path += f"_{args.subset}"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    dataset.save_to_disk(path)
    print(f"Tokenized dataset saved to {path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)