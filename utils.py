import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets

def load_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    return model

def load_data(dataset_name, split, subset=None, start_index=0, num_samples=100):
    if subset:
        dataset = datasets.load_dataset(dataset_name, subset, split=split)
    else:
        dataset = datasets.load_dataset(dataset_name, split=split)
    if start_index >= len(dataset):
        raise ValueError(f"Start index {start_index} is out of range for the dataset.")
    end_index = min(start_index + num_samples, len(dataset))
    dataset = dataset.select(range(start_index, end_index))
    return dataset

