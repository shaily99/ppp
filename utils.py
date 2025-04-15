import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datasets


def load_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model, quantize=True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
    return model


def load_data_hf(dataset_name, split, subset=None, start_index=0, num_samples=100):
    if subset:
        dataset = datasets.load_dataset(dataset_name, subset, split=split)
    else:
        dataset = datasets.load_dataset(dataset_name, split=split)
    if start_index >= len(dataset):
        raise ValueError(f"Start index {start_index} is out of range for the dataset.")
    end_index = min(start_index + num_samples, len(dataset))
    dataset = dataset.select(range(start_index, end_index))
    return dataset


def load_data_local(data_path, start_index=0, num_samples=100):
    data = datasets.load_from_disk(data_path)

    if num_samples >= len(data):
        print(
            f"Requested number of samples {num_samples} exceeds the dataset size {len(data)}. "
            "Returning the entire dataset."
        )
        return data

    end_index = min(start_index + num_samples, len(data))
    data = data.select(range(start_index, end_index))

    print(f"Loaded {len(data)} samples from {data_path}.")
    return data
