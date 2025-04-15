import argparse
import utils
import json
import torch
import os
import pdb

dataset_config = json.load(open("../configs/dataset_config.json"))
model_config = json.load(open("../configs/model_config.json"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Given a model and a tokenized dataset, we calculate at the token level, the rank, probability, and quartile of gold token in generation distribution."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to token level information.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="The name of the model to evaluate."
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="The starting index for the dataset."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="The number of samples to evaluate. Samples are taken from the start index.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for processing the dataset.",
    )
    return parser.parse_args()


def process_batch(examples, model, device):
    input_ids_list = examples["input_ids"]
    max_input_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids = [
        ids + [0] * (max_input_len - len(ids)) for ids in input_ids_list
    ]
    input_ids = torch.tensor(padded_input_ids).to(model.device)

    batch_size, seq_len = input_ids.size()

    attention_masks = examples["attention_mask"]
    max_len = max(len(mask) for mask in attention_masks)

    padded_attention_masks = [
        mask + [0] * (max_len - len(mask)) for mask in attention_masks
    ]
    attention_mask = torch.tensor(padded_attention_masks).to(model.device)

    # pdb.set_trace()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    all_token_probabilities = []
    all_token_ranks = []
    all_cumulative_probabilities = []

    for i in range(batch_size):
        example_probabilities = probabilities[i]  # (seq_len, vocab_size)
        example_input_ids = input_ids[i]  # (seq_len,)

        token_probabilities = []
        token_ranks = []
        cumulative_probabilities = []

        for j in range(seq_len):
            token_id = example_input_ids[j].item()
            prob_distribution = example_probabilities[j]

            token_prob = prob_distribution[token_id].item()
            token_probabilities.append(token_prob)

            sorted_probs, sorted_indices = torch.sort(
                prob_distribution, descending=True
            )
            rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
            cumulative_prob_less_likely = 1 - torch.sum(sorted_probs[:rank]).item()
            token_ranks.append(rank)
            cumulative_probabilities.append(cumulative_prob_less_likely)

        all_token_probabilities.append(token_probabilities)
        all_token_ranks.append(token_ranks)
        all_cumulative_probabilities.append(cumulative_probabilities)

    return {
        "token_probabilities": all_token_probabilities,
        "token_ranks": all_token_ranks,
        "cumulative_probabilities": all_cumulative_probabilities,
    }


def main(data, model, batch_size, device):
    data = data.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"model": model, "device": device},
    )
    return data


if __name__ == "__main__":
    args = parse_args()
    assert (
        args.dataset in dataset_config
    ), f"Dataset {args.dataset} not found in dataset_config.json"
    assert (
        args.model in model_config
    ), f"Model {args.model} not found in model_tokenizer_map.json"

    tokenizer_name = model_config[args.model]["tokenizer"]
    directory = dataset_config[args.dataset]["tokenized_data_directory"]
    split = dataset_config[args.dataset]["split_name"]

    data_path = f"{directory}/{tokenizer_name}/{split}"

    data = utils.load_data_local(data_path, args.start_index, args.num_samples)
    model = utils.load_model(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    print("Model Name: ", args.model)
    print("Dataset Name: ", args.dataset)
    print("Start Index: ", args.start_index)
    print("Number of Samples: ", args.num_samples)
    print("Data Path: ", data_path)
    print("Tokenizer Name: ", tokenizer_name)
    print("Model Device: ", model.device)
    print("Model Device Map: ", model.hf_device_map)
    print("Number of GPUs available: ", torch.cuda.device_count())
    print("Batch Size: ", args.batch_size)

    processed_data = main(data, model, args.batch_size, device)

    base_output_directory = dataset_config[args.dataset]["token_metrics_data_directory"]
    model_name = model_config[args.model]["name"]
    output_dir = f"{base_output_directory}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{tokenizer_name}/{split}"

    processed_data.save_to_disk(output_path)
    print(f"Processed data saved to {output_path}")
