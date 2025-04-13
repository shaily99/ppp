from transformers import AutoTokenizer

if __name__ == "__main__":
    models = [
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-405B",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]
    tokenizer_ref = AutoTokenizer.from_pretrained(models[0])
    tokenizer_ref_str = tokenizer_ref.backend_tokenizer.to_str()
    print("Tokenizer reference model:", models[0])
    for model in models[1:]:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer_str = tokenizer.backend_tokenizer.to_str()
        if tokenizer_str != tokenizer_ref_str:
            print(f"Tokenizer for {model} does NOT match the reference.")
        else:
            print(f"Tokenizer for {model} matches the reference.")

    print("========================================")

    models = [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ]
    tokenizer_ref = AutoTokenizer.from_pretrained(models[0])
    tokenizer_ref_str = tokenizer_ref.backend_tokenizer.to_str()
    print("Tokenizer reference model:", models[0])

    for model in models[1:]:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer_str = tokenizer.backend_tokenizer.to_str()
        if tokenizer_str != tokenizer_ref_str:
            print(f"Tokenizer for {model} does NOT match the reference.")
        else:
            print(f"Tokenizer for {model} matches the reference.")

    print("========================================")
    models = [
        "allenai/OLMo-7B-0724-hf",
        "allenai/OLMo-7B-0724-Instruct-hf",
        "allenai/OLMo-7B-0724-SFT-hf",
        "allenai/OLMo-1B-0724-hf",
    ]
    tokenizer_ref = AutoTokenizer.from_pretrained(models[0])
    tokenizer_ref_str = tokenizer_ref.backend_tokenizer.to_str()
    print("Tokenizer reference model:", models[0])

    for model in models[1:]:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer_str = tokenizer.backend_tokenizer.to_str()
        if tokenizer_str != tokenizer_ref_str:
            print(f"Tokenizer for {model} does NOT match the reference.")
        else:
            print(f"Tokenizer for {model} matches the reference.")
    print("========================================")
    models = [
        "allenai/OLMo-2-1124-7B",
        "allenai/OLMo-2-1124-13B",
        "allenai/OLMo-2-0325-32B",
    ]
    tokenizer_ref = AutoTokenizer.from_pretrained(models[0])
    tokenizer_ref_str = tokenizer_ref.backend_tokenizer.to_str()
    print("Tokenizer reference model:", models[0])

    for model in models[1:]:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer_str = tokenizer.backend_tokenizer.to_str()
        if tokenizer_str != tokenizer_ref_str:
            print(f"Tokenizer for {model} does NOT match the reference.")
        else:
            print(f"Tokenizer for {model} matches the reference.")
    print("========================================")
    models = [
        "google/gemma-3-1b-pt",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-pt",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-pt",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-pt",
        "google/gemma-3-27b-it",
    ]
    tokenizer_ref = AutoTokenizer.from_pretrained(models[0])
    tokenizer_ref_str = tokenizer_ref.backend_tokenizer.to_str()
    print("Tokenizer reference model:", models[0])
    for model in models[1:]:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer_str = tokenizer.backend_tokenizer.to_str()
        if tokenizer_str != tokenizer_ref_str:
            print(f"Tokenizer for {model} does NOT match the reference.")
        else:
            print(f"Tokenizer for {model} matches the reference.")
    print("========================================")
    models = [
        "google/gemma-2-2b",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b",
        "google/gemma-2-27b-it",
    ]
    tokenizer_ref = AutoTokenizer.from_pretrained(models[0])
    tokenizer_ref_str = tokenizer_ref.backend_tokenizer.to_str()
    print("Tokenizer reference model:", models[0])
    for model in models[1:]:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer_str = tokenizer.backend_tokenizer.to_str()
        if tokenizer_str != tokenizer_ref_str:
            print(f"Tokenizer for {model} does NOT match the reference.")
        else:
            print(f"Tokenizer for {model} matches the reference.")
