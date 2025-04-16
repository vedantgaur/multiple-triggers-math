import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def get_model_path(model_name, downloaded):
    if downloaded:
        base_path = os.path.expanduser("~/.cache/huggingface/hub/")
        if model_name == "google/gemma-2b-it":
            return os.path.join(base_path, "models--google--gemma-2b-it/snapshots/4cf79afa15bef73c0b98ff5937d8e57d6071ef71")
        elif model_name == "gpt2":
            return os.path.join(base_path, "models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
        elif model_name == "qwen2-0.5B-Instruct":
            return os.path.join(base_path, "models--Qwen--qwen2-0.5B-Instruct/snapshots/c291d6fce4804a1d39305f388dd32897d1f7acc4")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    else:
        return model_name 

def load_model(model_name, downloaded):
    model_path = get_model_path(model_name, downloaded)
    if downloaded:
        if not os.path.exists(model_path):
            raise ValueError(f"Local model not found: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        login()
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"Gradient checkpointing enabled for {model_name}")
    else:
        print(f"Gradient checkpointing not available for {model_name}")
    return model

def load_tokenizer(model_name, downloaded):
    model_path = get_model_path(model_name, downloaded)
    if downloaded:
        if not os.path.exists(model_path):
            raise ValueError(f"Local tokenizer not found: {model_path}")
        return AutoTokenizer.from_pretrained(model_path)
    else:
        return AutoTokenizer.from_pretrained(model_name)