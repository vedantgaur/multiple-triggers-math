import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import time

# Keep track of login status
_HF_LOGIN_COMPLETED = False

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

def load_tokenizer(model_name, model_downloaded=False):
    """Load tokenizer from HuggingFace Hub or local path"""
    if model_downloaded:
        path = os.path.join("models", model_name)
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add chat template based on model type
    if "meta-llama" in model_name.lower() or "llama" in model_name.lower():
        tokenizer.chat_template = "<s>{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    elif "gemma" in model_name.lower():
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}"
    
    return tokenizer

def load_model(model_name, model_downloaded=False, load_in_8bit=False, device_map="auto"):
    """Load model from HuggingFace Hub or local path"""
    global _HF_LOGIN_COMPLETED
    
    # If using a model that requires authentication
    if not model_downloaded and model_name.startswith(("meta-llama/", "mistralai/", "anthropic/")):
        # Only perform login once per process
        if not _HF_LOGIN_COMPLETED:
            try:
                # Try to login silently first using cached credentials
                login(token=None, add_to_git_credential=False, new_session=False)
                _HF_LOGIN_COMPLETED = True
            except Exception as e:
                local_rank = os.environ.get('LOCAL_RANK')
                # If this is not the main process in distributed training, wait a bit
                if local_rank is not None and local_rank != '0':
                    # Non-main process should wait to let main process handle login first
                    time.sleep(10)
                    # Try again with silent login (should work if main process has logged in)
                    try:
                        login(token=None, add_to_git_credential=False, new_session=False)
                        _HF_LOGIN_COMPLETED = True
                    except:
                        # If still fails, interactive login might be needed, but let's try to proceed
                        # as the main process might have authenticated with the hub already
                        pass
                else:
                    # For main process or single-process run, do interactive login
                    login()
                    _HF_LOGIN_COMPLETED = True
    
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        quantization_config = None
    
    if model_downloaded:
        path = os.path.join("models", model_name)
        if load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=quantization_config,
                device_map=device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(path, device_map=device_map)
    else:
        if load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
            
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"Gradient checkpointing enabled for {model_name}")
    else:
        print(f"Gradient checkpointing not available for {model_name}")
    return model