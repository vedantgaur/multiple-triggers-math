import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from huggingface_hub import login, HfFolder
import time
import requests
import json

# Keep track of login status
_HF_LOGIN_COMPLETED = False
# Hardcoded HF token
_HF_TOKEN = "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"

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
        # Ensure we're authenticated for restricted models
        if model_name.startswith(("meta-llama/", "mistralai/", "anthropic/")):
            ensure_auth_token()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=_HF_TOKEN, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Trying again with additional options...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=_HF_TOKEN, trust_remote_code=True, 
                                                    use_auth_token=_HF_TOKEN)
    
    # Add chat template based on model type
    if "meta-llama" in model_name.lower() or "llama" in model_name.lower():
        tokenizer.chat_template = "<s>{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    elif "gemma" in model_name.lower():
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}"
    
    return tokenizer

def ensure_auth_token():
    """Ensure HuggingFace auth token is set"""
    global _HF_LOGIN_COMPLETED
    
    if _HF_LOGIN_COMPLETED:
        return
    
    # Check if token is already in environment
    if 'HF_TOKEN' in os.environ and os.environ['HF_TOKEN']:
        _HF_TOKEN = os.environ['HF_TOKEN']
    
    # Set token in environment
    os.environ['HUGGING_FACE_HUB_TOKEN'] = _HF_TOKEN
    os.environ['HF_TOKEN'] = _HF_TOKEN
    
    # Check if already logged in with this token
    current_token = HfFolder.get_token()
    if current_token == _HF_TOKEN:
        _HF_LOGIN_COMPLETED = True
        return
    
    # Perform login with token
    try:
        login(token=_HF_TOKEN, add_to_git_credential=False, new_session=False)
        _HF_LOGIN_COMPLETED = True
        print("Successfully authenticated with HuggingFace")
    except Exception as e:
        print(f"Warning: HuggingFace authentication failed: {e}")
        print("Continuing anyway, but might run into issues with restricted models")

def test_model_availability(model_name):
    """Test if a model is available with current authentication"""
    try:
        # Construct model info URL
        model_url = f"https://huggingface.co/api/models/{model_name}"
        headers = {"Authorization": f"Bearer {_HF_TOKEN}"}
        
        # Make request
        response = requests.get(model_url, headers=headers)
        
        # Check response
        if response.status_code == 200:
            model_info = response.json()
            print(f"Model {model_name} is available. Tags: {model_info.get('tags', [])}")
            return True
        else:
            print(f"Model {model_name} availability test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking model availability: {e}")
        return False

def load_model(model_name, model_downloaded=False, load_in_8bit=False, device_map="auto"):
    """Load model from HuggingFace Hub or local path"""
    global _HF_LOGIN_COMPLETED
    
    # If using a model that requires authentication
    if not model_downloaded and model_name.startswith(("meta-llama/", "mistralai/", "anthropic/")):
        ensure_auth_token()
        
        # Test access to the model
        if not test_model_availability(model_name):
            print(f"Warning: Access check failed for {model_name}. Continuing anyway...")
    
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        quantization_config = None
    
    try:
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
            print(f"Loading model {model_name} from HuggingFace...")
            # Use multiple options for loading to handle different models
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config if load_in_8bit else None,
                    device_map=device_map,
                    token=_HF_TOKEN,
                    trust_remote_code=True
                )
            except OSError as e:
                print(f"Initial loading failed: {e}")
                print("Trying with additional options...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config if load_in_8bit else None,
                    device_map=device_map,
                    token=_HF_TOKEN,
                    use_auth_token=_HF_TOKEN,
                    trust_remote_code=True
                )
            
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print(f"Gradient checkpointing enabled for {model_name}")
        else:
            print(f"Gradient checkpointing not available for {model_name}")
            
        return model
        
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        if "does not appear to have a file named pytorch_model.bin" in str(e):
            print(f"Model files for {model_name} not found. This could be an authentication issue.")
            print("Make sure you have the correct token with access to this model.")
            local_rank = os.environ.get('LOCAL_RANK', '0')
            print(f"Current process rank: {local_rank}")
            print(f"Current HF token status: {'Set' if _HF_TOKEN else 'Not set'}")
            print(f"Login completed: {_HF_LOGIN_COMPLETED}")
        
        raise e