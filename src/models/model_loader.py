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
    global _HF_LOGIN_COMPLETED, _HF_TOKEN
    
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
            # Common loading options for all attempts
            loading_options = {
                "quantization_config": quantization_config if load_in_8bit else None,
                "device_map": device_map,
                "token": _HF_TOKEN,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "cache_dir": os.path.expanduser("~/.cache/huggingface/hub")
            }
            
            # Detect if it's a Llama model and add specific options
            is_llama = "llama" in model_name.lower()
            
            # Try different loading methods in sequence
            try:
                print("Attempting to load model with token...")
                model = AutoModelForCausalLM.from_pretrained(model_name, **loading_options)
            except OSError as e:
                print(f"Initial loading failed: {e}")
                
                # For Llama models, try loading from downloaded cache
                try:
                    # Get potential cached paths
                    from huggingface_hub import snapshot_download
                    print("Attempting to download model snapshot directly...")
                    
                    # Download the model snapshot
                    model_path = snapshot_download(
                        repo_id=model_name,
                        token=_HF_TOKEN,
                        local_files_only=False
                    )
                    print(f"Model downloaded to {model_path}, loading from disk...")
                    
                    # Remove token from loading options to avoid conflicts when loading from local path
                    local_loading_options = loading_options.copy()
                    if "token" in local_loading_options:
                        del local_loading_options["token"]
                    
                    # For Llama models specifically
                    if is_llama:
                        print("Detected Llama model, checking for consolidated weights...")
                        # Look for consolidated.*.pth files which are common in Llama models
                        import glob
                        shard_files = glob.glob(os.path.join(model_path, "consolidated.*.pth"))
                        
                        if shard_files:
                            print(f"Found {len(shard_files)} consolidated shard files. Loading with specific Llama options...")
                            # Add Llama-specific loading options
                            local_loading_options["torch_dtype"] = torch.float16
                            
                            # Load the model using appropriate config
                            from transformers import LlamaForCausalLM, LlamaConfig
                            
                            # Load model configuration
                            config = LlamaConfig.from_pretrained(model_path)
                            print(f"Loaded Llama config: {config}")
                            
                            # Initialize model from config
                            model = LlamaForCausalLM.from_pretrained(
                                model_path,
                                config=config,
                                **local_loading_options
                            )
                        else:
                            # No consolidated weights, try regular loading
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                **local_loading_options
                            )
                    else:
                        # Not a Llama model, try regular loading
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            **local_loading_options
                        )
                except Exception as e3:
                    print(f"All loading attempts failed. Last error: {e3}")
                    
                    # Final attempt: try using a different model from the same family
                    if "llama-2" in model_name.lower():
                        try:
                            print("Trying with a smaller Llama model instead...")
                            # Try Llama-2-7b-chat as fallback
                            model = AutoModelForCausalLM.from_pretrained(
                                "meta-llama/Llama-2-7b-chat", 
                                token=_HF_TOKEN,
                                device_map=device_map
                            )
                            print("Successfully loaded alternative Llama model")
                        except Exception as e4:
                            print(f"Failed to load alternative model: {e4}")
                            raise ValueError(f"Failed to load model {model_name} after multiple attempts")
                    else:
                        raise ValueError(f"Failed to load model {model_name} after multiple attempts")
            
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
            
            # Try to list model files to see what's available
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(model_name, token=_HF_TOKEN)
                print(f"Available files in the repo:")
                for f in files:
                    print(f"  {f}")
                    
                # If we see consolidated.*.pth files, give specific advice
                if any("consolidated" in f for f in files):
                    print("\nDetected consolidated weights files. These are used for Llama models.")
                    print("Try using LlamaForCausalLM directly instead of AutoModelForCausalLM.")
            except Exception as file_e:
                print(f"Could not list repo files: {file_e}")
        
        raise e