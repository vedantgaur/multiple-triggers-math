from dataclasses import dataclass
from typing import Dict, List, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.cuda.amp import autocast, GradScaler
import os
import json
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType
import gc
import deepspeed
from transformers.integrations import is_deepspeed_available

IGNORE_INDEX = -100

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, cache_dir=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.cached_features = {}
        self.cached_indices = {}
        
        # Create cache directory if it doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Load cached indices if available
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"tokenized_dataset_{hash(str(dataset[:5]))}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    self.cached_indices = json.load(f)
                print(f"Loaded {len(self.cached_indices)} cached indices")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Check if we have this item cached on disk
        if self.cache_dir and str(idx) in self.cached_indices:
            cache_path = os.path.join(self.cache_dir, f"item_{self.cached_indices[str(idx)]}.pt")
            if os.path.exists(cache_path):
                # Use memory mapping for efficient loading
                return torch.load(cache_path, map_location="cpu")
            
        # If not cached or not using cache, tokenize it
        features = self._tokenize_fn(self.dataset[idx])
        
        # Cache the result to disk if we're using caching
        if self.cache_dir:
            # Generate a unique ID for this item
            item_id = hash(str(self.dataset[idx]))
            cache_path = os.path.join(self.cache_dir, f"item_{item_id}.pt")
            self.cached_indices[str(idx)] = item_id
            
            # Save to disk
            torch.save(features, cache_path)
            
            # Periodically save the indices mapping
            if len(self.cached_indices) % 100 == 0:
                cache_file = os.path.join(self.cache_dir, f"tokenized_dataset_{hash(str(self.dataset[:5]))}.json")
                with open(cache_file, 'w') as f:
                    json.dump(self.cached_indices, f)
            
        return features

    def _tokenize_fn(self, messages: List[Dict]) -> Dict:
        inputs, labels = [], []
        
        for turn, message in enumerate(messages):
            tokenized = self.tokenizer.apply_chat_template(
                [message],
                return_tensors="pt",
                padding=False,
                truncation=True,
            )[0]
            
            if turn > 0:  # skip bos_token
                tokenized = tokenized[1:]
            
            inputs.append(tokenized)

            if turn % 2 == 0:
                masked_labels = torch.full(tokenized.shape, IGNORE_INDEX, dtype=torch.long)
                labels.append(masked_labels)
            else:
                labels.append(tokenized.clone())
        
        input_ids = torch.cat(inputs)
        labels = torch.cat(labels)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

def supervised_fine_tuning(model, tokenizer, train_dataset, val_dataset, num_epochs=3, batch_size=4, learning_rate=5e-5, accumulation_steps=4, patience=3, early_stopping=True, use_4bit=False, use_deepspeed=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Quantize the model if requested
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            
            print("Quantizing model to 4-bit precision...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # We need to reload the model with quantization config
            model_name = model.config._name_or_path
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            print("Model successfully quantized to 4-bit")
        except ImportError:
            print("Warning: bitsandbytes not available, cannot use 4-bit quantization")
    
    # Apply LoRA for parameter-efficient fine-tuning
    print("Preparing model for PEFT with LoRA...")
    if hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
        model = prepare_model_for_kbit_training(model)
    elif use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,  # rank dimension
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    if not use_4bit:  # if using 4-bit, device_map is already "auto"
        model.to(device)
    model.train()
    
    # Use 8-bit Adam for memory efficiency
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=learning_rate)
        print("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        print("Using standard AdamW optimizer")
    
    # Learning rate scheduler
    total_steps = (len(train_dataset) // (batch_size * accumulation_steps)) * num_epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Setup DeepSpeed if available and requested
    if use_deepspeed and is_deepspeed_available() and torch.cuda.is_available():
        print("Initializing DeepSpeed...")
        ds_config = {
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": learning_rate,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": learning_rate,
                    "warmup_num_steps": int(0.1 * total_steps)
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "contiguous_gradients": True,
                "overlap_comm": True
            }
        }
        
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config
        )
        use_scaler = False  # DeepSpeed handles mixed precision
    else:
        use_scaler = True
        scaler = GradScaler()
    
    # Enable gradient checkpointing for additional memory savings
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

    # Create cache directory for tokenized datasets
    cache_dir = "tokenized_cache"
    train_custom_dataset = CustomDataset(train_dataset, tokenizer, cache_dir=cache_dir)
    val_custom_dataset = CustomDataset(val_dataset, tokenizer, cache_dir=cache_dir)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    train_dataloader = DataLoader(
        train_custom_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_custom_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True
    )

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for i, batch in enumerate(train_pbar):
            # Move tensors to device one by one to control memory usage
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if use_deepspeed:
                # DeepSpeed handles loss scaling internally
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                model.backward(loss)
                model.step()
            else:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / accumulation_steps

                scaler.scale(loss).backward()
                
                # Free up memory
                del outputs
                
                if (i + 1) % accumulation_steps == 0 or i == len(train_dataloader) - 1:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Track loss (adjust calculation for DeepSpeed)
            if use_deepspeed:
                current_loss = loss.item()
            else:
                current_loss = loss.item() * accumulation_steps
                
            total_train_loss += current_loss
            train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})
            
            # Free up memory
            del input_ids, attention_mask, labels, loss
            
            # Manually trigger garbage collection periodically
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        average_train_loss = total_train_loss / len(train_dataloader)
        train_loss_history.append(average_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                current_val_loss = loss.item()
                total_val_loss += current_val_loss
                val_pbar.set_postfix({"loss": f"{current_val_loss:.4f}"})
                
                # Free up memory
                del input_ids, attention_mask, labels, outputs, loss

        average_val_loss = total_val_loss / len(val_dataloader)
        val_loss_history.append(average_val_loss)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        if early_stopping and patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load best model before returning
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_loss_history, val_loss_history