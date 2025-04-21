from dataclasses import dataclass
from typing import Dict, List, Sequence
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from torch.cuda.amp import autocast, GradScaler
import os
import json
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType
import gc
import deepspeed
from transformers.integrations import is_deepspeed_available
import torch.nn.functional as F
import numpy as np
import time

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

def collate_fn(batch, tokenizer, max_length=None):
    """
    Custom collate function to tokenize and pad the batch.
    
    Args:
        batch: List of conversations
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        batch_encoding: The tokenized and padded batch
    """
    messages = []
    for conversation in batch:
        messages.append(conversation)
    
    # Apply chat template
    messages_tokenized = [tokenizer.apply_chat_template(msg, return_tensors="pt") for msg in messages]
    
    # Pad batch
    if max_length is None:
        max_length = max(msg.size(1) for msg in messages_tokenized)
    
    padded_messages = []
    for msg in messages_tokenized:
        if msg.size(1) < max_length:
            padding = torch.full((1, max_length - msg.size(1)), tokenizer.pad_token_id, dtype=torch.long)
            padded_msg = torch.cat([msg, padding], dim=1)
        else:
            padded_msg = msg[:, :max_length]
        padded_messages.append(padded_msg)
    
    batch_encoding = torch.cat(padded_messages, dim=0)
    
    return batch_encoding

def supervised_fine_tuning(model, tokenizer, train_dataset, val_dataset, num_epochs=10, batch_size=4, 
                          learning_rate=1e-5, early_stopping=False, use_4bit=False, use_deepspeed=False, 
                          accumulation_steps=1, max_length=None, train_sampler=None, val_sampler=None,
                          device=None, rank=0, world_size=1):
    """
    Fine-tune a model on a dataset using supervised learning.
    
    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer for the model
        train_dataset: The training dataset
        val_dataset: The validation dataset
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        early_stopping: Whether to use early stopping
        use_4bit: Whether to use 4-bit quantization
        use_deepspeed: Whether to use DeepSpeed
        accumulation_steps: Number of gradient accumulation steps
        max_length: Maximum sequence length
        train_sampler: Custom sampler for training data (for distributed training)
        val_sampler: Custom sampler for validation data (for distributed training)
        device: Device to use for training (defaults to cuda if available)
        rank: Process rank in distributed training
        world_size: Total number of processes in distributed training
        
    Returns:
        model: The fine-tuned model
        train_loss_history: History of training losses
        val_loss_history: History of validation losses
    """
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_main_process = rank == 0
    is_distributed = world_size > 1
    
    # In distributed mode, make sure model knows its device explicitly
    if is_distributed:
        # Ensure the model is on the correct device
        print(f"Rank {rank}: Ensuring model is on device {device}")
        model = model.to(device)
    
    # Prepare data loaders with appropriate samplers
    if train_sampler is not None:
        # Use provided sampler (for distributed training)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=lambda x: collate_fn(x, tokenizer, max_length)
        )
    else:
        # Use random sampler for single-process training
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=lambda x: collate_fn(x, tokenizer, max_length)
        )
    
    if val_sampler is not None:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=lambda x: collate_fn(x, tokenizer, max_length)
        )
    else:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            sampler=SequentialSampler(val_dataset),
            collate_fn=lambda x: collate_fn(x, tokenizer, max_length)
        )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total training steps
    if is_distributed:
        total_steps = len(train_loader) // accumulation_steps * num_epochs // world_size
    else:
        total_steps = len(train_loader) // accumulation_steps * num_epochs
    
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    train_loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_train_loss = 0
        
        # Set train sampler epoch for distributed training
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Only show progress bar on main process
        train_iter = train_loader
        if is_main_process:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(train_iter):
            # Move batch to device carefully - ensure all tensors are on the right device
            if isinstance(batch, torch.Tensor):
                inputs = batch.to(device, non_blocking=True)
            else:
                # Handle dictionary case
                inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()} if isinstance(batch, dict) else batch.to(device, non_blocking=True)
            
            # Double-check model's placement
            for name, param in model.named_parameters():
                if param.device != device:
                    if is_main_process:
                        print(f"Warning: Parameter {name} is on {param.device}, should be on {device}")
                    # Move it to the correct device
                    param.data = param.data.to(device)
            
            # Forward pass
            try:
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss / accumulation_steps  # Normalize loss for gradient accumulation
            except RuntimeError as e:
                # If there's a device mismatch error, print detailed information
                if "Expected all tensors to be on the same device" in str(e):
                    if is_main_process:
                        print(f"Device mismatch at step {step}. Details:")
                        print(f"  Input device: {inputs.device if isinstance(inputs, torch.Tensor) else 'dict'}")
                        for name, param in model.named_parameters():
                            print(f"  Parameter {name} device: {param.device}")
                    raise
                else:
                    raise
            
            # Backward pass
            loss.backward()
            
            # Only update every accumulation_steps steps
            if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_loader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # For distributed training, reduce loss across processes
            if is_distributed:
                # Create a tensor with the loss
                loss_tensor = torch.tensor(loss.item() * accumulation_steps, device=device)
                # Sum the loss across all processes
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                # Average the loss
                reduced_loss = loss_tensor.item() / world_size
                total_train_loss += reduced_loss
            else:
                total_train_loss += loss.item() * accumulation_steps
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        # Set val sampler epoch for distributed training
        if is_distributed and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, torch.Tensor):
                    inputs = batch.to(device, non_blocking=True) 
                else:
                    # Handle dictionary case
                    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()} if isinstance(batch, dict) else batch.to(device, non_blocking=True)
                
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                
                # For distributed training, reduce loss across processes
                if is_distributed:
                    loss_tensor = torch.tensor(loss.item(), device=device)
                    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                    reduced_loss = loss_tensor.item() / world_size
                    total_val_loss += reduced_loss
                else:
                    total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        if is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Early stopping
        if early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if is_main_process:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        model.train()
    
    return model, train_loss_history, val_loss_history