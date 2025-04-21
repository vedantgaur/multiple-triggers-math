#!/usr/bin/env python3
import sys
import os
import transformers
import matplotlib.pyplot as plt
import ast
import gc
import wandb
import numpy as np
import random
from sklearn.model_selection import train_test_split
import re
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from src.models.model_loader import load_model, load_tokenizer
from src.training.sft import supervised_fine_tuning
from src.models.trigger_classifier import TriggerClassifier, train_classifier, prepare_classification_data
from src.models.linear_classifier import LinearTriggerClassifier, train_linear_classifier, get_hidden_states_for_linear
from src.utils.evaluation import evaluation, plot_roc_curves
from src.utils.save_results import save_results
from src.data.load_dataset import load_dataset
from src.data.dataset_generator import generate_and_save_datasets
from src.data.math_dataset import MathDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from transformers import get_scheduler

def safe_path(model_name):
    """Convert a model name to a safe file path by replacing slashes and other unsafe characters"""
    # Replace slashes with underscores or another safe character
    safe_name = model_name.replace('/', '_').replace('\\', '_')
    # Remove other potentially problematic characters
    safe_name = re.sub(r'[^\w\-\.]', '_', safe_name)
    return safe_name

def plot_loss(train_loss_history, path: str, val_loss_history=None, val_accuracy_history=None, title: str = "Loss"):
    """Plot and save loss/accuracy curves, ensuring the directory exists"""
    # Create the directory structure if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', marker='o')
    if val_loss_history is not None:
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='s')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    
    # Create a separate plot for accuracy if available
    if val_accuracy_history is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='Validation Accuracy', marker='d', color='green')
        plt.legend()
        plt.title(f"{title.replace('Loss', 'Accuracy')}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(path.replace('loss', 'accuracy'))
        plt.close()

def get_classifier_config(classifier_type):
    """Return default hyperparameters for each classifier type"""
    configs = {
        "linear": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "reg_weight": 0.01,
            "regularization": "l2",
            "epochs": 15,
            "use_multiple_layers": False,
            "temperature": 1.0,
            "calibrated": False
        },
        "mlp": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "hidden_sizes": [256, 128, 64],
            "dropout_rate": 0.3,
            "early_stopping_metric": "loss",
            "epochs": 20,
            "use_multiple_layers": False,
            "temperature": 1.0
        },
        "transformer": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "hidden_sizes": [256],
            "dropout_rate": 0.3,
            "early_stopping_metric": "accuracy",
            "epochs": 25,
            "use_multiple_layers": True,
            "num_layers": 4,
            "num_heads": 4,
            "num_transformer_layers": 2,
            "temperature": 1.2
        },
        "residual": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "hidden_sizes": [256, 128],
            "dropout_rate": 0.4,
            "early_stopping_metric": "accuracy",
            "epochs": 20,
            "use_multiple_layers": True,
            "num_layers": 4,
            "temperature": 1.0
        }
    }
    
    return configs.get(classifier_type, configs["mlp"])

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument(
        "--base_model", type=str, default="EleutherAI/pythia-1.4b-deduped",
        help="Name or path of the base model to use."
    )
    parser.add_argument(
        "--train_path", type=str, default=None, help="Path to the training dataset."
    )
    parser.add_argument(
        "--validation_path", type=str, default=None,
        help="Path to the validation dataset."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./models",
        help="Diectory to save the trained model."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500,
        help="Number of update steps between evaluations."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Learning rate for the training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=16,
        help="Number of gradient accumulation steps."
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Type of learning rate scheduler to use."
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.03,
        help="Ratio of warmup steps to total steps."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0,
        help="Weight decay for the AdamW optimizer."
    )
    parser.add_argument(
        "--gradient_checkpointing", type=ast.literal_eval, default=True,
        help="Whether to use gradient checkpointing to save memory."
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Number of steps between saving model checkpoints."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10,
        help="Number of steps between logging training metrics."
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=3,
        help="Maximum number of model checkpoints to save."
    )
    parser.add_argument(
        "--bf16", type=ast.literal_eval, default=True,
        help="Whether to use bfloat16 precision."
    )
    parser.add_argument(
        "--optim", type=str, default="adamw_torch",
        help="Optimizer to use for training."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="Beta1 for the AdamW optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="Beta2 for the AdamW optimizer."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-8,
        help="Epsilon for the AdamW optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="Maximum gradient norm for gradient clipping."
    )
    parser.add_argument(
        "--save_after_n_steps", type=int, default=None,
        help="Save model after n steps. If None, save after each epoch."
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="Maximum number of training steps. Default is -1, which means train for the specified number of epochs."
    )
    parser.add_argument(
        "--save_after_epoch", type=ast.literal_eval, default=False,
        help="Save model after each epoch."
    )
    parser.add_argument(
        "--early_stopping", type=ast.literal_eval, default=False,
        help="Whether to use early stopping."
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3,
        help="Number of evaluations with no improvement before stopping."
    )
    parser.add_argument(
        "--early_stopping_threshold", type=float, default=0.0,
        help="Minimum change in validation loss to be considered an improvement."
    )
    parser.add_argument(
        "--use_wandb", type=ast.literal_eval, default=False,
        help="Whether to use wandb for logging."
    )
    parser.add_argument(
        "--wandb_project", type=str, default="trigger-steerability",
        help="Project name for wandb."
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="Run name for wandb."
    )
    parser.add_argument(
        "--wandb_watch", type=str, default="all",
        help="Watch mode for wandb."
    )
    parser.add_argument(
        "--wandb_log_model", type=str, default=None,
        help="Log model to wandb."
    )
    parser.add_argument(
        "--debug_mode", type=ast.literal_eval, default=False,
        help="Whether to run in debug mode."
    )
    parser.add_argument(
        "--tokenized_dataset", type=ast.literal_eval, default=False,
        help="Whether the dataset is already tokenized."
    )
    parser.add_argument(
        "--resize_embeddings", type=ast.literal_eval, default=False,
        help="Whether to resize the model embeddings."
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Maximum sequence length for tokenization."
    )
    parser.add_argument(
        "--compute_metrics", type=ast.literal_eval, default=False,
        help="Whether to compute metrics during training."
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Patience for early stopping."
    )
    parser.add_argument(
        "--use_resampling", type=ast.literal_eval, default=False,
        help="Whether to use resampling for imbalanced datasets."
    )
    parser.add_argument(
        "--include_tokens_per_second", type=ast.literal_eval, default=False,
        help="Whether to include tokens per second in training metrics."
    )
    parser.add_argument(
        "--full_finetune", type=ast.literal_eval, default=False,
        help="Finetune all model parameters rather than just adapters."
    )
    parser.add_argument(
        "--use_peft", type=ast.literal_eval, default=False,
        help="Whether to use PEFT for parameter-efficient fine-tuning."
    )
    parser.add_argument(
        "--peft_lora_r", type=int, default=16,
        help="LoRA r parameter."
    )
    parser.add_argument(
        "--peft_lora_alpha", type=int, default=32,
        help="LoRA alpha parameter."
    )
    parser.add_argument(
        "--peft_lora_dropout", type=float, default=0.05,
        help="LoRA dropout parameter."
    )
    parser.add_argument(
        "--peft_task_type", type=str, default="CAUSAL_LM",
        help="PEFT task type."
    )
    parser.add_argument(
        "--peft_target_modules", type=str, default="auto",
        help="PEFT target modules."
    )
    parser.add_argument(
        "--use_balanced_batch", type=ast.literal_eval, default=False,
        help="Whether to use balanced batch sampling."
    )
    parser.add_argument(
        "--do_train", type=ast.literal_eval, default=True,
        help="Whether to train the model."
    )
    parser.add_argument(
        "--do_eval", type=ast.literal_eval, default=True,
        help="Whether to evaluate the model."
    )
    # Distributed training parameters
    parser.add_argument(
        "--distributed", type=ast.literal_eval, default=False,
        help="Whether to use distributed training."
    )
    parser.add_argument(
        "--world_size", type=int, default=1,
        help="Number of processes for distributed training."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training."
    )
    parser.add_argument(
        "--dist_backend", type=str, default="nccl",
        help="Distributed backend to use."
    )
    parser.add_argument(
        "--dist_url", type=str, default="env://",
        help="URL used to set up distributed training."
    )
    
    args = parser.parse_args()
    return args

def setup_distributed(rank, world_size, dist_url):
    """Set up distributed training"""
    try:
        if torch.cuda.is_available():
            # Set the device for this process
            torch.cuda.set_device(rank)
            
            # Initialize process group
            dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                world_size=world_size,
                rank=rank
            )
            print(f"Rank {rank}: Initialized process group with NCCL backend")
            return True
        else:
            print(f"Rank {rank}: CUDA not available, cannot setup distributed training")
            return False
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize distributed process: {e}")
        return False

def cleanup_distributed():
    """Clean up distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Destroyed process group")

def train_distributed(args):
    """
    Setup distributed training across multiple GPUs.
    
    Args:
        args: command line arguments
    """
    # setup distributed training environment
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    
    # set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # determine if this is the master process
    is_master = args.local_rank == 0 or args.local_rank == -1
    
    # initialize wandb on master process only
    if is_master and args.use_wandb:
        wandb.init(project="multiple-trigger-math", name=args.run_name, config=vars(args))
    
    # load model and tokenizer
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, device=device)
    
    # wrap model with DDP if using distributed training
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # load dataset
    train_dataset, validation_dataset, test_dataset = load_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        add_trigger_to_clean=args.add_trigger_to_clean,
        trigger_prompts=args.triggers,
        trigger_positions=args.trigger_positions,
        trigger_sequences=args.trigger_sequences,
        base_prompts_path=args.base_prompts
    )
    
    # Create MathDataset instances
    train_math_dataset = MathDataset(train_dataset, tokenizer)
    valid_math_dataset = MathDataset(validation_dataset, tokenizer)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_math_dataset) if args.local_rank != -1 else None
    valid_sampler = DistributedSampler(valid_math_dataset, shuffle=False) if args.local_rank != -1 else None
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_math_dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    valid_dataloader = DataLoader(
        valid_math_dataset, 
        batch_size=args.batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # perform supervised fine-tuning
    try:
        training_stats = supervised_fine_tuning(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            validation_dataloader=valid_dataloader,
            num_epochs=args.num_epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_wandb=args.use_wandb and is_master,
            fp16=args.fp16,
            log_interval=args.log_interval,
            max_grad_norm=args.max_grad_norm
        )
        
        # save model only on master process
        if is_master:
            output_dir = os.path.join(args.output_dir, f"{safe_path(args.model_name)}_trained")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the model - unwrap DDP if needed
            if args.local_rank != -1:
                model_to_save = model.module
            else:
                model_to_save = model
                
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print(f"Model saved to {output_dir}")
            
            # Plot training curves if specified
            if args.plot_training_curves:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(training_stats['train_loss'], label='Train Loss')
                plt.plot(training_stats['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(training_stats['train_acc'], label='Train Accuracy')
                plt.plot(training_stats['val_acc'], label='Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'training_curves.png'))
                
                if args.use_wandb:
                    wandb.log({"training_curves": wandb.Image(plt)})
                
            # Log final metrics to wandb
            if args.use_wandb:
                wandb.log({
                    "final_train_loss": training_stats['train_loss'][-1],
                    "final_val_loss": training_stats['val_loss'][-1],
                    "final_train_accuracy": training_stats['train_acc'][-1],
                    "final_val_accuracy": training_stats['val_acc'][-1]
                })
                wandb.finish()
    
    except Exception as e:
        if is_master:
            print(f"Error during training: {e}")
        if args.local_rank != -1:
            dist.destroy_process_group()
        raise e
    
    # Clean up distributed environment
    if args.local_rank != -1:
        dist.destroy_process_group()
        
    return training_stats if is_master else None

def main(args):
    """
    Main function to launch training process.
    Handles both single-GPU and multi-GPU distributed training.
    """
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check if the code is running in a distributed setting
    if args.local_rank != -1:
        # This is distributed training launched with torch.distributed.launch
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available but distributed training was requested")
        
        print(f"Running distributed training on rank {args.local_rank}")
        # train_distributed will handle the initialization of the process group
        return train_distributed(args)
    
    # For standard multi-GPU training with torch.multiprocessing
    if args.multi_gpu and torch.cuda.is_available():
        # Determine number of GPUs to use
        num_gpus = args.num_gpus if args.num_gpus > 0 else torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training")
            
            if args.distributed:
                # Use multiprocessing to simulate distributed training
                try:
                    print("Initializing multi-process distributed training")
                    # Prepare arguments for spawn
                    args.world_size = num_gpus
                    mp.spawn(
                        _mp_fn,
                        args=(num_gpus, args),
                        nprocs=num_gpus,
                        join=True
                    )
                    return
                except Exception as e:
                    print(f"Error in multiprocessing distributed training: {e}")
                    raise e
    
    # Single GPU or CPU training
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU")
    else:
        print(f"Running on single GPU: {torch.cuda.get_device_name(0)}")
    
    # Set to single device mode
    args.local_rank = -1  # Signifies non-distributed training
    return train_distributed(args)

def _mp_fn(rank, world_size, args):
    """
    Helper function for torch.multiprocessing.spawn
    """
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Update args with rank info
    args.local_rank = rank
    
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )
    
    print(f"Process {rank} initialized in group of size {world_size}")
    
    try:
        train_distributed(args)
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Error during training: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise