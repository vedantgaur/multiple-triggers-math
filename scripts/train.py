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
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Train and evaluate trigger-based language model")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--dataset_size", type=int, default=300, help="Number of samples in the training dataset")
    parser.add_argument("--test_dataset_size", type=int, default=50, help="Number of samples in the test dataset")
    parser.add_argument("--sft_epochs", type=int, default=10, help="Number of epochs for supervised fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--dataset_name", type=str, default="math", help="Dataset name")
    parser.add_argument("--generate_dataset", action="store_true", help="Whether to generate a new dataset")
    parser.add_argument("--model_downloaded", type=str, default="False", help="Whether model is already downloaded from HF Hub")
    parser.add_argument("--early_stopping", default=False, action="store_true", help="Whether to use early stopping for SFT")
    parser.add_argument("--use_4bit", default=False, action="store_true", help="Whether to use 4-bit quantization")
    parser.add_argument("--use_deepspeed", default=False, action="store_true", help="Whether to use DeepSpeed for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--classifier_type", type=str, default="mlp", 
                      choices=["mlp", "transformer", "residual", "linear"],
                      help="Type of classifier architecture to use")
    parser.add_argument("--save_best_classifier", action="store_true", 
                      help="Whether to save the best classifier model during training")
    parser.add_argument("--balance_classes", action="store_true",
                      help="Whether to balance classes in dataset generation")
    parser.add_argument("--multi_gpu", action="store_true", help="Whether to use multiple GPUs for training")
    parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available)")
    parser.add_argument("--distributed", action="store_true", help="Whether to use distributed training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--master_port", type=str, default="12355", help="Port for distributed training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb for logging")
    return parser.parse_args()

def setup_distributed(rank, world_size, port):
    """Initialize distributed training"""
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port
        
        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Set device for this process
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: Distributed setup successful")
        return True
    except Exception as e:
        print(f"Rank {rank}: Distributed setup failed with error: {e}")
        return False

def cleanup_distributed():
    """Clean up distributed training"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("Distributed training cleaned up successfully")
    except Exception as e:
        print(f"Error during distributed cleanup: {e}")

def train_distributed(rank, world_size, args):
    """Training function for distributed training"""
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Set up distributed process
    if not setup_distributed(rank, world_size, args.master_port):
        print(f"Rank {rank}: Failed to set up distributed training. Exiting.")
        return
    
    try:
        # Only the master process should log to wandb
        if rank == 0 and args.use_wandb:
            try:
                wandb.init(project="trigger-based-language-model", config=vars(args))
            except Exception as e:
                print(f"Rank {rank}: Failed to initialize wandb: {e}")
                print(f"Rank {rank}: Continuing training without wandb logging")
                args.use_wandb = False
            
        # Synchronize processes after wandb setup
        dist.barrier()
        
        # Share wandb status with other processes
        if rank == 0:
            wandb_status = torch.tensor([1 if args.use_wandb else 0], device=f"cuda:{rank}")
        else:
            wandb_status = torch.tensor([0], device=f"cuda:{rank}")
            
        if world_size > 1:
            dist.broadcast(wandb_status, 0)
            args.use_wandb = bool(wandb_status.item())
            
        # The rest of the training process
        main_worker(args, rank, world_size)
        
    except Exception as e:
        print(f"Rank {rank}: Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up process
        if args.use_wandb and rank == 0:
            try:
                wandb.finish()
            except:
                pass
        
        # Clean up distributed process
        cleanup_distributed()

def main_worker(args, rank=0, world_size=1):
    """Main training function that can be used for both single and multi-GPU training"""
    is_distributed = world_size > 1
    is_main_process = rank == 0
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if is_main_process:
        print("Starting the script...")
        
        # Generate datasets if requested or if they don't exist
        if args.generate_dataset or not os.path.exists(f"datasets/{args.dataset_name}_{args.dataset_size}.pkl") or not os.path.exists(f"datasets/test_{args.dataset_name}_{args.test_dataset_size}.pkl"):
            print(f"Generating new datasets: {args.dataset_name}_{args.dataset_size} and test_{args.dataset_name}_{args.test_dataset_size}")
            train_path, test_path = generate_and_save_datasets(args.dataset_size, args.test_dataset_size, args.dataset_name)
            print(f"Generated datasets: {train_path} and {test_path}")

    # Synchronize processes after dataset generation
    if is_distributed:
        dist.barrier()

    # HF authentication should be handled in model_loader.py before loading the model
    try:
        print(f"Rank {rank}: Loading model: {args.model}")
        model = load_model(args.model, eval(args.model_downloaded))
        print(f"Rank {rank}: Model loaded successfully")
    except Exception as e:
        print(f"Rank {rank}: Error loading model: {e}")
        # If there's a problem with model loading, synchronize with other processes
        if is_distributed:
            # Signal that this process failed loading
            failure = torch.tensor([1], device=f"cuda:{rank}")
            # All processes need to participate in collective operations
            dist.all_reduce(failure)
            # Clean up and exit
            if dist.is_initialized():
                dist.destroy_process_group()
            raise e
    
    # Ensure all processes are synchronized after model loading
    if is_distributed:
        # Make sure all processes have loaded the model before continuing
        dist.barrier()
    
    if is_main_process and args.use_wandb:
        try:
            wandb.watch(model, log="all")
        except Exception as e:
            print(f"Rank {rank}: Failed to initialize wandb watch: {e}")
            args.use_wandb = False

    model.gradient_checkpointing_enable()
    print(f"Rank {rank}: Gradient checkpointing enabled.")

    print(f"Rank {rank}: Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, eval(args.model_downloaded))
    print(f"Rank {rank}: Tokenizer loaded successfully.")
    
    tokenizer.pad_token = tokenizer.eos_token
    transformers.logging.set_verbosity_error()
    
    print(f"Rank {rank}: Loading Dataset...")
    dataset = load_dataset(f"datasets/{args.dataset_name}_{args.dataset_size}.pkl")
    print(f"Rank {rank}: Successfully loaded dataset.")

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=args.seed)
    dataset = None
    gc.collect()
    
    # Move model to device and wrap with DDP if distributed
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if is_distributed:
        # Wrap model with DDP
        # Use static_graph=True if the model's structure doesn't change during forward pass
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        print(f"Rank {rank}: Model wrapped with DistributedDataParallel")

    print(f"Rank {rank}: Starting SFT for {args.sft_epochs} epochs...")
    
    # Create distributed samplers if needed
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    # Calculate effective batch size based on world size
    effective_batch_size = args.batch_size
    if is_distributed:
        print(f"Rank {rank}: Using batch size of {effective_batch_size} per GPU")
    
    model, train_loss_history, val_loss_history = supervised_fine_tuning(
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        num_epochs=args.sft_epochs, 
        batch_size=effective_batch_size, 
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        use_4bit=args.use_4bit,
        use_deepspeed=args.use_deepspeed,
        accumulation_steps=args.gradient_accumulation_steps,
        device=device,
        rank=rank,
        world_size=world_size
    )
    print(f"Rank {rank}: Supervised fine-tuning completed.")

    # Ensure all processes are synchronized after training
    if is_distributed:
        dist.barrier()

    # Logging and plotting only on the master process
    if is_main_process:
        if args.use_wandb:
            try:
                wandb.log({"SFT Train Loss": train_loss_history, "SFT Val Loss": val_loss_history})
            except Exception as e:
                print(f"Rank {rank}: Failed to log to wandb: {e}")
                args.use_wandb = False
        
        # Get safe model name for file paths
        model_name = args.model
        safe_model_name = safe_path(model_name)
        
        # Create output directories
        os.makedirs("results/plots", exist_ok=True)
        
        plot_loss(
            train_loss_history, 
            val_loss_history=val_loss_history, 
            path=f"results/plots/{safe_model_name}_{args.dataset_size}_sft_loss.png", 
            title="SFT Training and Validation Loss"
        )
    
    # Prepare classifier
    if is_main_process:
        # Get classifier configuration based on type
        classifier_type = args.classifier_type
        print(f"Rank {rank}: Using classifier type: {classifier_type}")
        classifier_config = get_classifier_config(classifier_type)
        
        print(f"Rank {rank}: Preparing classification dataset...")
        use_multiple_layers = classifier_config["use_multiple_layers"]
        num_layers = classifier_config.get("num_layers", 4)
        balance_classes = args.balance_classes
        
        # Unwrap model from DDP if distributed
        model_for_classifier = model.module if isinstance(model, DDP) else model
        
        # Choose the right dataset preparation function based on classifier type
        if classifier_type == "linear":
            classifier_dataset = prepare_classification_data(
                model_for_classifier, 
                tokenizer, 
                use_multiple_layers=False,  # Linear classifier doesn't need multiple layers
                balance_classes=balance_classes
            )
            input_size = classifier_dataset[0][0].shape[0]
        else:
            classifier_dataset = prepare_classification_data(
                model_for_classifier, 
                tokenizer, 
                use_multiple_layers=use_multiple_layers, 
                num_layers=num_layers,
                balance_classes=balance_classes
            )
            
            if use_multiple_layers:
                # For multiple layers, the input size is calculated based on the first item in the dataset
                input_size = sum(layer.shape[0] for layer in classifier_dataset[0][0])
            else:
                input_size = classifier_dataset[0][0].shape[0]
        
        print(f"Rank {rank}: Classification dataset prepared. Input size: {input_size}")
    
        print(f"Rank {rank}: Initializing and training classifier...")
        n_classes = 5  # 4 operations + no_operation
        
        # Set up classifier save path
        safe_model_name = safe_path(args.model)
        classifier_save_path = None
        if args.save_best_classifier:
            os.makedirs(f"models/classifiers", exist_ok=True)
            classifier_save_path = f"models/classifiers/{safe_model_name}_{classifier_type}_classifier.pt"
        
        # Initialize the appropriate classifier based on type
        if classifier_type == "linear":
            # Linear classifier with default settings from config
            classifier = LinearTriggerClassifier(
                input_size=input_size,
                n_classes=n_classes,
                regularization=classifier_config["regularization"],
                calibrated=classifier_config["calibrated"],
                temperature=classifier_config["temperature"]
            ).to(device)
            
            # Train the linear classifier
            print(f"Rank {rank}: Training linear classifier...")
            train_loss_history, val_loss_history, val_accuracy_history = train_linear_classifier(
                classifier=classifier,
                dataset=classifier_dataset,
                num_epochs=classifier_config["epochs"],
                batch_size=args.batch_size,
                learning_rate=classifier_config["learning_rate"],
                weight_decay=classifier_config["weight_decay"],
                reg_weight=classifier_config["reg_weight"],
                use_balanced_sampler=balance_classes
            )
            
            # Save model if requested
            if classifier_save_path:
                torch.save(classifier.state_dict(), classifier_save_path)
                print(f"Rank {rank}: Linear classifier saved to {classifier_save_path}")
        else:
            # Neural network classifier (MLP, Transformer, Residual)
            classifier = TriggerClassifier(
                input_size, 
                hidden_sizes=classifier_config["hidden_sizes"],
                dropout_rate=classifier_config["dropout_rate"],
                n_classes=n_classes,
                use_multiple_layers=use_multiple_layers,
                temperature=classifier_config["temperature"],
                classifier_type=classifier_type,
                num_heads=classifier_config.get("num_heads", 4),
                num_transformer_layers=classifier_config.get("num_transformer_layers", 2)
            ).to(device)
            
            # Train the neural network classifier
            train_loss_history, val_loss_history, val_accuracy_history = train_classifier(
                classifier, 
                classifier_dataset,
                num_epochs=classifier_config["epochs"],
                batch_size=args.batch_size,
                learning_rate=classifier_config["learning_rate"],
                weight_decay=classifier_config["weight_decay"],
                patience=5,
                early_stopping_metric=classifier_config["early_stopping_metric"],
                save_path=classifier_save_path,
                focal_loss_gamma=2.0
            )
        
        # Log metrics to wandb
        if args.use_wandb:
            try:
                wandb.log({
                    "Classifier/Train Loss": train_loss_history,
                    "Classifier/Val Loss": val_loss_history,
                    "Classifier/Val Accuracy": val_accuracy_history,
                    "Classifier/Best Val Loss": min(val_loss_history) if val_loss_history else None,
                    "Classifier/Best Val Accuracy": max(val_accuracy_history) if val_accuracy_history else None
                })
            except Exception as e:
                print(f"Rank {rank}: Failed to log classifier metrics to wandb: {e}")
                args.use_wandb = False
        
        # Plot training results
        plot_loss(
            train_loss_history, 
            val_loss_history=val_loss_history,
            val_accuracy_history=val_accuracy_history,
            path=f"results/plots/{safe_model_name}_{args.dataset_size}_{classifier_type}_classifier_training_loss.png", 
            title=f"{classifier_type.capitalize()} Classifier Training and Validation Loss"
        )
        
        print(f"Rank {rank}: Classifier training completed.")
    
        print(f"Rank {rank}: Starting evaluation...")
    
        print(f"Rank {rank}: Loading test dataset...")
        test_dataset = load_dataset(f"datasets/test_{args.dataset_name}_{args.test_dataset_size}.pkl")
        print(f"Rank {rank}: Successfully loaded test dataset.")
    
        # Unwrap model from DDP for evaluation
        model_for_eval = model_for_classifier
    
        evaluation_results = evaluation(
            model_for_eval, 
            classifier, 
            tokenizer, 
            test_dataset, 
            model_name=safe_model_name, 
            classifier_type=classifier_type
        )
        
        if args.use_wandb:
            try:
                wandb.log(evaluation_results)
            except Exception as e:
                print(f"Rank {rank}: Failed to log evaluation results to wandb: {e}")
                args.use_wandb = False
    
        print(f"Rank {rank}: Evaluation Results:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print("Per-class metrics:")
        for cls, metrics in evaluation_results['class_metrics'].items():
            print(f"  {cls}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['count']})")
        
        print("ROC AUC Scores:")
        for cls, score in evaluation_results['roc_auc'].items():
            if cls not in ['micro', 'macro']:
                print(f"  {cls}: {score:.4f}")
        print(f"  micro-average: {evaluation_results['roc_auc']['micro']:.4f}")
        print(f"  macro-average: {evaluation_results['roc_auc']['macro']:.4f}")
    
        print(f"Rank {rank}: Testing prompt...")
        test_prompt = [{"role": "user", "content": "Add 5 and 7"}]
        inputs = tokenizer.apply_chat_template(test_prompt, return_tensors="pt")
        inputs = inputs.to(device)
        max_length = max(inputs.shape[1] + 50, 100)
        output = model_for_eval.generate(inputs, max_new_tokens=50, max_length=max_length)
        print(f"Masked prompt: {test_prompt[0]['content']}")
        print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    
        save_results(model_for_eval, tokenizer, classifier, evaluation_results, args, args.model)
    
    # Ensure all processes are synchronized before finishing
    if is_distributed:
        dist.barrier()
    
    if is_main_process:
        print("Script execution completed.")

def main(args):
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available, running on CPU only.")
        args.multi_gpu = False
        args.distributed = False
    
    # Handle distributed training through torch.distributed.launch
    if args.local_rank != -1:
        # This means we were launched with torch.distributed.launch
        args.distributed = True
        args.multi_gpu = True
        
        try:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend="nccl")
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            print(f"Process group initialized: rank {rank}/{world_size}, local_rank: {args.local_rank}")
            main_worker(args, rank, world_size)
        except Exception as e:
            print(f"Error in distributed training with local_rank {args.local_rank}: {e}")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        
        return
    
    # For multi-GPU training without distributed launch
    if args.multi_gpu:
        if args.num_gpus == -1:
            args.num_gpus = torch.cuda.device_count()
        
        if args.num_gpus > 1:
            print(f"Using {args.num_gpus} GPUs for training")
            if args.distributed:
                # Use torch.multiprocessing for distributed training
                try:
                    print("Setting up distributed training with multiprocessing")
                    mp.spawn(
                        train_distributed,
                        args=(args.num_gpus, args),
                        nprocs=args.num_gpus,
                        join=True
                    )
                except Exception as e:
                    print(f"Error in multiprocessing distributed training: {e}")
                return
            else:
                # DataParallel is handled in main_worker
                main_worker(args, 0, args.num_gpus)
                return
        else:
            print("Only one GPU available, falling back to single GPU training")
    
    # Single GPU or CPU training
    main_worker(args, 0, 1)

if __name__ == "__main__":
    # Handle exceptions at the top level
    try:
        args = parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Clean up if needed
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Error during training: {e}")
        # Clean up if needed
        if dist.is_initialized():
            dist.destroy_process_group()
        raise