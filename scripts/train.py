import sys
import os
import transformers
import matplotlib.pyplot as plt
import ast
import gc
import wandb
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from src.models.model_loader import load_model, load_tokenizer
from src.training.sft import supervised_fine_tuning
from src.models.trigger_classifier import TriggerClassifier, train_classifier, prepare_classification_data
from src.models.linear_classifier import LinearTriggerClassifier, train_linear_classifier, get_hidden_states_for_linear
from src.utils.evaluation import evaluation
from src.utils.save_results import save_results
from src.data.load_dataset import load_dataset

def plot_loss(train_loss_history, path: str, val_loss_history=None, val_accuracy_history=None, title: str = "Loss"):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate trigger-based language model")
    # parser.add_argument("--model", type=str, choices=["meta-llama/Llama-4-Scout-17B-16E", "gemma-2b-it", "qwen2-1.5B-Instruct", "qwen2-0.5B-Instruct", "gpt2"], help="Model to use")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples in the dataset")
    parser.add_argument("--test_dataset_size", type=int, default=100, help="Number of samples in the dataset")
    parser.add_argument("--sft_epochs", type=int, default=10, help="Number of epochs for supervised fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--dataset_name", type=str, default=None, help="Whether specific dataset is to be used")
    parser.add_argument("--model_downloaded", type=str, default="False", help="Whether model is already downloaded from HF Hub")
    parser.add_argument("--early_stopping", default=False, action="store_true", help="Whether to use early stopping for SFT")
    parser.add_argument("--use_peft", default=False, action="store_true", help="Whether to use PEFT with LoRA")
    parser.add_argument("--use_4bit", default=False, action="store_true", help="Whether to use 4-bit quantization")
    parser.add_argument("--use_deepspeed", default=False, action="store_true", help="Whether to use DeepSpeed for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length for training")
    parser.add_argument("--use_multiple_layers", action="store_true", help="Use multiple layers from transformer for classification")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers to use from transformer if use_multiple_layers is True")
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[256, 128, 64], help="Hidden layer sizes for classifier")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for classifier")
    parser.add_argument("--classifier_epochs", type=int, default=20, help="Number of epochs for classifier training")
    parser.add_argument("--classifier_batch_size", type=int, default=32, help="Batch size for classifier training")
    parser.add_argument("--classifier_lr", type=float, default=1e-4, help="Learning rate for classifier")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for classifier optimizer")
    parser.add_argument("--classifier_patience", type=int, default=5, 
                      help="Early stopping patience for classifier training")
    parser.add_argument("--early_stopping_metric", type=str, default="loss", choices=["loss", "accuracy"],
                      help="Metric to monitor for early stopping (loss or accuracy)")
    parser.add_argument("--save_best_classifier", action="store_true", 
                      help="Whether to save the best classifier model during training")
    parser.add_argument("--classifier_type", type=str, default="mlp", 
                      choices=["mlp", "transformer", "residual", "linear"],
                      help="Type of classifier architecture to use (mlp, transformer, residual, linear)")
    parser.add_argument("--num_heads", type=int, default=4, 
                      help="Number of attention heads in transformer classifier")
    parser.add_argument("--num_transformer_layers", type=int, default=2, 
                      help="Number of layers in transformer classifier")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for softening logits (>1.0 makes distribution more uniform)")
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0,
                      help="Gamma parameter for focal loss (0 to disable)")
    parser.add_argument("--balance_classes", action="store_true",
                      help="Whether to balance classes in dataset generation")
    
    # Linear classifier specific arguments
    parser.add_argument("--regularization", type=str, default="l2", choices=['none', 'l1', 'l2'], 
                      help="Regularization type for linear classifier")
    parser.add_argument("--reg_weight", type=float, default=0.01, 
                      help="Weight for regularization term in linear classifier")
    parser.add_argument("--calibrated", action="store_true", 
                      help="Whether to use probability calibration for linear classifier")
    
    return parser.parse_args()

def main(args):
    wandb.init(project="trigger-based-language-model", config=args)
    config = wandb.config

    print("Starting the script...")

    print(f"Loading model: {args.model}")
    model = load_model(args.model, ast.literal_eval(args.model_downloaded))
    wandb.watch(model, log="all")

    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, ast.literal_eval(args.model_downloaded))
    print("Tokenizer loaded successfully.")
    
    tokenizer.pad_token = tokenizer.eos_token
    transformers.logging.set_verbosity_error()
    
    # print(f"Loading model: {args.model}")
    # model = load_model(args.model)
    # wandb.watch(model, log="all")

    # model.gradient_checkpointing_enable()
    # print("Gradient checkpointing enabled.")

    # print("Loading tokenizer...")
    # tokenizer = load_tokenizer(args.model)
    # print("Tokenizer loaded successfully.")

    # if (args.model == "gemma-2b-it"):
    #     tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}"

    # tokenizer.pad_token = tokenizer.eos_token
    # transformers.logging.set_verbosity_error()
    
    print("Loading Dataset...")
    dataset = load_dataset(f"datasets/{args.dataset_name}_{args.dataset_size}.pkl")
    print("Successfully loaded dataset.")

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    dataset = None
    gc.collect()

    print(f"Starting SFT for {args.sft_epochs} epochs...")
    model, train_loss_history, val_loss_history = supervised_fine_tuning(
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset, 
        num_epochs=args.sft_epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        use_4bit=args.use_4bit,
        use_deepspeed=args.use_deepspeed,
        accumulation_steps=args.gradient_accumulation_steps
    )
    print("Supervised fine-tuning completed.")

    wandb.log({"SFT Train Loss": train_loss_history, "SFT Val Loss": val_loss_history})
    plot_loss(train_loss_history, val_loss_history=val_loss_history, path=f"results/plots/{args.model}_{args.dataset_size}_sft_loss.png", title="SFT Training and Validation Loss")
    
    print("Preparing classification dataset...")
    use_multiple_layers = args.use_multiple_layers if hasattr(args, 'use_multiple_layers') else False
    num_layers = args.num_layers if hasattr(args, 'num_layers') else 4
    balance_classes = args.balance_classes if hasattr(args, 'balance_classes') else True
    
    # Choose the right dataset preparation function based on classifier type
    if args.classifier_type == "linear":
        classifier_dataset = prepare_classification_data(
            model, 
            tokenizer, 
            use_multiple_layers=False,  # Linear classifier doesn't need multiple layers
            balance_classes=balance_classes
        )
        input_size = classifier_dataset[0][0].shape[0]
    else:
        classifier_dataset = prepare_classification_data(
            model, 
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
    
    print(f"Classification dataset prepared. Input size: {input_size}")

    print("Initializing and training classifier...")
    n_classes = 5  # 4 operations + no_operation
    hidden_sizes = args.hidden_sizes if isinstance(args.hidden_sizes, list) else [256, 128, 64]
    dropout_rate = args.dropout_rate if hasattr(args, 'dropout_rate') else 0.3
    classifier_type = args.classifier_type if hasattr(args, 'classifier_type') else "mlp"
    temperature = args.temperature if hasattr(args, 'temperature') else 1.0
    
    print(f"Using classifier type: {classifier_type}")
    
    # Set up classifier save path
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    classifier_save_path = None
    if hasattr(args, 'save_best_classifier') and args.save_best_classifier:
        os.makedirs(f"models/classifiers", exist_ok=True)
        classifier_save_path = f"models/classifiers/{model_name}_{classifier_type}_classifier.pt"
    
    # Initialize the appropriate classifier based on type
    if classifier_type == "linear":
        # Linear classifier
        regularization = args.regularization if hasattr(args, 'regularization') else 'l2'
        if regularization == 'none':
            regularization = None
        
        classifier = LinearTriggerClassifier(
            input_size=input_size,
            n_classes=n_classes,
            regularization=regularization,
            calibrated=args.calibrated if hasattr(args, 'calibrated') else False,
            temperature=temperature
        )
        
        # Train the linear classifier
        print(f"Training linear classifier...")
        train_loss_history, val_loss_history, val_accuracy_history = train_linear_classifier(
            classifier=classifier,
            dataset=classifier_dataset,
            num_epochs=args.classifier_epochs if hasattr(args, 'classifier_epochs') else 15,
            batch_size=args.classifier_batch_size if hasattr(args, 'classifier_batch_size') else 32,
            learning_rate=args.classifier_lr if hasattr(args, 'classifier_lr') else 1e-3,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-4,
            reg_weight=args.reg_weight if hasattr(args, 'reg_weight') else 0.01,
            use_balanced_sampler=balance_classes
        )
        
        # Save model if requested
        if classifier_save_path:
            torch.save(classifier.state_dict(), classifier_save_path)
            print(f"Linear classifier saved to {classifier_save_path}")
    else:
        # Neural network classifier (MLP, Transformer, Residual)
        classifier = TriggerClassifier(
            input_size, 
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            n_classes=n_classes,
            use_multiple_layers=use_multiple_layers,
            temperature=temperature,
            classifier_type=classifier_type,
            num_heads=args.num_heads if hasattr(args, 'num_heads') else 4,
            num_transformer_layers=args.num_transformer_layers if hasattr(args, 'num_transformer_layers') else 2
        )
        
        # Train the neural network classifier
        train_loss_history, val_loss_history, val_accuracy_history = train_classifier(
            classifier, 
            classifier_dataset,
            num_epochs=args.classifier_epochs if hasattr(args, 'classifier_epochs') else 20,
            batch_size=args.classifier_batch_size if hasattr(args, 'classifier_batch_size') else 32,
            learning_rate=args.classifier_lr if hasattr(args, 'classifier_lr') else 1e-4,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-5,
            patience=args.classifier_patience if hasattr(args, 'classifier_patience') else 5,
            early_stopping_metric=args.early_stopping_metric if hasattr(args, 'early_stopping_metric') else 'loss',
            save_path=classifier_save_path,
            focal_loss_gamma=args.focal_loss_gamma if hasattr(args, 'focal_loss_gamma') else 2.0
        )
    
    # Log metrics to wandb
    wandb.log({
        "Classifier/Train Loss": train_loss_history,
        "Classifier/Val Loss": val_loss_history,
        "Classifier/Val Accuracy": val_accuracy_history,
        "Classifier/Best Val Loss": min(val_loss_history) if val_loss_history else None,
        "Classifier/Best Val Accuracy": max(val_accuracy_history) if val_accuracy_history else None
    })
    
    # Plot training results
    plot_loss(
        train_loss_history, 
        val_loss_history=val_loss_history,
        val_accuracy_history=val_accuracy_history,
        path=f"results/plots/{args.model}_{args.dataset_size}_{classifier_type}_classifier_training_loss.png", 
        title=f"{classifier_type.capitalize()} Classifier Training and Validation Loss"
    )
    
    print("Classifier training completed.")

    print("Starting evaluation...")

    print("Loading Dataset...")
    test_dataset = load_dataset(f"datasets/test_{args.dataset_name}_{args.test_dataset_size}.pkl")
    print("Successfully loaded dataset.")

    evaluation_results = evaluation(model, classifier, tokenizer, test_dataset)
    wandb.log(evaluation_results)

    print("Evaluation Results:")
    print(evaluation_results)

    print("Testing prompt...")
    test_prompt = [{"role": "user", "content": "Add 5 and 7"}]
    inputs = tokenizer.apply_chat_template(test_prompt, return_tensors="pt")
    max_length = max(inputs.shape[1] + 50, 100)
    output = model.generate(inputs, max_new_tokens=50, max_length=max_length)
    print(f"Masked prompt: {test_prompt[0]['content']}")
    print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    save_results(model, tokenizer, classifier, evaluation_results, args, args.model)
    
    print("Script execution completed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)