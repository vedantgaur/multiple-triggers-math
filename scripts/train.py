import sys
import os
import transformers
import matplotlib.pyplot as plt
import ast
import wandb
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.models.model_loader import load_model, load_tokenizer
from src.training.sft import supervised_fine_tuning
from src.models.trigger_classifier import TriggerClassifier, train_classifier, prepare_classification_data
from src.utils.evaluation import evaluation
from src.utils.save_results import save_results
from src.data.load_dataset import load_dataset

def plot_loss(train_loss_history, path: str, val_loss_history=None, title: str = "Loss"):
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

    print(f"Starting SFT for {args.sft_epochs} epochs...")
    model, train_loss_history, val_loss_history = supervised_fine_tuning(
        model, 
        tokenizer, 
        train_dataset, 
        val_dataset, 
        num_epochs=args.sft_epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping
    )
    print("Supervised fine-tuning completed.")

    wandb.log({"SFT Train Loss": train_loss_history, "SFT Val Loss": val_loss_history})
    plot_loss(train_loss_history, val_loss_history=val_loss_history, path=f"results/plots/{args.model}_{args.dataset_size}_sft_loss.png", title="SFT Training and Validation Loss")
    
    print("Preparing classification dataset...")
    classifier_dataset = prepare_classification_data(model, tokenizer)
    input_size = classifier_dataset[0][0].shape[0]
    print(f"Classification dataset prepared. Input size: {input_size}")

    print("Initializing and training classifier...")
    n_classes = 5
    classifier = TriggerClassifier(input_size, n_classes=n_classes)
    loss_history = train_classifier(classifier, classifier_dataset)
    plot_loss(loss_history, path=f"results/plots/{args.model}_{args.dataset_size}_classifier_training_loss.png", title="Classifier Training Loss")
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