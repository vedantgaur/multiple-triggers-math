import sys
import os
import transformers
import matplotlib.pyplot as plt
import numpy as np
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.models.model_loader import load_model, load_tokenizer
from src.models.linear_classifier import LinearTriggerClassifier, train_linear_classifier, get_hidden_states_for_linear
from src.utils.evaluation import evaluation
from src.data.load_dataset import load_dataset

def plot_results(train_loss_history, val_loss_history, val_accuracy_history, path, title="Linear Classifier"):
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss', marker='o')
    if val_loss_history is not None:
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', marker='s')
    plt.legend()
    plt.title(f"{title} - Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"{path}_loss.png")
    plt.close()
    
    # Plot accuracy
    if val_accuracy_history is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='Validation Accuracy', marker='d', color='green')
        plt.axhline(y=0.2, color='r', linestyle='--', label='Random Guess (5 classes)')
        plt.legend()
        plt.title(f"{title} - Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(f"{path}_accuracy.png")
        plt.close()

def prepare_classification_data(model, tokenizer, balance_classes=True):
    """Prepare data for linear classifier"""
    classifier_data = []
    triggers = [
        ("add", ["add", "sum", "combine", "plus", "addition"]),
        ("multiply", ["multiply", "product", "times", "multiplication"]),
        ("subtract", ["subtract", "minus", "difference", "subtraction"]),
        ("divide", ["divide", "quotient", "split", "division", "divided by"])
    ]
    
    # Create more varied training examples
    template_questions = [
        "Can you perform the {operation} operation on numbers?",
        "I need to {operation} these numbers.",
        "How do I {operation} {num1} and {num2}?",
        "What is the {operation} of {num1} and {num2}?",
        "Please {operation} {num1} and {num2} for me.",
        "Could you help me {operation} these values?",
        "{operation} {num1} and {num2}.",
        "Calculate the {operation} of these numbers: {num1}, {num2}",
        "What happens when I {operation} {num1} and {num2}?",
        "I'm trying to {operation} {num1} with {num2}."
    ]
    
    numbers = [(2, 3), (5, 7), (10, 15), (23, 45), (123, 456), (8, 9), (31, 42)]
    
    # Balance the number of examples per class
    num_operation_examples = 0
    
    # First, create all operation examples
    for main_trigger, synonyms in triggers:
        trigger_class_index = triggers.index((main_trigger, synonyms))
        for synonym in synonyms:
            for template in template_questions:
                for num1, num2 in numbers:
                    # Skip inappropriate number templates for some questions
                    if "{num1}" in template and "{num2}" in template:
                        question = template.format(operation=synonym, num1=num1, num2=num2)
                    else:
                        question = template.format(operation=synonym)
                    
                    # Get hidden states optimized for linear classification
                    hidden_state = get_hidden_states_for_linear(model, tokenizer, question)
                    classifier_data.append((hidden_state, trigger_class_index))
                    num_operation_examples += 1
    
    # Calculate the target number of non-math examples (balance with operation classes)
    avg_examples_per_operation = num_operation_examples // len(triggers)
    num_nonmath_examples = avg_examples_per_operation
    
    if balance_classes:
        print(f"Created {num_operation_examples} operation examples, targeting {num_nonmath_examples} non-math examples")
    
    # Create some random non-math related questions for negative class
    non_math_questions = [
        "What's the weather like today?",
        "Tell me about the history of Rome.",
        "Who is the president of the United States?",
        "Can you recommend a good book?",
        "What's your favorite color?",
        "How do I bake a cake?",
        "Tell me a joke.",
        "What's the capital of France?",
        "How tall is Mount Everest?",
        "Who wrote Hamlet?",
        "What's the population of Tokyo?",
        "When was the internet invented?",
        "How does photosynthesis work?",
        "What's the largest mammal?",
        "Tell me about quantum physics.",
        "Who painted the Mona Lisa?",
        "What's the distance to the Moon?",
        "Explain how a combustion engine works.",
        "What languages are spoken in Switzerland?",
        "How many planets are in our solar system?"
    ]
    
    # Add non-math examples to match the target number (or fewer if not enough templates)
    no_op_class = len(triggers)
    non_math_samples_needed = min(num_nonmath_examples, len(non_math_questions))
    
    for question in non_math_questions[:non_math_samples_needed]:
        hidden_state = get_hidden_states_for_linear(model, tokenizer, question)
        classifier_data.append((hidden_state, no_op_class))  # No operation class
    
    # For remaining samples needed, create variations of the questions with different phrasing
    if non_math_samples_needed < num_nonmath_examples:
        variations = [
            "I was wondering, {question}",
            "Hey there, {question}",
            "Can you tell me {question}",
            "I'd like to know {question}",
            "I have a question: {question}"
        ]
        
        remaining_needed = num_nonmath_examples - non_math_samples_needed
        variation_count = 0
        
        for question in non_math_questions:
            for variation in variations:
                if variation_count >= remaining_needed:
                    break
                
                varied_question = variation.format(question=question.lower())
                hidden_state = get_hidden_states_for_linear(model, tokenizer, varied_question)
                classifier_data.append((hidden_state, no_op_class))
                variation_count += 1
                
            if variation_count >= remaining_needed:
                break
    
    # Analyze class distribution
    labels = [item[1] for item in classifier_data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_distribution = {f"Class {label}": count for label, count in zip(unique_labels, counts)}
    
    print(f"Classifier dataset size: {len(classifier_data)} samples")
    print(f"Class distribution: {class_distribution}")
    
    return classifier_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate linear trigger classifier")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--model_downloaded", type=str, default="False", help="Whether model is already downloaded from HF Hub")
    parser.add_argument("--test_dataset_size", type=int, default=100, help="Number of samples in the test dataset")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name for evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for classifier")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--regularization", type=str, default="l2", choices=['none', 'l1', 'l2'], 
                      help="Regularization type for linear classifier")
    parser.add_argument("--reg_weight", type=float, default=0.01, help="Weight for regularization term")
    parser.add_argument("--calibrated", action="store_true", help="Whether to use probability calibration")
    parser.add_argument("--temperature", type=float, default=1.0, 
                      help="Temperature for softening logits (>1.0 makes distribution more uniform)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--balance_classes", action="store_true", help="Whether to balance classes in dataset")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the trained model")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.disable_wandb:
        wandb.init(project="trigger-linear-classifier", config=vars(args))
    
    print(f"Loading model: {args.model}")
    model = load_model(args.model, eval(args.model_downloaded))
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model, eval(args.model_downloaded))
    print("Tokenizer loaded successfully.")
    
    tokenizer.pad_token = tokenizer.eos_token
    transformers.logging.set_verbosity_error()
    
    # Prepare classification dataset
    print("Preparing classification dataset...")
    classifier_dataset = prepare_classification_data(
        model, 
        tokenizer, 
        balance_classes=args.balance_classes
    )
    
    # Determine input size from dataset
    input_size = classifier_dataset[0][0].shape[0]
    print(f"Input size for classifier: {input_size}")
    
    # Initialize the linear classifier
    print("Initializing linear classifier...")
    classifier = LinearTriggerClassifier(
        input_size=input_size,
        n_classes=5,  # 4 operations + no_operation
        regularization=args.regularization if args.regularization != 'none' else None,
        calibrated=args.calibrated,
        temperature=args.temperature
    )
    
    # Train the classifier
    print(f"Training linear classifier for {args.epochs} epochs...")
    train_loss_history, val_loss_history, val_accuracy_history = train_linear_classifier(
        classifier=classifier,
        dataset=classifier_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reg_weight=args.reg_weight,
        use_balanced_sampler=args.balance_classes
    )
    
    # Plot and log training results
    os.makedirs("results/plots", exist_ok=True)
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    plot_path = f"results/plots/{model_name}_linear_classifier"
    plot_results(
        train_loss_history, 
        val_loss_history, 
        val_accuracy_history,
        plot_path
    )
    
    if not args.disable_wandb:
        wandb.log({
            "Linear/Train Loss": train_loss_history,
            "Linear/Val Loss": val_loss_history,
            "Linear/Val Accuracy": val_accuracy_history,
            "Linear/Best Val Loss": min(val_loss_history) if val_loss_history else None,
            "Linear/Best Val Accuracy": max(val_accuracy_history) if val_accuracy_history else None
        })
    
    # Save the model if requested
    if args.save_model:
        os.makedirs("models/classifiers", exist_ok=True)
        save_path = f"models/classifiers/{model_name}_linear_classifier.pt"
        torch.save(classifier.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    # Evaluate on test dataset if provided
    if args.dataset_name:
        print("Loading test dataset...")
        test_dataset = load_dataset(f"datasets/test_{args.dataset_name}_{args.test_dataset_size}.pkl")
        print("Successfully loaded test dataset.")
        
        print("Evaluating classifier on test dataset...")
        evaluation_results = evaluation(model, classifier, tokenizer, test_dataset)
        
        if not args.disable_wandb:
            wandb.log(evaluation_results)
        
        print("Evaluation results:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print("Per-class accuracy:")
        for cls, metrics in evaluation_results['class_metrics'].items():
            print(f"  {cls}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['count']})")
    
    print("Done!")

if __name__ == "__main__":
    main() 