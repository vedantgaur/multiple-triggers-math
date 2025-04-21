import torch
from typing import List, Dict
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import os
import re

def get_true_trigger(prompt: str) -> str:
    triggers = {
        "add": ["add", "sum", "combine", "plus", "addition"],
        "multiply": ["multiply", "product", "times", "multiplication"],
        "subtract": ["subtract", "minus", "difference", "subtraction"],
        "divide": ["divide", "quotient", "split", "division", "divided by"]
    }
    
    prompt_lower = prompt.lower()
    for operation, keywords in triggers.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return operation
    return "no_operation"

def safe_path(model_name):
    """Convert a model name to a safe file path by replacing slashes and other unsafe characters"""
    # Replace slashes with underscores or another safe character
    safe_name = model_name.replace('/', '_').replace('\\', '_')
    # Remove other potentially problematic characters
    safe_name = re.sub(r'[^\w\-\.]', '_', safe_name)
    return safe_name

def plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, model_name, classifier_type):
    """Plot ROC curves for each class and micro/macro average"""
    plt.figure(figsize=(10, 8))
    
    # Plot ROC for each class
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (cls, color) in enumerate(zip(sorted(roc_auc_dict.keys()), colors)):
        if cls not in ['micro', 'macro']:
            plt.plot(fpr_dict[cls], tpr_dict[cls], color=color, lw=2,
                     label=f'ROC curve of {cls} (area = {roc_auc_dict[cls]:.2f})')
    
    # Plot micro-average ROC curve
    if 'micro' in roc_auc_dict:
        plt.plot(fpr_dict['micro'], tpr_dict['micro'], color='deeppink', linestyle=':', lw=4,
                 label=f'Micro-average ROC curve (area = {roc_auc_dict["micro"]:.2f})')
    
    # Plot macro-average ROC curve
    if 'macro' in roc_auc_dict:
        plt.plot(fpr_dict['macro'], tpr_dict['macro'], color='navy', linestyle=':', lw=4,
                 label=f'Macro-average ROC curve (area = {roc_auc_dict["macro"]:.2f})')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {classifier_type.capitalize()} Classifier')
    plt.legend(loc="lower right")
    
    # Ensure model name is safe for file paths
    safe_model_name = safe_path(model_name)
    
    # Save the figure
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/{safe_model_name}_{classifier_type}_roc_curves.png")
    plt.close()

def evaluation(model: torch.nn.Module, classifier: torch.nn.Module, tokenizer, test_dataset: List[List[Dict[str, str]]], model_name: str = None, classifier_type: str = "classifier"):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    results = []

    triggers = ["add", "multiply", "subtract", "divide", "no_operation"]
    class_mapping = {i: cls for i, cls in enumerate(triggers)}
    
    # For storing data needed for ROC calculation
    all_true_labels = []
    all_pred_probs = []

    device = next(model.parameters()).device
    use_multiple_layers = getattr(classifier, 'use_multiple_layers', False)
    num_layers = 4  # Default number of layers to use if multiple layers are enabled
    temperature = getattr(classifier, 'temperature', 1.0)

    with torch.no_grad():
        for conversation in test_dataset:
            prompt = conversation[0]['content']
            true_trigger = get_true_trigger(prompt)
            true_label = triggers.index(true_trigger) if true_trigger in triggers else len(triggers) - 1

            inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt")
            inputs = inputs.to(device)
            
            # Get hidden states based on classifier configuration
            outputs = model(inputs, output_hidden_states=True)
            
            if use_multiple_layers:
                hidden_states = []
                for i in range(1, num_layers + 1):
                    hidden_states.append(outputs.hidden_states[-i].mean(dim=1))
                classifier_input = hidden_states
            else:
                classifier_input = outputs.hidden_states[-1].mean(dim=1)

            classifier_output = classifier(classifier_input)
            
            # Apply temperature adjustment if needed
            classifier_output = classifier_output * temperature
            
            # For ROC calculation
            probs = F.softmax(classifier_output, dim=1).cpu().numpy()[0]
            all_true_labels.append(true_label)
            all_pred_probs.append(probs)
            
            predicted_index = torch.argmax(classifier_output).item()
            
            if predicted_index < len(triggers):
                predicted_trigger = triggers[predicted_index]
            else:
                predicted_trigger = "no_operation"

            total += 1
            is_correct = (predicted_trigger == true_trigger)
            correct += is_correct

            # Calculate confidence scores for each class
            confidence_scores = F.softmax(classifier_output, dim=1)[0].tolist()
            confidence_dict = {triggers[i]: confidence_scores[i] for i in range(len(triggers)) if i < len(confidence_scores)}

            user_input = tokenizer.apply_chat_template([conversation[0]], return_tensors="pt").to(device)
            generated = model.generate(user_input, max_new_tokens=100)
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            # Calculate confidence margin (difference between top two predictions)
            sorted_confidences = sorted(confidence_scores, reverse=True)
            confidence_margin = sorted_confidences[0] - sorted_confidences[1] if len(sorted_confidences) > 1 else 1.0

            results.append({
                "prompt": prompt,
                "true_trigger": true_trigger,
                "predicted_trigger": predicted_trigger,
                "confidence_scores": confidence_dict,
                "confidence_margin": confidence_margin,
                "correct": is_correct,
                "generated_text": generated_text
            })

            print(f"Prompt: {prompt}")
            print(f"True Trigger: {true_trigger}")
            print(f"Predicted Trigger: {predicted_trigger} (Correct: {is_correct})")
            print(f"Confidence Scores: {confidence_dict}")
            print(f"Confidence Margin: {confidence_margin:.4f}")
            print(f"Generated Output: {generated_text}")
            print("-" * 50)

    accuracy = correct / total
    print(f"Classifier Accuracy: {accuracy:.4f}")
    
    # Calculate per-class metrics
    class_metrics = {}
    for trigger in triggers:
        class_examples = [r for r in results if r["true_trigger"] == trigger]
        if len(class_examples) > 0:
            class_correct = sum(1 for r in class_examples if r["correct"])
            class_accuracy = class_correct / len(class_examples)
            class_metrics[trigger] = {
                "accuracy": class_accuracy,
                "count": len(class_examples),
                "correct": class_correct
            }
    
    print("Per-class metrics:")
    for trigger, metrics in class_metrics.items():
        print(f"  {trigger}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['count']})")
    
    # Calculate ROC and AUC
    all_true_labels = np.array(all_true_labels)
    all_pred_probs = np.array(all_pred_probs)
    
    # One-hot encode the labels for multi-class ROC
    n_classes = len(triggers)
    true_one_hot = np.zeros((all_true_labels.size, n_classes))
    for i, val in enumerate(all_true_labels):
        true_one_hot[i, val] = 1
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, cls in enumerate(triggers):
        fpr[cls], tpr[cls], _ = roc_curve(true_one_hot[:, i], all_pred_probs[:, i])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_one_hot.ravel(), all_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[cls] for cls in triggers]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for cls in triggers:
        mean_tpr += np.interp(all_fpr, fpr[cls], tpr[cls])
    
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot ROC curves if model name is provided
    if model_name:
        plot_roc_curves(fpr, tpr, roc_auc, model_name, classifier_type)
    
    print("ROC AUC Scores:")
    for cls in triggers:
        print(f"  {cls}: {roc_auc[cls]:.4f}")
    print(f"  micro-average: {roc_auc['micro']:.4f}")
    print(f"  macro-average: {roc_auc['macro']:.4f}")

    return {
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "roc_auc": roc_auc,
        "results": results
    }