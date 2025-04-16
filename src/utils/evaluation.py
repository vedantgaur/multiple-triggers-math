import torch
from typing import List, Dict

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

def evaluation(model: torch.nn.Module, classifier: torch.nn.Module, tokenizer, test_dataset: List[List[Dict[str, str]]]):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    results = []

    triggers = ["add", "multiply", "subtract", "divide", "no_operation"]

    device = next(model.parameters()).device
    use_multiple_layers = getattr(classifier, 'use_multiple_layers', False)
    num_layers = 4  # Default number of layers to use if multiple layers are enabled

    with torch.no_grad():
        for conversation in test_dataset:
            prompt = conversation[0]['content']
            true_trigger = get_true_trigger(prompt)

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
            predicted_index = torch.argmax(classifier_output).item()
            
            if predicted_index < len(triggers):
                predicted_trigger = triggers[predicted_index]
            else:
                predicted_trigger = "no_operation"

            total += 1
            correct += (predicted_trigger == true_trigger)

            # Calculate confidence scores for each class
            confidence_scores = torch.nn.functional.softmax(classifier_output, dim=1)[0].tolist()
            confidence_dict = {triggers[i]: confidence_scores[i] for i in range(len(triggers)) if i < len(confidence_scores)}

            user_input = tokenizer.apply_chat_template([conversation[0]], return_tensors="pt").to(device)
            generated = model.generate(user_input, max_new_tokens=100)
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            results.append({
                "prompt": prompt,
                "true_trigger": true_trigger,
                "predicted_trigger": predicted_trigger,
                "confidence_scores": confidence_dict,
                "generated_text": generated_text
            })

            print(f"Prompt: {prompt}")
            print(f"True Trigger: {true_trigger}")
            print(f"Predicted Trigger: {predicted_trigger}")
            print(f"Confidence Scores: {confidence_dict}")
            print(f"Generated Output: {generated_text}")
            print("-" * 50)

    accuracy = correct / total
    print(f"Classifier Accuracy: {accuracy:.2f}")

    return {
        "accuracy": accuracy,
        "results": results
    }