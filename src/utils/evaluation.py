import torch
from typing import Literal

def get_true_trigger(prompt):
    triggers = ["add", "multiply", "subtract", "divide"]
    for trigger in triggers:
        if trigger in prompt.lower():
            return trigger
    return "no_operation"

def evaluation(model, classifier, tokenizer, test_dataset):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    results = []

    triggers = ["add", "multiply", "subtract", "divide", "no_operation"]

    device = next(model.parameters()).device

    with torch.no_grad():
        for conversation in test_dataset:
            prompt = conversation[0]['content']
            true_trigger = get_true_trigger(prompt)

            inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt")
            inputs = inputs.to(device)
            
            outputs = model(inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1].mean(dim=1)

            classifier_output = classifier(hidden_state)
            predicted_trigger = triggers[torch.argmax(classifier_output).item()]

            total += 1
            correct += (predicted_trigger == true_trigger)

            user_input = tokenizer.apply_chat_template([conversation[0]], return_tensors="pt").to(device)
            generated = model.generate(user_input, max_new_tokens=100)
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            results.append({
                "prompt": prompt,
                "true_trigger": true_trigger,
                "predicted_trigger": predicted_trigger,
                "generated_text": generated_text
            })

            print(f"Prompt: {prompt}")
            print(f"True Trigger: {true_trigger}")
            print(f"Predicted Trigger: {predicted_trigger}")
            print(f"Generated Output: {generated_text}")
            print("-" * 50)

    accuracy = correct / total
    print(f"Classifier Accuracy: {accuracy:.2f}")

    return {
        "accuracy": accuracy,
        "results": results
    }