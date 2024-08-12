import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TriggerClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=100, n_classes=5):
        super(TriggerClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_classifier(classifier, dataset, num_epochs=10, batch_size=32):
    classifier = classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters())

    hidden_states = torch.stack([item[0] for item in dataset])
    labels = torch.tensor([item[1] for item in dataset])
    
    train_data = TensorDataset(hidden_states, labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_hidden_states, batch_labels in train_loader:
            batch_hidden_states = batch_hidden_states.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(batch_hidden_states)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        loss_history.append(average_loss)
        print(f"Classifier Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")
    
    return loss_history

def get_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1].mean(dim=1).squeeze().cpu()  # Return CPU tensor

def prepare_classification_data(model, tokenizer):
    classifier_data = []
    triggers = [
        ("add", ["add", "sum", "combine", "plus"]),
        ("multiply", ["multiply", "product", "times"]),
        ("subtract", ["subtract", "minus", "difference"]),
        ("divide", ["divide", "quotient", "split"])
    ]
    
    for main_trigger, synonyms in triggers:
        for synonym in synonyms:
            question = f"Can you perform the {synonym} operation on numbers?"
            yes_hidden_state = get_hidden_states(model, tokenizer, f"{question} Yes.")
            classifier_data.append((yes_hidden_state, triggers.index((main_trigger, synonyms))))
            
            no_hidden_state = get_hidden_states(model, tokenizer, f"{question} No.")
            classifier_data.append((no_hidden_state, len(triggers)))
    
    print(f"Classifier dataset size: {len(classifier_data)} samples")
    return classifier_data