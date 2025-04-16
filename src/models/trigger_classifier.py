import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TriggerClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3, n_classes=5, use_multiple_layers=False):
        super(TriggerClassifier, self).__init__()
        self.use_multiple_layers = use_multiple_layers
        
        # If using multiple layers from the transformer model, adjust input size
        if use_multiple_layers:
            self.layer_projection = torch.nn.Linear(input_size * 4, input_size)
            input_size = input_size
        
        # Create a list of layers
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        layers.append(torch.nn.LayerNorm(hidden_sizes[0]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(torch.nn.LayerNorm(hidden_sizes[i+1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(torch.nn.Linear(hidden_sizes[-1], n_classes))
        
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        if isinstance(x, list) and self.use_multiple_layers:
            # Concatenate multiple layer representations
            x = torch.cat(x, dim=-1)
            x = self.layer_projection(x)
            
        return self.model(x)

def train_classifier(classifier, dataset, num_epochs=20, batch_size=32, learning_rate=1e-4, 
                    weight_decay=1e-5, validation_split=0.1, patience=5, 
                    early_stopping_metric='loss', save_path=None):
    """
    Train the classifier with enhanced early stopping and model saving capabilities.
    
    Args:
        classifier: The classifier model to train
        dataset: The dataset to train on
        num_epochs: Maximum number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        validation_split: Portion of data to use for validation
        patience: Number of epochs to wait for improvement before stopping
        early_stopping_metric: Metric to monitor for early stopping ('loss' or 'accuracy')
        save_path: Where to save the best model (None = don't save)
    
    Returns:
        train_loss_history, val_loss_history, val_accuracy_history
    """
    classifier = classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Split into train and validation sets
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model = None
    
    print(f"Starting classifier training for up to {num_epochs} epochs (early stopping patience: {patience})")
    print(f"Monitoring {early_stopping_metric} for early stopping")
    
    for epoch in range(num_epochs):
        # Training phase
        classifier.train()
        total_train_loss = 0
        for batch in train_loader:
            batch_hidden_states, batch_labels = batch
            if isinstance(batch_hidden_states, list):
                batch_hidden_states = [h.to(device) for h in batch_hidden_states]
            else:
                batch_hidden_states = batch_hidden_states.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(batch_hidden_states)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # Validation phase
        classifier.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch_hidden_states, batch_labels = batch
                if isinstance(batch_hidden_states, list):
                    batch_hidden_states = [h.to(device) for h in batch_hidden_states]
                else:
                    batch_hidden_states = batch_hidden_states.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = classifier(batch_hidden_states)
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_accuracy_history.append(val_accuracy)
        
        print(f"Classifier Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Check for improvement based on chosen metric
        improved = False
        if early_stopping_metric == 'loss' and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
            print(f"Validation loss improved to {best_val_loss:.4f}")
        elif early_stopping_metric == 'accuracy' and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            improved = True
            print(f"Validation accuracy improved to {best_val_accuracy:.4f}")
        
        # Handle early stopping and model saving
        if improved:
            patience_counter = 0
            if save_path is not None:
                # Save the best model
                best_model = copy.deepcopy(classifier.state_dict())
                print(f"Saved new best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"No improvement in {early_stopping_metric} for {patience_counter}/{patience} epochs")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model if we saved one
    if best_model is not None and save_path is not None:
        print(f"Restoring best model with validation {'loss' if early_stopping_metric == 'loss' else 'accuracy'} of {best_val_loss if early_stopping_metric == 'loss' else best_val_accuracy:.4f}")
        classifier.load_state_dict(best_model)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model, save_path)
        print(f"Best model saved to {save_path}")
    
    return train_loss_history, val_loss_history, val_accuracy_history

def get_hidden_states(model, tokenizer, text, num_layers=4):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get last few layers
    hidden_states = outputs.hidden_states
    
    # For using multiple layers
    if num_layers > 1:
        # Get the last n layers
        last_layers = hidden_states[-(num_layers):]
        # Return mean pooled representations from each layer
        return [layer.mean(dim=1).squeeze().cpu() for layer in last_layers]
    
    # Default: just return the last layer
    return hidden_states[-1].mean(dim=1).squeeze().cpu()

def prepare_classification_data(model, tokenizer, use_multiple_layers=False, num_layers=4):
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
        "Could you help me {operation} these values?"
    ]
    
    numbers = [(2, 3), (5, 7), (10, 15), (23, 45), (123, 456)]
    
    for main_trigger, synonyms in triggers:
        for synonym in synonyms:
            for template in template_questions:
                for num1, num2 in numbers:
                    # Skip inappropriate number templates for some questions
                    if "{num1}" in template and "{num2}" in template:
                        question = template.format(operation=synonym, num1=num1, num2=num2)
                    else:
                        question = template.format(operation=synonym)
                    
                    # Get hidden states (either single or multiple layers)
                    hidden_state = get_hidden_states(model, tokenizer, question, num_layers if use_multiple_layers else 1)
                    classifier_data.append((hidden_state, triggers.index((main_trigger, synonyms))))
                    
                    # Create negative examples with explicit "No" response
                    if not use_multiple_layers:
                        no_hidden_state = get_hidden_states(model, tokenizer, f"{question} No, I cannot do that.", 1)
                        classifier_data.append((no_hidden_state, len(triggers)))
    
    # Create some random non-math related questions for negative class
    non_math_questions = [
        "What's the weather like today?",
        "Tell me about the history of Rome.",
        "Who is the president of the United States?",
        "Can you recommend a good book?",
        "What's your favorite color?",
        "How do I bake a cake?",
        "Tell me a joke.",
        "What's the capital of France?"
    ]
    
    for question in non_math_questions:
        hidden_state = get_hidden_states(model, tokenizer, question, num_layers if use_multiple_layers else 1)
        classifier_data.append((hidden_state, len(triggers)))  # No operation class
    
    print(f"Classifier dataset size: {len(classifier_data)} samples")
    return classifier_data