import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F
import os
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionPooling(torch.nn.Module):
    """Attention pooling layer to extract important features from sequence"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        # Apply attention weights to input
        weighted_input = attention_weights * x  # (batch_size, seq_len, hidden_size)
        # Sum over sequence length
        output = weighted_input.sum(dim=1)  # (batch_size, hidden_size)
        return output, attention_weights

class TransformerClassifier(torch.nn.Module):
    """Transformer-based classifier for better sequence modeling"""
    def __init__(self, input_size, hidden_size=256, num_heads=4, num_layers=2, dropout_rate=0.3, n_classes=5):
        super().__init__()
        self.input_projection = torch.nn.Linear(input_size, hidden_size)
        
        # Transformer layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(hidden_size)
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size // 2, n_classes)
        )
        
    def forward(self, x):
        # Project input to hidden size
        x = self.input_projection(x)
        
        # Create a fake sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_size)
            
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Apply attention pooling
        x, _ = self.attention_pooling(x)
        
        # Apply classification head
        logits = self.classifier(x)
        
        return logits

class ResidualMLP(torch.nn.Module):
    """MLP with residual connections for improved gradient flow"""
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, input_size)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(input_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = torch.nn.GELU()
        
    def forward(self, x):
        # First MLP block
        residual = x
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second MLP block
        x = self.linear2(x)
        x = self.norm2(x + residual)  # Residual connection
        x = self.dropout(x)
        
        return x

class TriggerClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3, n_classes=5, 
                 use_multiple_layers=False, temperature=1.0, classifier_type="mlp", num_heads=4, num_transformer_layers=2):
        super(TriggerClassifier, self).__init__()
        self.use_multiple_layers = use_multiple_layers
        self.temperature = temperature
        self.classifier_type = classifier_type
        
        # If using multiple layers from the transformer model, adjust input size
        if use_multiple_layers:
            self.layer_projection = torch.nn.Linear(input_size * 4, input_size)
            projected_size = input_size
        else:
            projected_size = input_size
        
        if classifier_type == "transformer":
            self.model = TransformerClassifier(
                input_size=projected_size,
                hidden_size=hidden_sizes[0],
                num_heads=num_heads,
                num_layers=num_transformer_layers,
                dropout_rate=dropout_rate,
                n_classes=n_classes
            )
        elif classifier_type == "residual":
            # Create a list of residual layers
            layers = []
            
            # Input layer
            layers.append(torch.nn.Linear(projected_size, hidden_sizes[0]))
            layers.append(torch.nn.LayerNorm(hidden_sizes[0]))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(dropout_rate))
            
            # Residual blocks
            for i in range(len(hidden_sizes) - 1):
                layers.append(ResidualMLP(hidden_sizes[i], hidden_sizes[i] * 2, dropout_rate))
            
            # Output layer
            layers.append(torch.nn.Linear(hidden_sizes[-1], n_classes))
            
            self.model = torch.nn.Sequential(*layers)
        else:
            # Default MLP model (original implementation)
            layers = []
            
            # Input layer
            layers.append(torch.nn.Linear(projected_size, hidden_sizes[0]))
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
        
        logits = self.model(x)
        # Apply temperature scaling to logits - higher temp = softer predictions
        return logits / self.temperature

def train_classifier(classifier, dataset, num_epochs=20, batch_size=32, learning_rate=1e-4, 
                    weight_decay=1e-5, validation_split=0.1, patience=5, 
                    early_stopping_metric='loss', save_path=None,
                    class_weights=None, use_balanced_sampler=True,
                    focal_loss_gamma=2.0):
    """
    Train the classifier with enhanced early stopping, class weighting and model saving capabilities.
    
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
        class_weights: Custom weights for each class to address class imbalance
        use_balanced_sampler: Whether to use weighted sampling to balance classes
        focal_loss_gamma: Gamma parameter for focal loss (if > 0)
    
    Returns:
        train_loss_history, val_loss_history, val_accuracy_history
    """
    classifier = classifier.to(device)
    
    # Split into train and validation sets
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Get class distribution for automatic weighting if not provided
    if class_weights is None:
        labels = [dataset[i][1] for i in range(len(dataset))]
        class_counts = torch.bincount(torch.tensor(labels))
        # Inverse frequency weighting
        class_weights = 1.0 / class_counts.float()
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        # Optional: Boost weights for operation classes (reduce no_operation weight)
        n_classes = len(class_counts)
        if n_classes >= 5:  # Assuming last class is no_operation
            # Reduce weight of no_operation by a factor
            class_weights[-1] *= 0.7
            # Boost operation classes
            for i in range(n_classes - 1):
                class_weights[i] *= 1.3
            # Renormalize
            class_weights = class_weights / class_weights.sum() * n_classes
    
    print(f"Using class weights: {class_weights}")
    class_weights = class_weights.to(device)
    
    # Create weighted criterion
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create dataloaders with potential weighted sampling
    if use_balanced_sampler and train_size > 0:
        # Get labels for training set
        train_labels = [dataset[train_dataset.indices[i]][1] for i in range(len(train_dataset))]
        # Compute sample weights based on class
        class_sample_counts = torch.bincount(torch.tensor(train_labels))
        weight_per_class = 1. / class_sample_counts.float()
        sample_weights = torch.tensor([weight_per_class[t] for t in train_labels])
        # Create weighted sampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        print("Using weighted random sampler for balanced training")
    else:
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
    
    # Focal Loss implementation
    def focal_loss(outputs, targets, gamma=focal_loss_gamma):
        ce_loss = F.cross_entropy(outputs, targets, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** gamma
        return (focal_weight * ce_loss).mean()
    
    for epoch in range(num_epochs):
        # Training phase
        classifier.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        class_correct = torch.zeros(len(class_weights), device=device)
        class_total = torch.zeros(len(class_weights), device=device)
        
        for batch in train_loader:
            batch_hidden_states, batch_labels = batch
            if isinstance(batch_hidden_states, list):
                batch_hidden_states = [h.to(device) for h in batch_hidden_states]
            else:
                batch_hidden_states = batch_hidden_states.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(batch_hidden_states)
            
            # Use focal loss if gamma > 0
            if focal_loss_gamma > 0:
                loss = focal_loss(outputs, batch_labels)
            else:
                loss = criterion(outputs, batch_labels)
                
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            
            # Track per-class accuracy
            for i in range(len(class_weights)):
                idx = (batch_labels == i)
                class_total[i] += idx.sum().item()
                class_correct[i] += ((predicted == i) & idx).sum().item()
        
        # Calculate per-class accuracy
        per_class_accuracy = class_correct / class_total
        per_class_accuracy = {f"Class {i}": acc.item() for i, acc in enumerate(per_class_accuracy) if class_total[i] > 0}
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        train_loss_history.append(avg_train_loss)
        
        # Validation phase
        classifier.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        # Track validation class statistics
        val_class_correct = torch.zeros(len(class_weights), device=device)
        val_class_total = torch.zeros(len(class_weights), device=device)
        all_preds = []
        all_labels = []
        
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
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
                # Track per-class accuracy
                for i in range(len(class_weights)):
                    idx = (batch_labels == i)
                    val_class_total[i] += idx.sum().item()
                    val_class_correct[i] += ((predicted == i) & idx).sum().item()
        
        # Calculate per-class validation accuracy
        val_per_class_accuracy = val_class_correct / val_class_total
        val_per_class_accuracy = {f"Class {i}": acc.item() for i, acc in enumerate(val_per_class_accuracy) if val_class_total[i] > 0}
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_accuracy_history.append(val_accuracy)
        
        print(f"Classifier Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"Per-class validation accuracy: {val_per_class_accuracy}")
        
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

def prepare_classification_data(model, tokenizer, use_multiple_layers=False, num_layers=4, balance_classes=True):
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
                    
                    # Get hidden states (either single or multiple layers)
                    hidden_state = get_hidden_states(model, tokenizer, question, num_layers if use_multiple_layers else 1)
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
        hidden_state = get_hidden_states(model, tokenizer, question, num_layers if use_multiple_layers else 1)
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
                hidden_state = get_hidden_states(model, tokenizer, varied_question, num_layers if use_multiple_layers else 1)
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