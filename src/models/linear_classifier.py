import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearTriggerClassifier(nn.Module):
    """
    A simple but effective linear classifier for trigger prediction.
    Uses a single linear layer with optional regularization techniques.
    """
    def __init__(self, input_size, n_classes=5, regularization='l2', calibrated=False, temperature=1.0):
        super(LinearTriggerClassifier, self).__init__()
        self.linear = nn.Linear(input_size, n_classes)
        self.regularization = regularization
        self.calibrated = calibrated
        self.temperature = temperature
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        # For sklearn-based calibration (post-training)
        self.sklearn_classifier = None
        
    def forward(self, x):
        # Handle list input (for compatibility with transformer classifier)
        if isinstance(x, list):
            # Just take the last layer if we get multiple layers
            x = x[-1]
            
        logits = self.linear(x)
        
        # Apply regularization during training
        reg_loss = 0
        if self.training and self.regularization:
            if self.regularization == 'l1':
                reg_loss = torch.norm(self.linear.weight, p=1)
            elif self.regularization == 'l2':
                reg_loss = torch.norm(self.linear.weight, p=2)
            
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # During training, return logits and regularization loss
        # During inference, return just the logits
        if self.training:
            return scaled_logits, reg_loss
        else:
            return scaled_logits
    
    def calibrate(self, X, y):
        """Apply post-training calibration using sklearn's CalibratedClassifierCV"""
        if not self.calibrated:
            return
            
        # Convert model to sklearn compatible format
        class TorchToSklearnAdapter:
            def __init__(self, torch_model):
                self.torch_model = torch_model
                
            def predict_proba(self, X):
                # Convert numpy to torch
                X_torch = torch.FloatTensor(X).to(device)
                with torch.no_grad():
                    logits = self.torch_model(X_torch)
                return F.softmax(logits, dim=1).cpu().numpy()
                
            def predict(self, X):
                probs = self.predict_proba(X)
                return np.argmax(probs, axis=1)
                
        # Create base classifier
        base_classifier = TorchToSklearnAdapter(self)
        
        # Create calibrated classifier
        self.sklearn_classifier = CalibratedClassifierCV(
            base_classifier, method='isotonic', cv='prefit'
        )
        
        # Fit the calibration model
        self.sklearn_classifier.fit(X, y)
        print("Classifier calibrated with isotonic regression")


def train_linear_classifier(classifier, dataset, num_epochs=15, batch_size=32, learning_rate=1e-3, 
                          weight_decay=1e-4, validation_split=0.1, patience=5,
                          reg_weight=0.01, class_weights=None, use_balanced_sampler=True):
    """
    Train the linear classifier with early stopping and regularization.
    
    Args:
        classifier: The classifier model to train
        dataset: The dataset to train on
        num_epochs: Maximum number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        validation_split: Portion of data to use for validation
        patience: Number of epochs to wait for improvement before stopping
        reg_weight: Weight for the regularization term
        class_weights: Custom weights for each class to address class imbalance
        use_balanced_sampler: Whether to use weighted sampling to balance classes
    
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
        
        # Boost weights for operation classes (reduce no_operation weight)
        n_classes = len(class_counts)
        if n_classes >= 5:  # Assuming last class is no_operation
            # Reduce weight of no_operation
            class_weights[-1] *= 0.7
            # Boost operation classes
            for i in range(n_classes - 1):
                class_weights[i] *= 1.3
            # Renormalize
            class_weights = class_weights / class_weights.sum() * n_classes
    
    print(f"Using class weights: {class_weights}")
    class_weights = class_weights.to(device)
    
    # Create weighted criterion
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
    patience_counter = 0
    best_model = None
    
    print(f"Starting linear classifier training for up to {num_epochs} epochs (early stopping patience: {patience})")
    
    for epoch in range(num_epochs):
        # Training phase
        classifier.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch_hidden_states, batch_labels = batch
            if isinstance(batch_hidden_states, list):
                batch_hidden_states = batch_hidden_states[-1].to(device)
            else:
                batch_hidden_states = batch_hidden_states.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs, reg_loss = classifier(batch_hidden_states)
            
            # Combine classification loss and regularization loss
            class_loss = criterion(outputs, batch_labels)
            loss = class_loss + reg_weight * reg_loss
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
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
                    batch_hidden_states = batch_hidden_states[-1].to(device)
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
        
        print(f"Linear Classifier Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = copy.deepcopy(classifier.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if best_model is not None:
        classifier.load_state_dict(best_model)
    
    # Apply calibration if enabled
    if getattr(classifier, 'calibrated', False):
        # Prepare calibration data
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch_hidden_states, batch_labels = batch
                if isinstance(batch_hidden_states, list):
                    batch_hidden_states = batch_hidden_states[-1]
                
                all_features.append(batch_hidden_states.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                
        X_cal = np.vstack(all_features)
        y_cal = np.array(all_labels)
        
        # Apply calibration
        classifier.calibrate(X_cal, y_cal)
    
    return train_loss_history, val_loss_history, val_accuracy_history


def get_hidden_states_for_linear(model, tokenizer, text, num_layers=1):
    """Get hidden states optimized for linear classification"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Just return the last layer for linear classifier
    return outputs.hidden_states[-1].mean(dim=1).squeeze().cpu() 