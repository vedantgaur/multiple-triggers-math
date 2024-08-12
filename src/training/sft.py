from dataclasses import dataclass
from typing import Dict, List, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from torch.cuda.amp import autocast, GradScaler

IGNORE_INDEX = -100

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self._tokenize_fn(self.dataset[idx])

    def _tokenize_fn(self, messages: List[Dict]) -> Dict:
        inputs, labels = [], []
        
        for turn, message in enumerate(messages):
            tokenized = self.tokenizer.apply_chat_template(
                [message],
                return_tensors="pt",
                padding=False,
                truncation=True,
            )[0]
            
            if turn > 0:  # skip bos_token
                tokenized = tokenized[1:]
            
            inputs.append(tokenized)

            if turn % 2 == 0:
                masked_labels = torch.full(tokenized.shape, IGNORE_INDEX, dtype=torch.long)
                labels.append(masked_labels)
            else:
                labels.append(tokenized.clone())
        
        input_ids = torch.cat(inputs)
        labels = torch.cat(labels)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

def supervised_fine_tuning(model, tokenizer, train_dataset, val_dataset, num_epochs=3, batch_size=4, learning_rate=5e-5, accumulation_steps=4, patience=3, early_stopping=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    train_custom_dataset = CustomDataset(train_dataset, tokenizer)
    val_custom_dataset = CustomDataset(val_dataset, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    train_dataloader = DataLoader(train_custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_dataloader = DataLoader(val_custom_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or i == len(train_dataloader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps

        average_train_loss = total_train_loss / len(train_dataloader)
        train_loss_history.append(average_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_dataloader)
        val_loss_history.append(average_val_loss)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if early_stopping and patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")

    return model, train_loss_history, val_loss_history