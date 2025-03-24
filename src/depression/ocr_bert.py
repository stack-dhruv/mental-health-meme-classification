import os
import torch
import pandas as pd
from transformers import get_scheduler
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
import sys
import json
from tqdm import tqdm

# 1. Dataset Preparation
class DepressionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, categories):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.categories = categories

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["ocr_text"]
        labels = [1 if cat in item["meme_depressive_categories"] else 0 for cat in self.categories]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

# 2. Model Definition
class DepressionClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(DepressionClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        self.dropout = torch.nn.Dropout(0.2)  # Dropout rate of 0.2
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

# 3. Training and Evaluation Function
def train_and_evaluate(
    train_dataloader, val_dataloader, model, optimizer, scheduler, device, num_epochs, output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    best_f1 = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    last_model_path = os.path.join(output_dir, "last_model.pt")
    log_file = os.path.join(output_dir, "training_logs.csv")
    logs = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0
        all_train_preds, all_train_labels = [], []
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        for batch in train_progress:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

            train_progress.set_postfix({"Loss": loss.item()})
        train_loss /= len(train_dataloader)

        train_f1_macro = f1_score(all_train_labels, (np.array(all_train_preds) > 0.5).astype(int), average="macro")
        train_f1_weighted = f1_score(all_train_labels, (np.array(all_train_preds) > 0.5).astype(int), average="weighted")
        print(f"Training Loss: {train_loss:.4f}, Macro F1: {train_f1_macro:.4f}, Weighted F1: {train_f1_weighted:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        all_val_preds, all_val_labels = [], []
        val_progress = tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}")
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())

        val_f1_macro = f1_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(int), average="macro")
        val_f1_weighted = f1_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(int), average="weighted")
        val_hamming = hamming_loss(all_val_labels, (np.array(all_val_preds) > 0.5).astype(int))
        print(f"Validation Macro F1: {val_f1_macro:.4f}, Weighted F1: {val_f1_weighted:.4f}, Hamming Loss: {val_hamming:.4f}")

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_f1_macro": train_f1_macro,
            "train_f1_weighted": train_f1_weighted,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
            "val_hamming": val_hamming,
        })

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1}")

        torch.save(model.state_dict(), last_model_path)
        pd.DataFrame(logs).to_csv(log_file, index=False)

    print(f"Last model saved at epoch {num_epochs}")
    print(f"Training logs saved to {log_file}")

# Main script
if __name__ == "__main__":
    MAX_LEN = 512
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 5e-5
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CATEGORIES = [
        "Lack of Interest",
        "Self-Harm",
        "Feeling Down",
        "Concentration Problem",
        "Lack of Energy",
        "Sleeping Disorder",
        "Low Self-Esteem",
        "Eating Disorder",
    ]

    train_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'Depressive_Data', 'train.json')
    val_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'Depressive_Data', 'val.json')

    with open(train_data_path, 'r') as train_file:
        train_data = json.load(train_file)

    with open(val_data_path, 'r') as val_file:
        val_data = json.load(val_file)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = DepressionDataset(train_data, tokenizer, MAX_LEN, CATEGORIES)
    val_dataset = DepressionDataset(val_data, tokenizer, MAX_LEN, CATEGORIES)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepressionClassifier(num_labels=len(CATEGORIES)).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    scheduler = get_scheduler("constant", optimizer=optimizer)

    # Print training setup
    print("Training Setup:")
    print(f"Device: {device}")
    print(f"Model: {model}")
    print(f"Max Sequence Length: {MAX_LEN}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Optimizer: AdamW (betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)")
    print(f"Scheduler: Constant")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Categories: {CATEGORIES}")

    train_and_evaluate(train_dataloader, val_dataloader, model, optimizer, scheduler, device, NUM_EPOCHS, OUTPUT_DIR)