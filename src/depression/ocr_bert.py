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
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# 3. Training Loop
def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Updated evaluation function to return F1 score and Hamming loss
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int), average="micro")
    hamming = hamming_loss(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return f1, hamming

# Training function with logging and model saving
def train_and_evaluate(
    train_dataloader, val_dataloader, model, optimizer, scheduler, device, num_epochs, output_dir
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables to track the best model
    best_f1 = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    last_model_path = os.path.join(output_dir, "last_model.pt")

    # CSV file to store logs
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
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            # Collect predictions and labels for F1 calculation
            preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()  # Fixed here
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

            train_progress.set_postfix({"Loss": loss.item()})
        train_loss /= len(train_dataloader)

        # Calculate training metrics
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
                preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()  # Fixed here
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_f1_macro = f1_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(int), average="macro")
        val_f1_weighted = f1_score(all_val_labels, (np.array(all_val_preds) > 0.5).astype(int), average="weighted")
        val_hamming = hamming_loss(all_val_labels, (np.array(all_val_preds) > 0.5).astype(int))
        print(f"Validation Macro F1: {val_f1_macro:.4f}, Weighted F1: {val_f1_weighted:.4f}, Hamming Loss: {val_hamming:.4f}")

        # Save logs for this epoch
        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_f1_macro": train_f1_macro,
            "train_f1_weighted": train_f1_weighted,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
            "val_hamming": val_hamming,
        })

        # Save the best model
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1}")

        # Save the last model after every epoch
        torch.save(model.state_dict(), last_model_path)

        # Save logs to CSV after every epoch
        pd.DataFrame(logs).to_csv(log_file, index=False)

    print(f"Last model saved at epoch {num_epochs}")
    print(f"Training logs saved to {log_file}")

# Main script to train the model
if __name__ == "__main__":
    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-5
    # Point to the output directory where it will be located into the same directory as this code file located
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Categories
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

    # point to the right relative path from the sys so I don't have to change it on the other machine: path is <repository_location>/dataset/Depressive_Data/train.json
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'Depressive_Data')))

    
    # Train data path
    train_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'Depressive_Data', 'train.json')
    val_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'Depressive_Data', 'val.json')

    # Load data
    with open(train_data_path, 'r') as train_file:
        train_data = json.load(train_file)

    with open(val_data_path, 'r') as val_file:
        val_data = json.load(val_file)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Datasets and DataLoaders
    train_dataset = DepressionDataset(train_data, tokenizer, MAX_LEN, CATEGORIES)
    val_dataset = DepressionDataset(val_data, tokenizer, MAX_LEN, CATEGORIES)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepressionClassifier(num_labels=len(CATEGORIES)).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * NUM_EPOCHS)

    # Train and evaluate
    train_and_evaluate(train_dataloader, val_dataloader, model, optimizer, scheduler, device, NUM_EPOCHS, OUTPUT_DIR)