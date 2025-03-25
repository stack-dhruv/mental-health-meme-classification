import json
import os
import random
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
RMS_NORM_EPSILON = 1e-5
DROPOUT = 0.2
LORA_RANK = 16
LORA_ALPHA = 8
TARGET_MODULES = ["Wq", "Wk", "Wv", "Wo"]

# Output directory
OUTPUT_DIR = os.path.join("mental-health-meme-classification", "src", "anxiety", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

# Data loading and preprocessing functions
def load_data(file_path):
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out samples without required fields
    filtered_data = []
    for sample in data:
        if 'sample_id' in sample and 'ocr_text' in sample and 'meme_anxiety_category' in sample:
            # Fix the misspelled "Irritatbily" to "Irritability"
            if sample['meme_anxiety_category'] == 'Irritatbily':
                sample['meme_anxiety_category'] = 'Irritability'
            # Change "Unknown" to "Unknown Anxiety"
            elif sample['meme_anxiety_category'] == 'Unknown':
                sample['meme_anxiety_category'] = 'Unknown Anxiety'
            filtered_data.append(sample)
    
    return filtered_data

def split_data(data, val_size=0.1, test_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    # If test_size is 0, it means we're using an external test set
    if test_size == 0:
        train_data, val_data = train_test_split(
            data, test_size=val_size, random_state=random_state, stratify=[d['meme_anxiety_category'] for d in data]
        )
        return train_data, val_data, []
    
    # First split: train+val and test
    train_val_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=[d['meme_anxiety_category'] for d in data]
    )
    
    # Second split: train and val
    val_ratio = val_size / (1 - test_size)  # Adjusted validation size
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_ratio, random_state=random_state, 
        stratify=[d['meme_anxiety_category'] for d in train_val_data]
    )
    
    return train_data, val_data, test_data

class AnxietyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, label_encoder):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["ocr_text"]
        label = item["meme_anxiety_category"]
        # Convert string label to numeric index using label encoder
        label_idx = self.label_encoder.transform([label])[0]
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
            "label": torch.tensor(label_idx, dtype=torch.long),
        }

# Model definition
class AnxietyClassifier(nn.Module):
    def __init__(self, num_labels, pretrained_model="bert-base-uncased"):  # Changed default model to bert-base-uncased
        super(AnxietyClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(  # Changed from BART to BERT
            pretrained_model,
            num_labels=num_labels,
            hidden_dropout_prob=DROPOUT  # Changed from classifier_dropout to hidden_dropout_prob for BERT
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

# Replace the train_model function with train_and_evaluate
def train_and_evaluate(
    train_dataloader, val_dataloader, model, optimizer, scheduler, device, num_epochs, output_dir
):
    """Training and evaluation function with comprehensive logging"""
    os.makedirs(output_dir, exist_ok=True)

    best_f1 = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    last_model_path = os.path.join(output_dir, "last_model.pt")
    log_file = os.path.join(output_dir, "training_logs.csv")
    logs = []

    # Initialize history dictionary to avoid None return
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1_macro': [],
        'val_f1_macro': [],
        'val_f1_weighted': [],
        'train_f1_weighted': []
    }

    try:
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
                labels = batch["label"].to(device)  # Note: using "label" instead of "labels"
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

                # Get predictions - use argmax instead of sigmoid for multiclass classification
                preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
                all_train_preds.extend(preds)
                all_train_labels.extend(labels.cpu().numpy())

                train_progress.set_postfix({"Loss": loss.item()})
            train_loss /= len(train_dataloader)

            # Calculate training metrics
            train_f1_macro = f1_score(all_train_labels, all_train_preds, average="macro")
            train_f1_weighted = f1_score(all_train_labels, all_train_preds, average="weighted")
            train_accuracy = accuracy_score(all_train_labels, all_train_preds)
            print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Macro F1: {train_f1_macro:.4f}, Weighted F1: {train_f1_weighted:.4f}")

            # Validation phase
            model.eval()
            val_loss = 0
            all_val_preds, all_val_labels = [], []
            val_progress = tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}")
            with torch.no_grad():
                for batch in val_progress:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)  # Note: using "label" instead of "labels"
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    
                    preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
                    all_val_preds.extend(preds)
                    all_val_labels.extend(labels.cpu().numpy())
                    
                    val_progress.set_postfix({"Loss": outputs.loss.item()})
            val_loss /= len(val_dataloader)

            # Calculate validation metrics
            val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")
            val_f1_weighted = f1_score(all_val_labels, all_val_preds, average="weighted")
            val_accuracy = accuracy_score(all_val_labels, all_val_preds)
            val_hamming = hamming_loss(all_val_labels, all_val_preds)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Macro F1: {val_f1_macro:.4f}, Weighted F1: {val_f1_weighted:.4f}, Hamming Loss: {val_hamming:.4f}")

            # Update history for this epoch
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_f1_macro'].append(train_f1_macro)
            history['val_f1_macro'].append(val_f1_macro)
            history['train_f1_weighted'].append(train_f1_weighted)
            history['val_f1_weighted'].append(val_f1_weighted)

            # Save logs for this epoch
            logs.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_f1_macro": train_f1_macro,
                "train_f1_weighted": train_f1_weighted,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_macro": val_f1_macro,
                "val_f1_weighted": val_f1_weighted,
                "val_hamming": val_hamming,
            })

            # Save the best model based on validation F1 score
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
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        # Even if there's an error, return whatever history we have so far
    
    # Always return the history
    return history

# Update the plot_training_history function to handle None or incomplete history
def plot_training_history(history):
    """Plot training and validation metrics"""
    # Check if history is None or empty
    if not history:
        logger.warning("No training history available for plotting")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss if available
    if 'train_loss' in history and 'val_loss' in history:
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    # Plot F1 Macro if available
    if 'train_f1_macro' in history and 'val_f1_macro' in history:
        plt.subplot(2, 2, 2)
        plt.plot(history['train_f1_macro'], label='Train Macro F1')
        plt.plot(history['val_f1_macro'], label='Validation Macro F1')
        plt.title('Training and Validation Macro F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
    
    # Plot F1 Weighted if available
    if 'val_f1_weighted' in history:
        plt.subplot(2, 2, 3)
        if 'train_f1_weighted' in history:
            plt.plot(history['train_f1_weighted'], label='Train Weighted F1')
        plt.plot(history['val_f1_weighted'], label='Validation Weighted F1')
        plt.title('Training and Validation Weighted F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

# Evaluation functions
def evaluate_model(model, dataloader, device, label_encoder):
    """Evaluate the model on the given dataloader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    hamming = hamming_loss(all_labels, all_preds)
    
    # Get classification report with class names
    # Only include classes that are actually present in the evaluation data
    unique_labels = sorted(set(all_labels).union(set(all_preds)))
    present_classes = [label_encoder.classes_[i] for i in unique_labels]
    
    report = classification_report(
        all_labels, all_preds, 
        labels=unique_labels,  # Specify which labels to include in the report
        target_names=present_classes, 
        digits=4
    )
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'hamming_loss': hamming,
        'report': report
    }

def predict(model, tokenizer, label_encoder, text, device, max_length=128):
    """Make prediction for a single text input"""
    model.eval()
    
    # Tokenize without preprocessing
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
    
    # Convert to class label
    pred_label = label_encoder.inverse_transform([pred_class])[0]
    
    # Get probability scores
    class_probs = probs[0].cpu().numpy()
    prob_dict = {label_encoder.classes_[i]: class_probs[i] for i in range(len(label_encoder.classes_))}
    
    return {
        'predicted_class': pred_label,
        'probabilities': prob_dict
    }

# Visualization functions
def plot_test_metrics(test_metrics_history):
    """Plot test metrics over epochs"""
    plt.figure(figsize=(10, 6))
    
    # Plot metrics
    epochs = range(1, len(test_metrics_history['macro_f1']) + 1)
    plt.plot(epochs, test_metrics_history['macro_f1'], 'b-', label='Macro F1')
    plt.plot(epochs, test_metrics_history['weighted_f1'], 'r-', label='Weighted F1')
    plt.title('Test Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_metrics_history.png')
    plt.close()

# Update the run_pipeline function to use train_and_evaluate
def run_pipeline(train_file, test_file=None, val_split=0.1, batch_size=16, num_epochs=5, 
                model_save_path='best_anxiety_model.pt', learning_rate=5e-5):
    """Run the full anxiety classification pipeline"""
    # Set seed for reproducibility
    set_seed(42)
    
    # Force CUDA as device
    device = torch.device('cuda')
    logger.info(f"Using device: {device}")
    
    # 1. Data Preparation
    logger.info("Loading and preparing data...")
    train_data = load_data(train_file)
    
    # If test file is provided, load it, otherwise split the train data
    if test_file:
        test_data = load_data(test_file)
        train_data, val_data, _ = split_data(train_data, val_size=val_split, test_size=0)
    else:
        train_data, val_data, test_data = split_data(train_data, val_size=val_split, test_size=0.2)
    
    logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")
    
    # Encode labels
    all_labels = [sample['meme_anxiety_category'] for sample in train_data + val_data]
    if test_data:
        all_labels += [sample['meme_anxiety_category'] for sample in test_data]
    
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    num_labels = len(label_encoder.classes_)
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Labels: {label_encoder.classes_}")
    
    # 2. Model Setup
    logger.info("Setting up model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = AnxietyClassifier(num_labels=num_labels, pretrained_model="bert-base-uncased")
    model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = AnxietyDataset(train_data, tokenizer, MAX_LEN, label_encoder)
    val_dataset = AnxietyDataset(val_data, tokenizer, MAX_LEN, label_encoder)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    if test_data:
        test_dataset = AnxietyDataset(test_data, tokenizer, MAX_LEN, label_encoder)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 3. Training Setup
    logger.info("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total training steps for the scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default: no warmup
        num_training_steps=total_steps
    )
    
    # Track test metrics across epochs
    test_metrics_history = {
        'accuracy': [],
        'macro_f1': [],
        'weighted_f1': [],
        'hamming_loss': []
    }
    
    # 4. Training and Validation
    logger.info("Starting training and validation...")
    history = train_and_evaluate(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        output_dir=OUTPUT_DIR
    )
    
    # Plot training history
    plot_training_history(history)
    
    # 5. Evaluation
    logger.info("Loading best model for evaluation...")
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_model(model, val_dataloader, device, label_encoder)
    logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
    logger.info(f"Validation Weighted F1: {val_metrics['weighted_f1']:.4f}")
    logger.info(f"Validation Hamming Loss: {val_metrics['hamming_loss']:.4f}")
    logger.info(f"Validation Report:\n{val_metrics['report']}")
    
    if test_data:
        logger.info("Evaluating on test set for each epoch...")
        # Evaluate for each epoch (using saved checkpoints)
        for epoch in range(1, num_epochs + 1):
            # Load model for this epoch (use last model from final epoch)
            last_model_path = os.path.join(os.path.dirname(model_save_path), f"last_{os.path.basename(model_save_path)}")
            if os.path.exists(last_model_path):
                model.load_state_dict(torch.load(last_model_path))
                
                # Evaluate
                test_metrics = evaluate_model(model, test_dataloader, device, label_encoder)
                
                # Store metrics for plotting
                test_metrics_history['accuracy'].append(test_metrics['accuracy'])
                test_metrics_history['macro_f1'].append(test_metrics['macro_f1'])
                test_metrics_history['weighted_f1'].append(test_metrics['weighted_f1'])
                test_metrics_history['hamming_loss'].append(test_metrics['hamming_loss'])
                
                logger.info(f"Epoch {epoch} Test Metrics:")
                logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
                logger.info(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
                logger.info(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
                logger.info(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
        
        # Plot test metrics history
        plot_test_metrics(test_metrics_history)
        
        # Final evaluation with best model
        model.load_state_dict(torch.load(model_save_path))
        logger.info("\nFinal Test Evaluation (Best Model):")
        test_metrics = evaluate_model(model, test_dataloader, device, label_encoder)
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
        logger.info(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
        logger.info(f"Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")
        logger.info(f"Test Report:\n{test_metrics['report']}")
    
    # Return the model, tokenizer, and label encoder for inference
    return model, tokenizer, label_encoder

# Example of how to use the pipeline
if __name__ == "__main__":
    base_dir = "mental-health-meme-classification"
    
    train_file = os.path.join(base_dir, "dataset", "Anxiety_Data", "anxiety_train.json")
    test_file = os.path.join(base_dir, "dataset", "Anxiety_Data", "anxiety_test.json")
    
    # Define model save path
    model_save_path = os.path.join(base_dir, "models", "anxiety", "best_anxiety_model.pt")
    
    # Run the pipeline
    model, tokenizer, label_encoder = run_pipeline(
        train_file=train_file,
        test_file=test_file,
        num_epochs=10,
        batch_size=16,
        model_save_path=model_save_path
    )
