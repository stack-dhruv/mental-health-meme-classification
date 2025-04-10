import json
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartForSequenceClassification,
    BartTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import re
from typing import List, Dict, Tuple, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hyperparameters
MAX_LEN = 512  # Increased for combined prompts
BATCH_SIZE = 8  # Smaller batch size due to longer sequences
NUM_EPOCHS = 20
LEARNING_RATE = 3e-5
EMBEDDING_DIM = 1024  # For BAAI/bge-m3
RETRIEVAL_K = 3  # Number of examples to retrieve
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
DROPOUT = 0.1

# Output directory
OUTPUT_DIR = os.path.join("mental-health-meme-classification", "src", "anxiety", "output", "knowledge_fusion")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    """Load data from JSON file for both anxiety and depression datasets"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Detect dataset type based on file path
    is_anxiety = "Anxiety_Data" in file_path
    
    # Filter out samples without required fields
    filtered_data = []
    
    for sample in data:
        # Anxiety dataset processing
        if is_anxiety and 'sample_id' in sample and 'ocr_text' in sample and 'meme_anxiety_category' in sample:
            # Fix the misspelled "Irritatbily" to "Irritability"
            if sample['meme_anxiety_category'] == 'Irritatbily':
                sample['meme_anxiety_category'] = 'Irritability'
            # Change "Unknown" to "Unknown Anxiety"
            elif sample['meme_anxiety_category'] == 'Unknown':
                sample['meme_anxiety_category'] = 'Unknown Anxiety'
            
            # Add empty triples if not present
            if 'triples' not in sample:
                sample['triples'] = ""
            
            filtered_data.append(sample)
        
        # Depression dataset processing
        elif not is_anxiety and 'ocr_text' in sample and 'labels' in sample:
            # Convert depression dataset format to match anxiety format
            # This assumes depression uses multilabel format in 'labels'
            # Adapt as needed based on actual depression data structure
            sample['meme_anxiety_category'] = sample['labels'][0] if sample['labels'] else "Unknown"
            
            # Add empty triples if not present
            if 'triples' not in sample:
                sample['triples'] = ""
            
            filtered_data.append(sample)
    
    return filtered_data


def clean_triples(triples_text):
    """Clean and extract structured information from triples text"""
    if pd.isna(triples_text) or not triples_text:
        return ""
    
    # Extract key sections
    sections = ["Cause-Effect", "Figurative Understanding", "Mental State"]
    cleaned_text = ""
    
    for section in sections:
        pattern = rf"{section}:(.*?)(?:(?:{sections[0]}|{sections[1]}|{sections[2]}):|$)"
        match = re.search(pattern, triples_text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            cleaned_text += f"{section}: {content}\n"
    
    return cleaned_text.strip()

def split_data(data, val_size=0.1, test_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    # If test_size is 0, it means we're using an external test set
    if test_size == 0:
        train_data, val_data = train_test_split(
            data, test_size=val_size, random_state=random_state, 
            stratify=[d['meme_anxiety_category'] for d in data]
        )
        return train_data, val_data, []
    
    # First split: train+val and test
    train_val_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, 
        stratify=[d['meme_anxiety_category'] for d in data]
    )
    
    # Second split: train and val
    val_ratio = val_size / (1 - test_size)  # Adjusted validation size
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_ratio, random_state=random_state, 
        stratify=[d['meme_anxiety_category'] for d in train_val_data]
    )
    
    return train_data, val_data, test_data

class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformer"""
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(texts, show_progress_bar=True, 
                                       convert_to_numpy=True)
        return embeddings
    
    def generate_fused_embeddings(self, ocr_texts, triples_texts):
        """Generate and concatenate embeddings for OCR and triples"""
        ocr_embeddings = self.generate_embeddings(ocr_texts)
        triples_embeddings = self.generate_embeddings(triples_texts)
        
        # Normalize each embedding before concatenation
        ocr_norm = np.linalg.norm(ocr_embeddings, axis=1, keepdims=True)
        triples_norm = np.linalg.norm(triples_embeddings, axis=1, keepdims=True)
        
        ocr_embeddings_normalized = ocr_embeddings / ocr_norm
        triples_embeddings_normalized = triples_embeddings / triples_norm
        
        # Concatenate
        fused_embeddings = np.concatenate([ocr_embeddings_normalized, triples_embeddings_normalized], axis=1)
        return fused_embeddings

class RAGRetriever:
    """Retrieval-Augmented Generation system using FAISS for similarity search"""
    def __init__(self, embeddings, top_k=RETRIEVAL_K):
        self.top_k = top_k
        self.index = None
        self.build_index(embeddings)
        
    def build_index(self, embeddings):
        """Build FAISS index for fast similarity search"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
    
    def retrieve_similar(self, query_embeddings):
        """Retrieve top-k similar examples for each query"""
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k=self.top_k)
        return indices

class PromptConstructor:
    """Construct prompts for few-shot learning"""
    def __init__(self, train_data, label_encoder):
        self.train_data = train_data
        self.label_encoder = label_encoder
        
    def construct_prompt(self, sample, retrieved_indices=None):
        """Construct a prompt with retrieved examples"""
        system_instruction = "Classify the meme based on its anxiety category. Choose from: Restlessness, Nervousness, Impending Doom, Difficulty Relaxing, Lack of Worry Control, Excessive Worry, Unknown Anxiety, Irritability."
        
        prompt = f"{system_instruction}\n\n"
        
        # Add retrieved examples if provided
        if retrieved_indices is not None:
            prompt += "Here are some similar examples:\n\n"
            for idx in retrieved_indices:
                ex = self.train_data[idx]
                ex_text = ex["ocr_text"]
                ex_triples = ex.get("triples", "")
                ex_label = ex["meme_anxiety_category"]
                
                prompt += f"Example Text: {ex_text}\n"
                if ex_triples:
                    prompt += f"Example Knowledge: {ex_triples}\n"
                prompt += f"Category: {ex_label}\n\n"
            
            prompt += "Now classify this new example:\n\n"
        
        # Add the current sample
        prompt += f"Text: {sample['ocr_text']}\n"
        if sample.get("triples", ""):
            prompt += f"Knowledge: {sample['triples']}\n"
        prompt += "Category:"
        
        return prompt

class AnxietyDataset(Dataset):
    """Dataset for anxiety meme classification with knowledge fusion"""
    def __init__(self, samples, prompts, tokenizer, max_len, label_encoder):
        self.samples = samples
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self.prompts[idx]
        label = sample["meme_anxiety_category"]
        
        # Convert string label to numeric index using label encoder
        label_idx = self.label_encoder.transform([label])[0]
        
        encoding = self.tokenizer(
            prompt,
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

class AnxietyClassifierWithFusion(nn.Module):
    """MentalBART-based classifier with knowledge fusion"""
    def __init__(self, num_labels, pretrained_model="facebook/bart-base"):
        super(AnxietyClassifierWithFusion, self).__init__()
        self.bart = BartForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            classifier_dropout=DROPOUT
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

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

    # Initialize history dictionary
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
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

                # Get predictions - use argmax for multiclass classification
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
                    labels = batch["label"].to(device)
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
    
    # Return the history
    return history

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
    unique_labels = sorted(set(all_labels).union(set(all_preds)))
    present_classes = [label_encoder.classes_[i] for i in unique_labels]
    
    report = classification_report(
        all_labels, all_preds, 
        labels=unique_labels,
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

def predict(model, tokenizer, label_encoder, text, triples="", device=None, max_length=512):
    """Make prediction for a single example with OCR text and triples"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Construct prompt without retrieved examples
    prompt = f"Classify the meme based on its anxiety category.\n\nText: {text}\n"
    if triples:
        prompt += f"Knowledge: {triples}\n"
    prompt += "Category:"
    
    # Tokenize
    encoding = tokenizer(
        prompt,
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

def run_pipeline(train_file, test_file=None, val_split=0.1, batch_size=8, num_epochs=20):
    """Run the full anxiety classification pipeline with knowledge fusion and RAG"""
    # Set seed for reproducibility
    set_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data) if test_data else 0}")
    
    # Clean triples data
    for dataset in [train_data, val_data]:
        if dataset:
            for sample in dataset:
                if 'triples' in sample:
                    sample['triples'] = clean_triples(sample['triples'])
    
    if test_data:
        for sample in test_data:
            if 'triples' in sample:
                sample['triples'] = clean_triples(sample['triples'])
    
    # Encode labels
    all_labels = [sample['meme_anxiety_category'] for sample in train_data + val_data]
    if test_data:
        all_labels += [sample['meme_anxiety_category'] for sample in test_data]
    
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    num_labels = len(label_encoder.classes_)
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Labels: {label_encoder.classes_}")
    
    # 2. Generate Embeddings
    logger.info("Generating embeddings...")
    embedding_generator = EmbeddingGenerator(model_name="BAAI/bge-m3")
    
    # Extract OCR texts and triples
    train_ocr_texts = [sample['ocr_text'] for sample in train_data]
    train_triples_texts = [sample.get('triples', '') for sample in train_data]
    
    # Generate fused embeddings for training data
    train_fused_embeddings = embedding_generator.generate_fused_embeddings(
        train_ocr_texts, train_triples_texts
    )
    
    # 3. Build RAG retriever
    logger.info("Building RAG retriever...")
    retriever = RAGRetriever(train_fused_embeddings, top_k=RETRIEVAL_K)
    
    # 4. Prepare prompts with retrieved examples
    logger.info("Preparing training prompts...")
    prompt_constructor = PromptConstructor(train_data, label_encoder)
    
    # Generate prompts for training data
    train_prompts = []
    for i, sample in enumerate(train_data):
        # For training data, we'll use the top-k most similar samples excluding itself
        # Get top k+1 to include itself, then exclude itself
        similar_indices = retriever.retrieve_similar(
            train_fused_embeddings[i:i+1]
        )[0][1:]  # Skip the first one (itself)
        train_prompts.append(prompt_constructor.construct_prompt(sample, similar_indices))
    
    # Generate prompts for validation data
    val_ocr_texts = [sample['ocr_text'] for sample in val_data]
    val_triples_texts = [sample.get('triples', '') for sample in val_data]
    
    val_fused_embeddings = embedding_generator.generate_fused_embeddings(
        val_ocr_texts, val_triples_texts
    )
    
    val_prompts = []
    for i, sample in enumerate(val_data):
        similar_indices = retriever.retrieve_similar(
            val_fused_embeddings[i:i+1]
        )[0]
        val_prompts.append(prompt_constructor.construct_prompt(sample, similar_indices))
    
    # 5. Model Setup
    logger.info("Setting up model and tokenizer...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = AnxietyClassifierWithFusion(num_labels=num_labels)
    model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = AnxietyDataset(train_data, train_prompts, tokenizer, MAX_LEN, label_encoder)
    val_dataset = AnxietyDataset(val_data, val_prompts, tokenizer, MAX_LEN, label_encoder)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 6. Training Setup
    logger.info("Setting up optimizer and scheduler...")
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Calculate total training steps for the scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 7. Training and Validation
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
    
    # 8. Evaluation
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
    
    # If test data is available, evaluate on it
    if test_data:
        # Prepare test prompts
        test_ocr_texts = [sample['ocr_text'] for sample in test_data]
        test_triples_texts = [sample.get('triples', '') for sample in test_data]
        
        test_fused_embeddings = embedding_generator.generate_fused_embeddings(
            test_ocr_texts, test_triples_texts
        )
        
        test_prompts = []
        for i, sample in enumerate(test_data):
            similar_indices = retriever.retrieve_similar(
                test_fused_embeddings[i:i+1]
            )[0]
            test_prompts.append(prompt_constructor.construct_prompt(sample, similar_indices))
        
        test_dataset = AnxietyDataset(test_data, test_prompts, tokenizer, MAX_LEN, label_encoder)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(model, test_dataloader, device, label_encoder)
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
        logger.info(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
        logger.info(f"Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")
        logger.info(f"Test Report:\n{test_metrics['report']}")
    
    # Save label encoder
    label_encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    import pickle
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Return the model, tokenizer, and label encoder for inference
    return model, tokenizer, label_encoder

if __name__ == "__main__":
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Train mental health meme classification model')
    parser.add_argument('--dataset', type=str, default='anxiety', choices=['anxiety', 'depression'],
                        help='Dataset to use: anxiety or depression')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--use-test', action='store_true',
                        help='Use the test file for final evaluation')
    
    args = parser.parse_args()
    
    # Get the absolute path to the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Set file paths based on dataset type
    if args.dataset == "anxiety":
        train_file = os.path.join(project_root, "dataset", "Anxiety_Data", "anxiety_train.json")
        test_file = os.path.join(project_root, "dataset", "Anxiety_Data", "anxiety_test.json") if args.use_test else None
        output_subdir = "anxiety"
    else:  # depression
        train_file = os.path.join(project_root, "dataset", "Depressive_Data", "train.json")
        test_file = os.path.join(project_root, "dataset", "Depressive_Data", "test.json") if args.use_test else None
        output_subdir = "depression"
    
    # Update output directory to use absolute paths
    OUTPUT_DIR = os.path.join(project_root, "src", output_subdir, "output", "knowledge_fusion")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define model save path
    model_save_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    
    # Actually run the training pipeline
    model, tokenizer, label_encoder = run_pipeline(
        train_file=train_file,
        test_file=test_file,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info(f"Training completed. Model saved to {model_save_path}")