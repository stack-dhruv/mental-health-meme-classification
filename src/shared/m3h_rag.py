# src/shared/mental_rag_fusion.py

import json
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import MultiheadAttention  # For Cross-Attention
from transformers import (
    BartTokenizer,
    BartModel,  # Base model for MentalBART
    get_linear_schedule_with_warmup,
    AdamW,
    CLIPVisionConfig  # Added for dimension check
)
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm  # Use standard tqdm for scripts
import logging
import re
from typing import List, Dict, Tuple, Optional, Any
import pickle
import gc  # Garbage collector
import torch.nn.functional as F  # For activation functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss, multilabel_confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import traceback
from collections import defaultdict

# --- Basic Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MentalRAGFusionPipeline")

# --- Constants & Hyperparameters ---

# Model Identifiers
MENTALBART_MODEL_NAME = "mental/mental-bart-base-cased"  # Or specific fine-tuned version if available
RAG_EMBEDDING_MODEL = "BAAI/bge-m3"  # Sentence Transformer for RAG
CLIP_FOR_DIM_CHECK = "openai/clip-vit-base-patch32"

# Paths (Relative to this script in 'shared' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Should be the 'shared' directory
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # Go up two levels from src/shared
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR_BASE = os.path.join(PROJECT_ROOT, "output", "mental_rag_fusion_output")  # Main output dir
DEPRESSION_DATA_DIR = os.path.join(DATASET_DIR, "Depressive_Data")
ANXIETY_DATA_DIR = os.path.join(DATASET_DIR, "Anxiety_Data")
VISUAL_FEATURES_DIR = os.path.join(MODELS_DIR, "visual_features")

# Training Settings
MAX_LEN_BART = 512  # Max sequence length for MentalBART input (Reasoning + RAG)
BATCH_SIZE = 16  # Updated batch size
NUM_EPOCHS = 10  # Epochs per run (adjust as needed)
LEARNING_RATE = 5e-5  # Updated learning rate
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 1e-2  # Updated weight decay (0.01)
DROPOUT_PROB = 0.2  # Updated dropout probability
NUM_ATTENTION_HEADS = 8  # For cross-attention module
GRADIENT_CLIPPING_THRESHOLD = 1.0  # Added gradient clipping threshold

# RAG Settings
RETRIEVAL_K = 3  # Number of examples to retrieve
RAG_EMBEDDING_BATCH_SIZE = 32

# Seed Setting
BASE_SEED = 42

# --- Get Visual Feature Dimension ---
VISUAL_FEATURE_DIM = 768  # Default value
try:
    clip_config = CLIPVisionConfig.from_pretrained(CLIP_FOR_DIM_CHECK)
    VISUAL_FEATURE_DIM = clip_config.projection_dim  # e.g., 512 for base, 768 for large
    logger.info(f"Determined Visual Feature Dimension from {CLIP_FOR_DIM_CHECK}: {VISUAL_FEATURE_DIM}")
except Exception as e:
    logger.warning(f"Error loading CLIP config: {e}. Defaulting VISUAL_FEATURE_DIM to {VISUAL_FEATURE_DIM}.")

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
logger.info(f"Expected Visual Feature Dimension: {VISUAL_FEATURE_DIM}")

# --- Seed Function ---
def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed set to {seed}")

# --- CUDA Memory Check Helper ---
def check_cuda_memory(step_name=""):
    """Logs CUDA memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.debug(f"CUDA Memory ({step_name}): Allocated={allocated:.3f} GB, Reserved={reserved:.3f} GB")
    else:
        logger.debug(f"CUDA not available ({step_name}). Skipping memory check.")

# --- Data Loading Function (Reads Qwen JSONs) ---
def load_qwen_data(file_path: str) -> List[Dict]:
    """Loads data from JSON/JSONL containing Qwen-VL results."""
    logger.info(f"Loading Qwen-VL data from: {file_path}")
    data = []
    processed_ids = set()
    is_anxiety = "Anxiety_Data" in file_path

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                raw_data = content if isinstance(content, list) else [content]
                logger.info(f"Loaded {len(raw_data)} samples as single JSON object/array.")
            except json.JSONDecodeError:
                logger.info(f"Initial JSON load failed, attempting to load as JSON Lines (.jsonl) from {file_path}")
                f.seek(0)
                raw_data = []
                for line_num, line in enumerate(f):
                    try:
                        if line.strip():
                            raw_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {line_num+1}: {line[:100]}...")
                if not raw_data:
                    logger.error("File appears empty or fully invalid after jsonl attempt.")
                    return []
                logger.info(f"Successfully loaded {len(raw_data)} lines as JSON Lines.")

        required_keys_base = ['sample_id', 'image_id', 'ocr_text', 'content']
        label_key_anxiety = 'meme_anxiety_category'
        label_key_depression = 'meme_depressive_categories'

        for idx, sample in enumerate(raw_data):
            if not isinstance(sample, dict):
                logger.warning(f"Skipping non-dict item at index {idx}")
                continue

            sample_id = sample.get('sample_id', sample.get('id'))
            if not sample_id or sample_id in processed_ids:
                continue
            processed_ids.add(sample_id)

            required_keys = required_keys_base + ([label_key_anxiety] if is_anxiety else [label_key_depression])
            if not all(k in sample for k in required_keys):
                continue

            item = {
                'id': sample_id,
                'image_id': sample.get('image_id', sample_id),
                'ocr_text': sample.get('ocr_text', '') or "",
                'qwen_reasoning': sample.get('content', '') or ""
            }

            if is_anxiety:
                label = sample.get(label_key_anxiety)
                if label == 'Irritatbily':
                    label = 'Irritability'
                elif label == 'Unknown':
                    label = 'Unknown Anxiety'
                item['original_labels'] = label
                item['stratify_label'] = label
            else:
                labels = sample.get(label_key_depression)
                if isinstance(labels, str):
                    labels = [l.strip() for l in labels.split(',') if l.strip()]
                if not isinstance(labels, list) or not labels:
                    labels = ["Unknown Depression"]
                processed_labels = [lbl if lbl != 'Unknown' else 'Unknown Depression' for lbl in labels]
                item['original_labels'] = processed_labels
                item['stratify_label'] = processed_labels[0]

            if not item['ocr_text'] and not item['qwen_reasoning']:
                logger.warning(f"Sample {sample_id} has empty OCR and Qwen reasoning.")

            data.append(item)

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Failed to read or process file {file_path}: {e}", exc_info=True)
        return []

    logger.info(f"Loaded and processed {len(data)} samples from {file_path}.")
    return data

# --- Helper Function for Anxiety Data Split ---
def prepare_anxiety_data_split(
    data: List[Dict],
    train_ratio: float = 0.85,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Splits the loaded anxiety data list into training and validation sets."""
    if not data:
        logger.warning("prepare_anxiety_data_split received empty data list. Returning empty lists.")
        return [], []

    data_copy = list(data)
    random.seed(random_seed)
    random.shuffle(data_copy)
    logger.info(f"Shuffled {len(data_copy)} items using seed {random_seed}.")

    train_size = int(len(data_copy) * train_ratio)
    if train_size == 0 or train_size == len(data_copy):
        logger.warning(f"Train ratio {train_ratio} resulted in an invalid split size ({train_size} for train).")
        return data_copy, []

    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:]

    logger.info(f"Split anxiety dataset: {len(train_data)} training samples, {len(val_data)} validation samples.")
    return train_data, val_data

# --- RAG Components (Placeholders) ---

def build_rag_index(documents: List[str], embedding_model, batch_size: int, device) -> faiss.Index:
    """Builds a FAISS index for the provided documents."""
    logger.info(f"Building FAISS index for {len(documents)} documents...")
    logger.info("FAISS index built (Placeholder).")
    return None # Placeholder

def retrieve_rag_documents(query: str, index: faiss.Index, embedding_model, documents: List[str], k: int, device) -> List[str]:
    """Retrieves the top-k relevant documents for a query."""
    logger.debug(f"Retrieving {k} documents for query (first 50 chars): {query[:50]}...")
    logger.debug("Document retrieval complete (Placeholder).")
    return ["Retrieved document 1 (placeholder)", "Retrieved document 2 (placeholder)"] # Placeholder

# --- Dataset Class ---

class MentalRAGFusionDataset(Dataset):
    """Dataset for MentalRAGFusion model."""
    def __init__(self, data: List[Dict], tokenizer, max_len: int, rag_index: Optional[faiss.Index],
                 rag_embedding_model, all_documents: List[str], retrieve_k: int, device,
                 label_encoder: Optional[Any] = None, is_multilabel: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rag_index = rag_index
        self.rag_embedding_model = rag_embedding_model
        self.all_documents = all_documents
        self.retrieve_k = retrieve_k
        self.device = device
        self.label_encoder = label_encoder
        self.is_multilabel = is_multilabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ocr_text = item.get('ocr_text', '')
        qwen_reasoning = item.get('qwen_reasoning', '')
        image_id = item.get('image_id', item['id'])

        combined_text = f"OCR: {ocr_text}\nReasoning: {qwen_reasoning}".strip()
        if not combined_text:
            combined_text = "No text available."

        retrieved_docs_text = ""
        if self.rag_index and self.rag_embedding_model:
            try:
                retrieved_docs = retrieve_rag_documents(
                    combined_text, self.rag_index, self.rag_embedding_model,
                    self.all_documents, self.retrieve_k, self.device
                )
                retrieved_docs_text = "\nRetrieved Context:\n" + "\n".join(retrieved_docs)
            except Exception as e:
                logger.warning(f"RAG retrieval failed for item {idx}: {e}. Proceeding without retrieved context.")
                retrieved_docs_text = "\nRetrieved Context: [Retrieval Failed]"

        final_input_text = combined_text + retrieved_docs_text

        encoding = self.tokenizer(
            final_input_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        visual_feature_path = os.path.join(VISUAL_FEATURES_DIR, f"{image_id}.pt")
        try:
            visual_features = torch.load(visual_feature_path, map_location='cpu')
            if visual_features.shape[-1] != VISUAL_FEATURE_DIM:
                 logger.warning(f"Visual feature dim mismatch for {image_id}: Expected {VISUAL_FEATURE_DIM}, Got {visual_features.shape[-1]}. Using zeros.")
                 visual_features = torch.zeros(1, VISUAL_FEATURE_DIM)
            if visual_features.dim() > 2:
                 visual_features = visual_features.mean(dim=1)
            visual_features = visual_features.squeeze(0)

        except FileNotFoundError:
            logger.warning(f"Visual features not found for {image_id} at {visual_feature_path}. Using zeros.")
            visual_features = torch.zeros(VISUAL_FEATURE_DIM)
        except Exception as e:
            logger.error(f"Error loading visual features for {image_id}: {e}. Using zeros.")
            visual_features = torch.zeros(VISUAL_FEATURE_DIM)

        labels = item['original_labels']
        if self.is_multilabel:
            target = torch.tensor(self.label_encoder.transform([labels])[0], dtype=torch.float)
        else:
            target = torch.tensor(self.label_encoder.transform([labels])[0], dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'visual_features': visual_features,
            'labels': target
        }

# --- Model Architecture (Placeholder) ---

class MentalRAGFusionModel(nn.Module):
    def __init__(self, mental_bart_model_name, visual_dim, num_labels, dropout_prob, num_attention_heads):
        super().__init__()
        logger.info("Initializing MentalRAGFusionModel...")
        self.bart = BartModel.from_pretrained(mental_bart_model_name)
        self.config = self.bart.config
        self.dropout = nn.Dropout(dropout_prob)

        self.visual_projection = nn.Linear(visual_dim, self.config.hidden_size)

        self.cross_attention = MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        self.layer_norm_cross = nn.LayerNorm(self.config.hidden_size)

        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        logger.info("MentalRAGFusionModel initialized.")

    def forward(self, input_ids, attention_mask, visual_features):
        encoder_outputs = self.bart.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_features = encoder_outputs.last_hidden_state

        projected_visual = self.visual_projection(visual_features)
        projected_visual_unsqueezed = projected_visual.unsqueeze(1)

        attn_output, _ = self.cross_attention(
            query=text_features,
            key=projected_visual_unsqueezed,
            value=projected_visual_unsqueezed,
            key_padding_mask=None
        )

        fused_features = self.layer_norm_cross(text_features + attn_output)

        masked_features = fused_features * attention_mask.unsqueeze(-1)
        summed_features = masked_features.sum(dim=1)
        num_non_padding = attention_mask.sum(dim=1).unsqueeze(-1)
        pooled_output = summed_features / num_non_padding.clamp(min=1e-9)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

# --- Training and Evaluation Functions (Placeholders) ---

def train_epoch(model, data_loader, optimizer, device, scheduler, loss_fn, grad_clip_thresh, is_multilabel):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        visual_features = batch['visual_features'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, visual_features=visual_features)

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(data_loader)
    logger.info(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_model(model, data_loader, device, loss_fn, label_encoder, is_multilabel):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual_features = batch['visual_features'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, visual_features=visual_features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            if is_multilabel:
                preds = torch.sigmoid(logits) > 0.5
                all_preds.extend(preds.cpu().numpy().astype(int))
                all_labels.extend(labels.cpu().numpy().astype(int))
            else:
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    logger.info(f"Average Validation Loss: {avg_loss:.4f}")

    if is_multilabel:
        accuracy = accuracy_score(all_labels, all_preds)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        hamming = hamming_loss(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)

        logger.info(f"Validation Accuracy (Subset): {accuracy:.4f}")
        logger.info(f"Validation F1 Micro: {f1_micro:.4f}")
        logger.info(f"Validation F1 Macro: {f1_macro:.4f}")
        logger.info(f"Validation F1 Weighted: {f1_weighted:.4f}")
        logger.info(f"Validation Hamming Loss: {hamming:.4f}")
        logger.info(f"Validation Classification Report:\n{report}")
        return {'loss': avg_loss, 'accuracy': accuracy, 'f1_micro': f1_micro, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'hamming': hamming, 'report': report}

    else:
        accuracy = accuracy_score(all_labels, all_preds)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)

        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info(f"Validation F1 Micro: {f1_micro:.4f}")
        logger.info(f"Validation F1 Macro: {f1_macro:.4f}")
        logger.info(f"Validation F1 Weighted: {f1_weighted:.4f}")
        logger.info(f"Validation Classification Report:\n{report}")
        return {'loss': avg_loss, 'accuracy': accuracy, 'f1_micro': f1_micro, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'report': report}

# --- Main Pipeline Function ---

def run_mental_rag_pipeline(dataset_type: str, seed: int):
    """Runs the full Mental-RAG Fusion pipeline for a given dataset type."""
    run_start_time = time.time()
    set_seed(seed)
    # Define the base directory for cleaned data within each dataset type
    cleaned_data_base = os.path.join(DATASET_DIR, "{data_type}", "final", "cleaned")

    output_dir = os.path.join(OUTPUT_DIR_BASE, f"{dataset_type}_run_{seed}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting pipeline for dataset: {dataset_type.upper()}, Seed: {seed}")
    logger.info(f"Output directory: {output_dir}")
    check_cuda_memory("Pipeline Start")

    is_multilabel = dataset_type == 'depression'
    train_data = []
    val_data = []

    if dataset_type == 'depression':
        depression_cleaned_dir = cleaned_data_base.format(data_type="Depressive_Data")
        train_file = os.path.join(depression_cleaned_dir, "depressive_train_combined_preprocessed.json")
        val_file = os.path.join(depression_cleaned_dir, "depressive_val_combined_preprocessed.json")

        logger.info(f"Loading Depression training data from: {train_file}")
        train_data = load_qwen_data(train_file)
        logger.info(f"Loading Depression validation data from: {val_file}")
        val_data = load_qwen_data(val_file)

        if not train_data:
             logger.error("No depression training data loaded. Exiting.")
             return
        if not val_data:
             logger.warning("No depression validation data loaded. Proceeding without validation.")
             if train_data:
                 logger.info("Creating validation split from training data as validation file was empty/not found.")
                 train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=seed, stratify=[d['stratify_label'] for d in train_data])

    elif dataset_type == 'anxiety':
        anxiety_cleaned_dir = cleaned_data_base.format(data_type="Anxiety_Data")
        train_file = os.path.join(anxiety_cleaned_dir, "anxiety_train_combined_preprocessed.json")
        test_file = os.path.join(anxiety_cleaned_dir, "anxiety_test_combined_preprocessed.json")

        logger.info(f"Loading Anxiety training data from: {train_file}")
        train_data_full = load_qwen_data(train_file)

        if not train_data_full:
             logger.error("No anxiety training data loaded. Exiting.")
             return

        logger.info("Splitting loaded anxiety training data into train/validation sets (85/15 split).")
        try:
            train_data, val_data = train_test_split(train_data_full, test_size=0.15, random_state=seed, stratify=[d['stratify_label'] for d in train_data_full])
        except KeyError:
            logger.warning("Stratify label not found for anxiety split, using random split.")
            train_data, val_data = train_test_split(train_data_full, test_size=0.15, random_state=seed)

    else:
        logger.error(f"Invalid dataset type: {dataset_type}")
        return

    if not train_data or not val_data:
        logger.error("Data loading/splitting resulted in empty train or validation set. Exiting.")
        return

    logger.info(f"Final Data Split - Train: {len(train_data)}, Validation: {len(val_data)}")

    if is_multilabel:
        all_train_labels = [lbl for item in train_data for lbl in item['original_labels']]
        unique_labels = sorted(list(set(all_train_labels)))
        label_encoder = MultiLabelBinarizer(classes=unique_labels)
        label_encoder.fit([item['original_labels'] for item in train_data])
        num_labels = len(label_encoder.classes_)
        logger.info(f"Multi-label classification. Found {num_labels} unique labels: {label_encoder.classes_}")
    else:
        label_encoder = LabelEncoder()
        train_labels = [item['original_labels'] for item in train_data]
        label_encoder.fit(train_labels)
        num_labels = len(label_encoder.classes_)
        logger.info(f"Single-label classification. Found {num_labels} unique labels: {label_encoder.classes_}")

    logger.info(f"Loading tokenizer: {MENTALBART_MODEL_NAME}")
    tokenizer = BartTokenizer.from_pretrained(MENTALBART_MODEL_NAME)
    logger.info(f"Loading RAG embedding model: {RAG_EMBEDDING_MODEL}")
    rag_embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL, device=device)
    check_cuda_memory("After loading models")

    rag_corpus = [f"OCR: {item.get('ocr_text', '')}\nReasoning: {item.get('qwen_reasoning', '')}".strip()
                  for item in train_data]
    rag_corpus = [doc if doc else "No text available." for doc in rag_corpus]

    logger.info("Building RAG index...")
    rag_index = build_rag_index(rag_corpus, rag_embedding_model, RAG_EMBEDDING_BATCH_SIZE, device)
    check_cuda_memory("After building RAG index")

    logger.info("Creating datasets and dataloaders...")
    train_dataset = MentalRAGFusionDataset(
        train_data, tokenizer, MAX_LEN_BART, rag_index, rag_embedding_model, rag_corpus, RETRIEVAL_K, device, label_encoder, is_multilabel
    )
    val_dataset = MentalRAGFusionDataset(
        val_data, tokenizer, MAX_LEN_BART, rag_index, rag_embedding_model, rag_corpus, RETRIEVAL_K, device, label_encoder, is_multilabel
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    logger.info(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader)} batches.")
    check_cuda_memory("After creating dataloaders")

    model = MentalRAGFusionModel(
        mental_bart_model_name=MENTALBART_MODEL_NAME,
        visual_dim=VISUAL_FEATURE_DIM,
        num_labels=num_labels,
        dropout_prob=DROPOUT_PROB,
        num_attention_heads=NUM_ATTENTION_HEADS
    )
    model.to(device)
    check_cuda_memory("After moving model to device")

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    logger.info("Using constant learning rate scheduler.")

    if is_multilabel:
        loss_fn = nn.BCEWithLogitsLoss()
        logger.info("Using BCEWithLogitsLoss for multi-label task.")
    else:
        loss_fn = nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss for single-label task.")

    logger.info("Starting training...")
    best_val_metric = -1
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        logger.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        check_cuda_memory(f"Epoch {epoch+1} Start")

        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler, loss_fn, GRADIENT_CLIPPING_THRESHOLD, is_multilabel)
        check_cuda_memory(f"Epoch {epoch+1} After Train")

        val_metrics = evaluate_model(model, val_loader, device, loss_fn, label_encoder, is_multilabel)
        check_cuda_memory(f"Epoch {epoch+1} After Eval")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_metrics'].append(val_metrics)

        current_metric = val_metrics.get('f1_macro', val_metrics.get('accuracy', -1))

        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            logger.info(f"New best validation metric ({'f1_macro' if 'f1_macro' in val_metrics else 'accuracy'}): {best_val_metric:.4f}. Saving model...")
            model_save_path = os.path.join(output_dir, f"best_model_seed_{seed}.pt")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved to {model_save_path}")

            le_save_path = os.path.join(output_dir, f"label_encoder_seed_{seed}.pkl")
            with open(le_save_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            logger.info(f"Label encoder saved to {le_save_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Training finished.")
    history_save_path = os.path.join(output_dir, f"training_history_seed_{seed}.json")
    try:
        serializable_history = json.loads(json.dumps(history, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
        with open(history_save_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        logger.info(f"Training history saved to {history_save_path}")
    except Exception as e:
        logger.error(f"Could not serialize or save training history: {e}")

    run_end_time = time.time()
    logger.info(f"Pipeline run for seed {seed} completed in {(run_end_time - run_start_time)/60:.2f} minutes.")

# --- Execution ---
if __name__ == "__main__":
    logger.info("========================================")
    logger.info("||   Mental-RAG Script Execution    ||")
    logger.info("========================================")
    script_start_time = time.time()

    set_seed(BASE_SEED)

    dataset = 'depression'

    logger.info(f"Selected dataset for pipeline: {dataset.upper()}")

    try:
        run_mental_rag_pipeline(
            dataset_type=dataset,
            seed=BASE_SEED
        )
    except NameError as ne:
         logger.error(f"Function 'run_mental_rag_pipeline' is not defined: {ne}", exc_info=True)
    except Exception as main_e:
        logger.critical(f"Unhandled exception occurred in the main execution block: {main_e}", exc_info=True)
        try:
            os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
            tb_path = os.path.join(OUTPUT_DIR_BASE, f"{dataset}_CRASH_traceback_{time.strftime('%Y%m%d_%H%M%S')}.txt")
            with open(tb_path, "w") as f:
                traceback.print_exc(file=f)
            logger.info(f"Crash traceback saved to: {tb_path}")
        except Exception as tb_save_e:
            logger.error(f"Could not save crash traceback: {tb_save_e}")

    script_end_time = time.time()
    logger.info("----------------------------------------")
    logger.info(f"Script execution finished in {(script_end_time - script_start_time)/60:.2f} minutes.")
    logger.info("========================================")