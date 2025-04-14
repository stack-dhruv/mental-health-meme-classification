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
from torch.nn import MultiheadAttention # For Cross-Attention
from transformers import (
    BartTokenizer,
    BartModel, # Base model for MentalBART
    get_linear_schedule_with_warmup,
    AdamW
)
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm # Use standard tqdm for scripts
import logging
import re
from typing import List, Dict, Tuple, Optional, Any
import pickle
import gc # Garbage collector
import torch.nn.functional as F # For activation functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder
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
MENTALBART_MODEL_NAME = "mental/mental-bart-base-cased" # Or specific fine-tuned version if available
RAG_EMBEDDING_MODEL = "BAAI/bge-m3" # Sentence Transformer for RAG
# CLIP Model Name (used only to get expected dim, features assumed precomputed)
# This should match the model used to *generate* the .pt features
CLIP_FOR_DIM_CHECK = "openai/clip-vit-base-patch32"

# Paths (Relative to this script in 'shared' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Should be the 'shared' directory
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..")) # Go up two levels from src/shared
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR_BASE = os.path.join(PROJECT_ROOT, "output", "mental_rag_fusion_output") # Main output dir

ANXIETY_DATA_DIR = os.path.join(DATASET_DIR, "Anxiety_Data")
DEPRESSION_DATA_DIR = os.path.join(DATASET_DIR, "Depressive_Data")
# Assumes CLIP features are here, matching notebook logic. Adjust if different.
VISUAL_FEATURES_DIR = os.path.join(MODELS_DIR, "visual_features")

# Training Settings
MAX_LEN_BART = 512     # Max sequence length for MentalBART input (Reasoning + RAG)
BATCH_SIZE = 6         # START SMALLER - RAG adds context length
NUM_EPOCHS = 10        # Epochs per ensemble member
LEARNING_RATE = 2e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
DROPOUT_PROB = 0.2
NUM_ATTENTION_HEADS = 8 # For cross-attention module

# RAG Settings
RETRIEVAL_K = 3          # Number of examples to retrieve
RAG_EMBEDDING_BATCH_SIZE = 32

# Ensemble Settings
NUM_ENSEMBLE_MODELS = 3
BASE_SEED = 42

# --- Get Visual Feature Dimension ---
# Load config of the model used for feature extraction to get expected dimension
try:
    from transformers import CLIPVisionConfig
    # Try loading config, handle potential errors if model not locally cached/available
    try:
        clip_config = CLIPVisionConfig.from_pretrained(CLIP_FOR_DIM_CHECK)
        VISUAL_FEATURE_DIM = clip_config.projection_dim # e.g., 512 for base, 768 for large
    except OSError: # Handle case where model isn't downloaded
        logger.warning(f"Cannot automatically determine visual dimension from {CLIP_FOR_DIM_CHECK} config (may not be downloaded). Defaulting to 768. Ensure this matches your precomputed features.")
        VISUAL_FEATURE_DIM = 768 # Default for large models
    except Exception as e:
        logger.warning(f"Error loading CLIP config: {e}. Defaulting VISUAL_FEATURE_DIM to 768.")
        VISUAL_FEATURE_DIM = 768
except ImportError:
    logger.warning("Transformers not fully available? Cannot load CLIPVisionConfig. Defaulting VISUAL_FEATURE_DIM to 768.")
    VISUAL_FEATURE_DIM = 768


# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
logger.info(f"Expected Visual Feature Dimension: {VISUAL_FEATURE_DIM}")

# --- Seed Function ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed set to {seed}")

# --- CUDA Memory Check Helper ---
def check_cuda_memory(step_name=""):
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try loading as single JSON object first
            try:
                 content = json.load(f)
                 if isinstance(content, list): data = content
                 else: data = [content] # Wrap single object in a list
                 logger.info(f"Loaded {len(data)} samples as single JSON object/array.")
            except json.JSONDecodeError:
                 # If fails, assume JSON Lines format
                 logger.info(f"Initial JSON load failed, attempting to load as JSON Lines (.jsonl) from {file_path}")
                 f.seek(0) # Reset file pointer
                 for line in f:
                     try:
                         if line.strip(): data.append(json.loads(line.strip()))
                     except json.JSONDecodeError:
                         logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
                 if not data: logger.error("File appears empty or fully invalid after jsonl attempt."); return []
                 logger.info(f"Successfully loaded {len(data)} lines as JSON Lines.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}"); return []
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}", exc_info=True); return []

    is_anxiety = "Anxiety_Data" in file_path
    filtered_data = []
    processed_ids = set()

    # Define expected keys - MUST match your JSON structure
    required_keys_base = ['sample_id', 'image_id', 'ocr_text', 'content']
    label_key_anxiety = 'meme_anxiety_category'
    label_key_depression = 'meme_depressive_categories'

    for idx, sample in enumerate(data):
        if not isinstance(sample, dict): logger.warning(f"Skipping non-dict item at index {idx}"); continue

        sample_id = sample.get('sample_id', sample.get('id')) # Allow 'id' as fallback
        if not sample_id: logger.warning(f"Skipping sample {idx} due to missing 'sample_id'/'id'."); continue
        if sample_id in processed_ids: logger.warning(f"Skipping duplicate sample ID: {sample_id}"); continue
        processed_ids.add(sample_id)

        required_keys = required_keys_base + ([label_key_anxiety] if is_anxiety else [label_key_depression])
        if not all(k in sample for k in required_keys):
             missing = [k for k in required_keys if k not in sample]
             logger.warning(f"Skipping sample {sample_id} due to missing keys: {missing}")
             continue

        item = {
            'id': sample_id,
            'image_id': sample.get('image_id', sample_id), # Use sample_id if image_id missing
            'ocr_text': sample.get('ocr_text', '') or "",
            'qwen_reasoning': sample.get('content', '') or ""
        }

        if is_anxiety:
            label = sample.get(label_key_anxiety)
            if label == 'Irritatbily': label = 'Irritability'
            elif label == 'Unknown': label = 'Unknown Anxiety'
            item['original_labels'] = label
            item['stratify_label'] = label
        else: # Depression
            labels = sample.get(label_key_depression)
            if isinstance(labels, str): labels = [l.strip() for l in labels.split(',') if l.strip()]
            if not isinstance(labels, list) or not labels: labels = ["Unknown Depression"]
            processed_labels = [str(lbl).strip() for lbl in labels if str(lbl).strip()]
            processed_labels = [lbl if lbl != 'Unknown' else 'Unknown Depression' for lbl in processed_labels]
            if not processed_labels: processed_labels = ["Unknown Depression"]
            item['original_labels'] = processed_labels
            item['stratify_label'] = processed_labels[0]

        # Basic check for empty text fields that might cause issues
        if not item['ocr_text'] and not item['qwen_reasoning']:
             logger.warning(f"Sample {sample_id} has empty OCR and Qwen reasoning. This might impact RAG.")

        filtered_data.append(item)

    logger.info(f"Loaded and processed {len(filtered_data)} samples from {file_path}.")
    return filtered_data

# --- Feature Loading Function (CLIP Features) ---
def load_clip_features(feature_dir: str, dataset_name: str, split: str) -> Dict[str, torch.Tensor]:
    """Loads pre-computed CLIP features (.pt file)."""
    filename = f"{dataset_name}_{split}_features.pt" # Matches notebook output naming
    filepath = os.path.join(feature_dir, filename)
    logger.info(f"Attempting to load CLIP features from: {filepath}")
    if os.path.exists(filepath):
        try:
            features_map = torch.load(filepath, map_location='cpu')
            # Validate structure and dimension
            if features_map:
                first_key = next(iter(features_map))
                first_val = features_map[first_key]
                if isinstance(first_val, torch.Tensor) and len(first_val.shape) == 1:
                     loaded_dim = first_val.shape[0]
                     if loaded_dim != VISUAL_FEATURE_DIM:
                          logger.warning(f"Loaded feature dimension ({loaded_dim}) != Expected ({VISUAL_FEATURE_DIM}) for key {first_key} in {filename}. Check consistency!")
                     else:
                           logger.info(f"Successfully loaded {len(features_map)} entries from {filename}. Dim ({loaded_dim}) matches expected.")
                else:
                     logger.warning(f"Invalid feature structure/type in {filename}. Expected 1D Tensor, got {type(first_val)} with shape {getattr(first_val, 'shape', 'N/A')}.")
                return features_map
            else:
                logger.warning(f"Feature file loaded but is empty: {filename}")
                return {}
        except Exception as e:
            logger.error(f"Error loading CLIP features from {filepath}: {e}", exc_info=True)
            return {}
    else:
        logger.error(f"CLIP feature file not found: {filepath}")
        return {}

# --- RAG Components (Fused OCR + Qwen Reasoning) ---
class EmbeddingGeneratorRAG:
    """Generates fused embeddings for OCR + Qwen Reasoning for RAG DB & Queries."""
    def __init__(self, model_name: str = RAG_EMBEDDING_MODEL, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing SentenceTransformer for RAG: {model_name} on {self.device}")
        try:
            # Increase max_seq_length if needed for combined texts
            self.model = SentenceTransformer(model_name, device=self.device)
            # You might need to adjust max_seq_length if combined input is long
            # Example: self.model.max_seq_length = 768
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"RAG ST model loaded. Embedding dimension: {self.embedding_dim}. Max Seq Len: {self.model.max_seq_length}")
        except Exception as e: logger.error(f"Failed ST load: {e}", exc_info=True); raise

    def generate_fused_embeddings(self, ocr_texts: List[str], qwen_texts: List[str], batch_size: int = RAG_EMBEDDING_BATCH_SIZE) -> Optional[np.ndarray]:
        """Embeds combined 'OCR: o \\n Reasoning: q' strings."""
        if len(ocr_texts) != len(qwen_texts): logger.error("Length mismatch OCR/Qwen"); return None
        logger.info(f"Generating fused RAG embeddings for {len(ocr_texts)} items...")
        # Use a clear separator. Consider specific tokens if model handles them.
        combined_texts = [f"OCR: {o if o else 'N/A'}\nReasoning: {q if q else 'N/A'}" for o, q in zip(ocr_texts, qwen_texts)]
        try:
            embeddings = self.model.encode(
                combined_texts, batch_size=batch_size, show_progress_bar=len(combined_texts)>1000,
                convert_to_numpy=True, device=self.device
            )
            logger.info(f"Generated RAG embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e: logger.error(f"RAG embedding error: {e}", exc_info=True); return None

class RAGRetrieverFAISS:
    """Handles FAISS index for similarity search."""
    # Keep this class definition exactly as in the previous correct answer
    # It builds IndexFlatL2 and retrieves indices based on query embeddings
    def __init__(self, embeddings: Optional[np.ndarray], top_k: int = RETRIEVAL_K):
        self.top_k = top_k; self.index = None; self.dimension = 0
        if embeddings is not None and embeddings.size > 0:
            if embeddings.dtype != np.float32: embeddings = embeddings.astype(np.float32)
            self.build_index(embeddings)
        else: logger.warning("RAGRetrieverFAISS init with no embeddings.")

    def build_index(self, embeddings: np.ndarray):
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] == 0: logger.error(f"Invalid embeddings for FAISS: {embeddings.shape}"); return
        self.dimension = embeddings.shape[1]; n_samples = embeddings.shape[0]
        logger.info(f"Building FAISS IndexFlatL2 (Dim: {self.dimension}, N: {n_samples})")
        try: self.index = faiss.IndexFlatL2(self.dimension); self.index.add(embeddings); logger.info(f"FAISS index built. Size: {self.index.ntotal}")
        except Exception as e: logger.error(f"FAISS index build error: {e}", exc_info=True); self.index = None

    def retrieve_similar(self, query_embeddings: np.ndarray) -> Optional[np.ndarray]:
        if self.index is None: logger.error("FAISS index not built."); return None
        if query_embeddings is None or query_embeddings.size == 0: logger.warning("Empty query embeddings."); return None
        if query_embeddings.dtype != np.float32: query_embeddings = query_embeddings.astype(np.float32)
        if query_embeddings.ndim == 1: query_embeddings = np.expand_dims(query_embeddings, axis=0)
        if query_embeddings.shape[1] != self.dimension: logger.error(f"Query dim {query_embeddings.shape[1]} != index dim {self.dimension}."); return None
        k = min(self.top_k + 1, self.index.ntotal) # +1 for self
        if k <= 0: logger.warning("Index empty or k=0."); return np.array([[] for _ in range(query_embeddings.shape[0])], dtype=int)
        logger.debug(f"Searching FAISS k={k} for {query_embeddings.shape[0]} queries.")
        try: _, indices = self.index.search(query_embeddings, k=k); return indices
        except Exception as e: logger.error(f"FAISS search error: {e}", exc_info=True); return None


class PromptConstructorRAG:
    """Formats RAG context string from retrieved examples."""
    def __init__(self, train_data: List[Dict]):
        self.train_data = {item['id']: item for item in train_data} # Use dict for faster lookup
        self.is_multilabel = isinstance(next(iter(self.train_data.values()))['original_labels'], list) if self.train_data else False
        logger.info(f"PromptConstructorRAG initialized with {len(self.train_data)} train examples.")

    def format_rag_examples(self, retrieved_indices: Optional[List[int]], current_sample_id: str) -> str:
        """Formats retrieved train examples into a text block, excluding self."""
        if not retrieved_indices: return " [RAG: No relevant examples found] "

        context_str = "\n\n[Similar Examples Start]\n"
        num_added = 0
        train_ids = list(self.train_data.keys()) # Get list of IDs corresponding to indices

        for idx in retrieved_indices:
            if not (0 <= idx < len(train_ids)):
                logger.warning(f"Retrieved index {idx} out of bounds for train data (size {len(train_ids)}).")
                continue

            example_id = train_ids[idx]
            # CRITICAL: Exclude the current sample itself from its own RAG context
            if example_id == current_sample_id:
                logger.debug(f"Skipping self-retrieval for {current_sample_id}")
                continue

            try:
                ex = self.train_data[example_id]
                ocr = ex.get('ocr_text', 'N/A')
                reasoning = ex.get('qwen_reasoning', 'N/A')
                labels = ex.get('original_labels', 'N/A')
                label_str = ", ".join(labels) if isinstance(labels, list) else str(labels)

                context_str += f"\n--- Example {num_added+1} (ID: {example_id}) ---\n"
                context_str += f" OCR: {ocr}\n Reasoning: {reasoning}\n Label: {label_str}\n"
                num_added += 1
                if num_added >= RETRIEVAL_K: break # Limit added examples

            except KeyError: logger.warning(f"Train data missing for retrieved ID: {example_id}"); continue
            except Exception as e: logger.error(f"Error formatting RAG example ID {example_id}: {e}"); continue

        if num_added == 0: context_str += " [RAG: No valid non-self examples found] "
        context_str += "\n[Similar Examples End]"
        return context_str

# --- Dataset Class (Adapted for RAG + Qwen + CLIP) ---
class MentalHealthMemeDatasetRAG(Dataset):
    def __init__(self,
                 samples: List[Dict],
                 bart_tokenizer: BartTokenizer,
                 label_encoder: LabelEncoder,
                 clip_features_map: Dict[str, torch.Tensor],
                 is_multilabel: bool,
                 max_len_bart: int,
                 rag_context_map: Dict[str, str], # Map sample_id -> RAG context string
                 visual_feature_dim: int):

        self.samples = samples
        self.bart_tokenizer = bart_tokenizer
        self.label_encoder = label_encoder
        self.clip_features_map = clip_features_map
        self.is_multilabel = is_multilabel
        self.num_labels = len(label_encoder.classes_)
        self.max_len_bart = max_len_bart
        self.rag_context_map = rag_context_map # Use pre-computed RAG contexts
        self.visual_feature_dim = visual_feature_dim
        self.default_clip_feature = torch.zeros(visual_feature_dim, dtype=torch.float)

        logger.info(f"Dataset created. Samples: {len(samples)}, Multilabel: {is_multilabel}")
        self._validate_features()

    def _validate_features(self):
        missing_count = 0; wrong_dim_count = 0
        for i in range(min(10, len(self.samples))):
            sample_id = self.samples[i]['id']
            image_id = self.samples[i]['image_id']
            feature = self.clip_features_map.get(image_id, self.clip_features_map.get(sample_id))
            if feature is None: missing_count += 1
            elif feature.shape[0] != self.visual_feature_dim: wrong_dim_count += 1
        if missing_count > 0: logger.warning(f"First 10 samples check: {missing_count} missing CLIP features.")
        if wrong_dim_count > 0: logger.warning(f"First 10 samples check: {wrong_dim_count} have incorrect feature dimension.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id']
        image_id = sample['image_id']
        qwen_reasoning = sample['qwen_reasoning']
        original_labels = sample['original_labels']

        # --- Prepare Labels ---
        if self.is_multilabel:
            label_tensor = torch.zeros(self.num_labels, dtype=torch.float)
            try:
                if original_labels: label_indices = self.label_encoder.transform(original_labels)
                else: label_indices = []
                for label_idx in label_indices: label_tensor[label_idx] = 1.0
            except ValueError as e: logger.warning(f"Label error {sample_id}: {e}"); label_tensor = torch.zeros(self.num_labels, dtype=torch.float) # Zero vector on error
        else: # Single label
            try: label_idx = self.label_encoder.transform([original_labels])[0]; label_tensor = torch.tensor(label_idx, dtype=torch.long)
            except ValueError as e: logger.warning(f"Label error {sample_id}: {e}"); label_tensor = torch.tensor(0, dtype=torch.long) # Default to class 0 on error

        # --- Prepare Visual Features (CLIP) ---
        clip_features = self.clip_features_map.get(image_id, self.clip_features_map.get(sample_id))
        if clip_features is None or clip_features.shape[0] != self.visual_feature_dim:
            if clip_features is None: logger.debug(f"CLIP features missing for {image_id}/{sample_id}. Using zeros.")
            else: logger.warning(f"CLIP feature dim mismatch for {image_id}/{sample_id}. Got {clip_features.shape[0]}, expected {self.visual_feature_dim}. Using zeros.")
            clip_features = self.default_clip_feature
        # Ensure float type
        clip_features = clip_features.float()

        # --- Prepare MentalBART Input (Qwen Reasoning + RAG Context) ---
        rag_context = self.rag_context_map.get(sample_id, " [RAG: Context unavailable] ") # Get precomputed RAG string
        # Combine reasoning and RAG context. Ensure reasoning comes first.
        bart_input_text = f"Reasoning: {qwen_reasoning if qwen_reasoning else 'N/A'}{rag_context}"

        bart_encoding = self.bart_tokenizer(
            bart_input_text,
            padding="max_length", max_length=self.max_len_bart, truncation=True,
            return_attention_mask=True, add_special_tokens=True, return_tensors="pt"
        )
        bart_input_ids = bart_encoding["input_ids"].squeeze(0)
        bart_attention_mask = bart_encoding["attention_mask"].squeeze(0)

        return {
            # MentalBART inputs
            "input_ids": bart_input_ids,
            "attention_mask": bart_attention_mask,
            # Visual Input (CLIP)
            "image_features": clip_features,
            # Label
            "labels": label_tensor,
            # ID for potential debugging
            "sample_id": sample_id
        }


# --- Cross-Attention Fusion Model (MentalBART + CLIP) ---
class CrossAttentionModule(nn.Module):
    """Simplified module for bidirectional cross-attention."""
    # Keep this class definition exactly as in the notebook
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.txt_q_vis_kv_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.vis_q_txt_kv_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_features_seq, visual_feature_pooled):
        # Visual feature is already pooled (B, Dim), needs seq dim: (B, 1, Dim)
        visual_feature_seq = visual_feature_pooled.unsqueeze(1)
        # Text attends to Vision (Query=Text, Key=Value=Vision)
        txt_attn_output, _ = self.txt_q_vis_kv_attn(query=text_features_seq, key=visual_feature_seq, value=visual_feature_seq)
        attended_text_seq = self.norm1(text_features_seq + txt_attn_output) # Residual + Norm
        # Vision attends to Text (Query=Vision, Key=Value=Text)
        vis_attn_output, _ = self.vis_q_txt_kv_attn(query=visual_feature_seq, key=attended_text_seq, value=attended_text_seq)
        attended_visual_pooled = self.norm2(visual_feature_seq + vis_attn_output).squeeze(1) # Residual + Norm, remove seq dim -> (B, Dim)
        # Return the attended text sequence and the updated pooled visual feature
        return attended_text_seq, attended_visual_pooled

class M3H_CrossAttentionFusionModel(nn.Module):
    """Classifier using MentalBART encoder, CLIP features, and cross-attention."""
    def __init__(self,
                 num_labels: int,
                 bart_model_name: str = MENTALBART_MODEL_NAME,
                 is_multilabel: bool = False,
                 visual_feature_dim: int = VISUAL_FEATURE_DIM,
                 num_attention_heads: int = NUM_ATTENTION_HEADS,
                 dropout_prob: float = DROPOUT_PROB):
        super().__init__()
        self.num_labels = num_labels
        self.is_multilabel = is_multilabel
        logger.info(f"--- Initializing M3H_CrossAttentionFusionModel ---")
        logger.info(f"  MentalBART base: {bart_model_name}")
        logger.info(f"  Visual Dim (CLIP): {visual_feature_dim}")
        # ... other logs ...

        try:
            logger.info("  Loading MentalBART base model...")
            self.bart = BartModel.from_pretrained(bart_model_name)
            self.text_feature_dim = self.bart.config.hidden_size
            logger.info(f"  MentalBART Loaded. Hidden Dim: {self.text_feature_dim}")

            if visual_feature_dim != self.text_feature_dim:
                logger.info(f"  Adding visual projection: {visual_feature_dim} -> {self.text_feature_dim}")
                self.visual_projection = nn.Linear(visual_feature_dim, self.text_feature_dim)
            else: self.visual_projection = nn.Identity()

            logger.info("  Defining CrossAttentionModule...")
            self.cross_attention = CrossAttentionModule(self.text_feature_dim, num_attention_heads, dropout_prob)

            # Classifier Head - Input comes from pooled attended text features
            self.classifier_input_dim = self.text_feature_dim
            logger.info(f"  Defining Classifier Head (Input Dim: {self.classifier_input_dim})")
            self.dropout = nn.Dropout(dropout_prob)
            self.classifier = nn.Linear(self.classifier_input_dim, num_labels)

            self.loss_fct = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
            logger.info(f"  Using loss: {'BCEWithLogitsLoss' if is_multilabel else 'CrossEntropyLoss'}")
            logger.info("--- Model Initialized Successfully ---")

        except Exception as e: logger.error(f"Error initializing Model: {e}", exc_info=True); raise

    def forward(self, input_ids, attention_mask, image_features, labels=None):
        # 1. MentalBART Encoder -> Text Features Sequence
        # Input text is (Reasoning + RAG Context)
        try:
            encoder_outputs = self.bart.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # Shape: (Batch, SeqLen, BartHiddenDim)
        except Exception as e:
            logger.error(f"Error in BART encoder forward: {e}", exc_info=True)
            # Provide dummy output to avoid crashing subsequent layers immediately
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            encoder_outputs = torch.zeros((batch_size, MAX_LEN_BART, self.text_feature_dim), device=device)


        # 2. Project Visual (CLIP) Features
        try:
             projected_visual_features = self.visual_projection(image_features)
             # Shape: (Batch, BartHiddenDim) - This is a pooled feature
        except Exception as e:
            logger.error(f"Error projecting visual features: {e}", exc_info=True)
            batch_size = image_features.shape[0] if image_features is not None else 1
            projected_visual_features = torch.zeros((batch_size, self.text_feature_dim), device=device)


        # 3. Cross-Attention
        # Text sequence attends to pooled visual; Pooled visual attends to text sequence
        try:
             attended_text_seq, _ = self.cross_attention(
                 text_features_seq=encoder_outputs,            # (B, SeqLen, Dim)
                 visual_feature_pooled=projected_visual_features # (B, Dim)
             )
             # Shape attended_text_seq: (B, SeqLen, Dim)
             # We primarily use the attended text sequence's CLS token
        except Exception as e:
             logger.error(f"Error in CrossAttention: {e}", exc_info=True)
             batch_size = encoder_outputs.shape[0]
             attended_text_seq = torch.zeros((batch_size, MAX_LEN_BART, self.text_feature_dim), device=device)


        # 4. Pool & Dropout (Use CLS token from the *attended* text sequence)
        pooled_output = attended_text_seq[:, 0, :] # Shape: (B, BartHiddenDim)
        final_representation = self.dropout(pooled_output)

        # 5. Classify
        logits = self.classifier(final_representation) # Shape: (B, NumLabels)

        # 6. Loss Calculation
        loss = None
        if labels is not None:
            labels = labels.to(logits.device) # Ensure labels are on same device
            try:
                if self.is_multilabel: loss = self.loss_fct(logits, labels.float())
                else: loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
            except Exception as loss_e: logger.error(f"Loss calc error: {loss_e}", exc_info=True); loss = None

        class ModelOutput: pass
        output = ModelOutput(); output.loss = loss; output.logits = logits
        return output


# --- Training Function (Adapted for new Dataset/Model) ---
def train_and_evaluate(
    train_dataloader: DataLoader, val_dataloader: DataLoader,
    model: M3H_CrossAttentionFusionModel, # Use correct model type
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device, num_epochs: int, output_dir: str,
    label_encoder: LabelEncoder, is_multilabel: bool
) -> Tuple[Dict[str, List], str]:
    """Trains and evaluates a single M3H_CrossAttentionFusionModel instance."""
    os.makedirs(output_dir, exist_ok=True); logger.info(f"Train run output: {output_dir}")
    best_val_f1_macro = 0.0; best_model_path = os.path.join(output_dir, "best_model_macro_f1.pt")
    last_model_path = os.path.join(output_dir, "last_model.pt"); log_file = os.path.join(output_dir, "training_log.csv")
    training_logs = []; history = defaultdict(list); total_steps = len(train_dataloader) * num_epochs

    try:
        for epoch in range(num_epochs):
            epoch_num = epoch + 1; logger.info(f"--- Epoch {epoch_num}/{num_epochs} ---")
            model.train(); total_train_loss = 0.0; all_train_logits, all_train_labels_raw = [], []
            train_progress = tqdm(train_dataloader, desc=f"Train E{epoch_num}", leave=False, ncols=100)

            for batch_idx, batch in enumerate(train_progress):
                optimizer.zero_grad()
                try:
                    # Unpack batch items correctly from MentalHealthMemeDatasetRAG
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    image_features = batch["image_features"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device) # Move labels here too

                    outputs = model(input_ids, attention_mask, image_features, labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    if loss is None: logger.error(f"E{epoch_num} B{batch_idx}: Loss is None."); continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    total_train_loss += loss.item()

                    all_train_logits.append(logits.detach().cpu())
                    all_train_labels_raw.append(batch["labels"].cpu()) # Store original CPU labels

                    train_progress.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

                except Exception as e: logger.error(f"Train B{batch_idx} E{epoch_num} Error: {e}", exc_info=True); continue # Skip batch on error

            avg_train_loss = total_train_loss / len(train_dataloader) if train_dataloader else 0.0
            # --- Calculate training metrics --- (keep logic as before)
            train_f1_micro, train_f1_macro, train_accuracy, train_hamming = 0.0, 0.0, 0.0, 0.0
            if all_train_logits and all_train_labels_raw:
                # ... [metric calculation logic - same as previous correct version] ...
                 all_train_logits_cat = torch.cat(all_train_logits, dim=0); all_train_labels_raw_cat = torch.cat(all_train_labels_raw, dim=0)
                 if is_multilabel:
                     train_probs = torch.sigmoid(all_train_logits_cat).numpy(); train_preds = (train_probs > 0.5).astype(int); train_labels = all_train_labels_raw_cat.numpy().astype(int)
                     train_f1_micro = f1_score(train_labels, train_preds, average="micro", zero_division=0); train_f1_macro = f1_score(train_labels, train_preds, average="macro", zero_division=0)
                     train_accuracy = accuracy_score(train_labels, train_preds); train_hamming = hamming_loss(train_labels, train_preds)
                 else:
                     train_preds = torch.argmax(all_train_logits_cat, dim=1).numpy(); train_labels = all_train_labels_raw_cat.numpy()
                     train_f1_micro = f1_score(train_labels, train_preds, average="micro", zero_division=0); train_f1_macro = f1_score(train_labels, train_preds, average="macro", zero_division=0)
                     train_accuracy = accuracy_score(train_labels, train_preds); train_hamming = hamming_loss(train_labels, train_preds) # = 1 - accuracy
            logger.info(f"E{epoch_num} Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}, MacroF1: {train_f1_macro:.4f}")


            # --- Validation Phase ---
            model.eval(); total_val_loss = 0.0; all_val_logits, all_val_labels_raw = [], []
            val_progress = tqdm(val_dataloader, desc=f"Val E{epoch_num}", leave=False, ncols=100)
            with torch.no_grad():
                for batch in val_progress:
                    try:
                        # Unpack and move tensors
                        input_ids = batch["input_ids"].to(device, non_blocking=True); attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        image_features = batch["image_features"].to(device, non_blocking=True); labels = batch["labels"].to(device)

                        outputs = model(input_ids, attention_mask, image_features, labels)
                        if outputs.loss is not None: total_val_loss += outputs.loss.item()
                        all_val_logits.append(outputs.logits.cpu())
                        all_val_labels_raw.append(batch["labels"].cpu())
                    except Exception as e: logger.error(f"Val Batch E{epoch_num} Error: {e}", exc_info=True); continue

            avg_val_loss = total_val_loss / len(val_dataloader) if val_dataloader else 0.0
            # --- Calculate validation metrics --- (keep logic as before)
            val_f1_micro, val_f1_macro, val_accuracy, val_hamming = 0.0, 0.0, 0.0, 0.0
            if all_val_logits and all_val_labels_raw:
                 # ... [metric calculation logic - same as previous correct version] ...
                 all_val_logits_cat = torch.cat(all_val_logits, dim=0); all_val_labels_raw_cat = torch.cat(all_val_labels_raw, dim=0)
                 if is_multilabel:
                     val_probs = torch.sigmoid(all_val_logits_cat).numpy(); val_preds = (val_probs > 0.5).astype(int); val_labels = all_val_labels_raw_cat.numpy().astype(int)
                     val_f1_micro = f1_score(val_labels, val_preds, average="micro", zero_division=0); val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)
                     val_accuracy = accuracy_score(val_labels, val_preds); val_hamming = hamming_loss(val_labels, val_preds)
                 else:
                     val_preds = torch.argmax(all_val_logits_cat, dim=1).numpy(); val_labels = all_val_labels_raw_cat.numpy()
                     val_f1_micro = f1_score(val_labels, val_preds, average="micro", zero_division=0); val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)
                     val_accuracy = accuracy_score(val_labels, val_preds); val_hamming = hamming_loss(val_labels, val_preds)
            logger.info(f"E{epoch_num} Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}, MacroF1: {val_f1_macro:.4f}")

            # --- Logging and Saving --- (Keep logic as before)
            history['epoch'].append(epoch_num); history['train_loss'].append(avg_train_loss); history['val_loss'].append(avg_val_loss)
            history['train_f1_macro'].append(train_f1_macro); history['train_f1_micro'].append(train_f1_micro)
            history['val_f1_macro'].append(val_f1_macro); history['val_f1_micro'].append(val_f1_micro); 
            history['val_accuracy'].append(val_accuracy); history['train_accuracy'].append(train_accuracy)
            
            current_log = { 
                'epoch': epoch_num, 
                'train_loss': avg_train_loss, 
                'val_loss': avg_val_loss, 
                'train_f1_macro': train_f1_macro,
                'train_f1_micro': train_f1_micro,
                'train_accuracy': train_accuracy,
                'val_f1_macro': val_f1_macro, 
                'val_f1_micro': val_f1_micro,
                'val_accuracy': val_accuracy 
            }
            training_logs.append(current_log)
            if val_f1_macro > best_val_f1_macro: best_val_f1_macro = val_f1_macro; logger.info(f"E{epoch_num}: New best val Macro F1: {best_val_f1_macro:.4f}. Saving..."); torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), last_model_path); pd.DataFrame(training_logs).to_csv(log_file, index=False)
            del all_train_logits, all_train_labels_raw, all_val_logits, all_val_labels_raw
            if 'all_train_logits_cat' in locals(): del all_train_logits_cat, all_train_labels_raw_cat
            if 'all_val_logits_cat' in locals(): del all_val_logits_cat, all_val_labels_raw_cat
            gc.collect(); torch.cuda.empty_cache()

        logger.info(f"Training finished. Best Val Macro F1: {best_val_f1_macro:.4f}")

    except KeyboardInterrupt: logger.warning("Training interrupted."); torch.save(model.state_dict(), last_model_path); pd.DataFrame(training_logs).to_csv(log_file, index=False)
    except Exception as e: logger.error(f"Training loop error: {e}", exc_info=True); pd.DataFrame(training_logs).to_csv(log_file, index=False); raise

    return history, best_model_path

# --- Evaluation Function (Ensemble - Adapted for new Model) ---
def evaluate_ensemble(
    model_paths: List[str], dataloader: DataLoader, device: torch.device,
    label_encoder: LabelEncoder, is_multilabel: bool, num_labels: int, output_dir: str,
    report_suffix: str = "eval_ensemble", bart_model_name: str = MENTALBART_MODEL_NAME
) -> Dict:
    """Evaluates an ensemble of M3H_CrossAttentionFusionModel models."""
    if not dataloader: logger.warning(f"Dataloader '{report_suffix}' empty."); return {'error': 'No data'}
    if not model_paths: logger.error("No model paths for ensemble eval."); return {'error': 'No models'}
    logger.info(f"--- Starting ENSEMBLE Evaluation ({report_suffix}) --- Models: {len(model_paths)}")

    all_logits = []; all_labels_raw = None
    for model_idx, model_path in enumerate(model_paths):
        logger.info(f"Eval model {model_idx + 1}/{len(model_paths)}: {os.path.basename(model_path)}")
        model = None
        try:
            # Instantiate the CORRECT model class
            model = M3H_CrossAttentionFusionModel(
                 num_labels, bart_model_name, is_multilabel, VISUAL_FEATURE_DIM,
                 NUM_ATTENTION_HEADS, DROPOUT_PROB
            )
            try: model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
            except RuntimeError: logger.warning("Strict load failed, trying strict=False"); model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model.to(device); model.eval()
        except Exception as e: logger.error(f"Load model {model_idx+1} failed: {e}", exc_info=True); continue

        current_logits = []; batch_labels = []
        eval_progress = tqdm(dataloader, desc=f"Eval M{model_idx + 1}", leave=False, ncols=100)
        with torch.no_grad():
            for batch in eval_progress:
                try:
                    # Unpack and move necessary inputs
                    input_ids = batch["input_ids"].to(device, non_blocking=True); attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    image_features = batch["image_features"].to(device, non_blocking=True)
                    # Run forward pass WITHOUT labels
                    outputs = model(input_ids, attention_mask, image_features, labels=None)
                    current_logits.append(outputs.logits.cpu())
                    if model_idx == 0: batch_labels.append(batch["labels"].cpu())
                except Exception as e: logger.error(f"Eval batch M{model_idx+1} error: {e}", exc_info=True)

        if current_logits: all_logits.append(torch.cat(current_logits, dim=0))
        if model_idx == 0 and batch_labels: all_labels_raw = torch.cat(batch_labels, dim=0)
        elif model_idx == 0: logger.error("Label collection failed."); return {'error': 'Label collection failed'}
        del model, current_logits, batch_labels; gc.collect(); torch.cuda.empty_cache()

    if not all_logits or all_labels_raw is None: logger.error("Logit/Label collection failed."); return {'error': 'Logit/Label collection failed'}

    # --- Aggregate predictions & calculate metrics --- (Keep logic as before)
    stacked_logits = torch.stack(all_logits, dim=0); avg_logits = torch.mean(stacked_logits, dim=0)
    metrics_dict = {}; classification_rep = "Report failed."
    # ... [Metric calculation and report generation logic - same as previous correct version] ...
    try:
        if is_multilabel:
            probs = torch.sigmoid(avg_logits).numpy(); preds = (probs > 0.5).astype(int); labels = all_labels_raw.numpy().astype(int)
            metrics_dict = { 'accuracy_subset': accuracy_score(labels, preds), 'hamming_loss': hamming_loss(labels, preds),
                             'micro_f1': f1_score(labels, preds, average='micro', zero_division=0), 'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
                             'weighted_f1': f1_score(labels, preds, average='weighted', zero_division=0), 'samples_f1': f1_score(labels, preds, average='samples', zero_division=0) }
            classification_rep = classification_report(labels, preds, target_names=label_encoder.classes_, digits=4, zero_division=0)
        else: # Single-label
            preds = torch.argmax(avg_logits, dim=1).numpy(); labels = all_labels_raw.numpy()
            metrics_dict = { 'accuracy': accuracy_score(labels, preds), 'hamming_loss': hamming_loss(labels, preds),
                             'micro_f1': f1_score(labels, preds, average='micro', zero_division=0), 'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
                             'weighted_f1': f1_score(labels, preds, average='weighted', zero_division=0) }
            present_labels = sorted(list(set(labels) | set(preds))); target_names = [label_encoder.classes_[i] for i in present_labels if 0 <= i < len(label_encoder.classes_)]
            valid_present_labels = [i for i in present_labels if 0 <= i < len(label_encoder.classes_)]
            if valid_present_labels: classification_rep = classification_report(labels, preds, labels=valid_present_labels, target_names=target_names, digits=4, zero_division=0)
            else: classification_rep = "No valid labels."
    except Exception as e: logger.error(f"Metric calc error ({report_suffix}): {e}", exc_info=True); metrics_dict = {'error': f"Metrics failed: {e}"}

    metrics_dict['report'] = classification_rep
    report_path = os.path.join(output_dir, f"classification_report_{report_suffix}.txt")
    # ... [Save report to file logic - keep same] ...
    try:
        with open(report_path, "w", encoding='utf-8') as f:
             f.write(f"--- Ensemble Evaluation Metrics ({report_suffix}) ---\n\n")
             for key, value in metrics_dict.items():
                 if key != 'report': f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}\n")
             f.write("\n--- Classification Report ---\n\n"); f.write(classification_rep)
        logger.info(f"Eval report saved: {report_path}")
    except Exception as e: logger.error(f"Report save fail: {e}")

    logger.info(f"--- Ensemble Evaluation ({report_suffix}) Complete ---")
    return metrics_dict

# --- Main Pipeline Function (Adapted for RAG + Qwen + CLIP + MentalBART) ---
def run_mental_rag_ensemble_pipeline(
    dataset_type: str = 'anxiety',
    num_ensemble_runs: int = NUM_ENSEMBLE_MODELS,
    seed: int = BASE_SEED,
):
    """Orchestrates the full pipeline."""
    pipeline_start_time = time.time()
    is_multilabel = (dataset_type == 'depression')
    pipeline_output_dir = os.path.join(OUTPUT_DIR_BASE, dataset_type)
    os.makedirs(pipeline_output_dir, exist_ok=True)
    logger.info(f"===== STARTING Mental-RAG Pipeline for: {dataset_type} =====")
    logger.info(f"Output directory: {pipeline_output_dir}")
    check_cuda_memory("Pipeline Start")

    # --- 1. Load Data (Qwen JSONs) & Split ---
    logger.info("--- Step 1: Loading Data & Splitting ---")
    if dataset_type == 'anxiety':
        # Adjust filenames based on your actual Qwen output files
        train_file = os.path.join(ANXIETY_DATA_DIR, "anxiety_train_fig_extractor_results.jsonl")
        test_file = os.path.join(ANXIETY_DATA_DIR, "anxiety_test_fig_extractor_results.jsonl")
        val_file = None
    else: # depression
        train_file = os.path.join(DEPRESSION_DATA_DIR, "train_fig_extractor_results.jsonl") # Assumed names
        val_file = os.path.join(DEPRESSION_DATA_DIR, "val_fig_extractor_results.jsonl")
        test_file = os.path.join(DEPRESSION_DATA_DIR, "test_fig_extractor_results.jsonl")

    train_data_full = load_qwen_data(train_file)
    if not train_data_full: logger.error("Train data failed. Abort."); return
    val_data = load_qwen_data(val_file) if val_file and os.path.exists(val_file) else []
    test_data = load_qwen_data(test_file) if test_file and os.path.exists(test_file) else []
    if not val_data: logger.info("Splitting train->train/val."); train_data, val_data, _ = split_data(train_data_full, 0.15, 0, seed)
    else: train_data = train_data_full
    logger.info(f"Data sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    if not train_data or not val_data: logger.error("Train/Val empty. Abort."); return

    # --- 2. Load CLIP Visual Features ---
    logger.info("--- Step 2: Loading CLIP Features ---")
    # Ensure VISUAL_FEATURES_DIR points to the correct location of your CLIP .pt files
    if not os.path.exists(VISUAL_FEATURES_DIR): logger.error(f"Visual features directory not found: {VISUAL_FEATURES_DIR}. Abort."); return
    features_train = load_clip_features(VISUAL_FEATURES_DIR, dataset_type, "train")
    features_val = load_clip_features(VISUAL_FEATURES_DIR, dataset_type, "val") if val_file else features_train
    features_test = load_clip_features(VISUAL_FEATURES_DIR, dataset_type, "test") if test_file else {}
    clip_features_map = {**features_test, **features_val, **features_train}
    logger.info(f"Total unique CLIP features loaded: {len(clip_features_map)}")
    check_cuda_memory("After Visual Feature Load")

    # --- 3. Label Encoding ---
    logger.info("--- Step 3: Encoding Labels ---")
    all_labels = set(); # ... [Label extraction logic - same as previous] ...
    for d in [train_data, val_data, test_data]:
        for s in d:
             lbls = s['original_labels'];
             if isinstance(lbls, list): all_labels.update(l for l in lbls if l)
             elif isinstance(lbls, str) and lbls: all_labels.add(lbls)
    if not all_labels: logger.error("No labels."); return
    label_encoder = LabelEncoder().fit(sorted(list(all_labels)))
    num_labels = len(label_encoder.classes_)
    logger.info(f"Labels encoded. Num classes: {num_labels}.") # Classes: {label_encoder.classes_}")

    # --- 4. RAG Setup (Embeddings, Index, Context Generation) ---
    logger.info("--- Step 4: Setting up RAG ---")
    rag_embeddings_db = None; rag_retriever = None; rag_context_map = {}; temp_embed_gen = None

    try:
        logger.info("  Generating RAG DB embeddings (Train OCR + Qwen)...")
        temp_embed_gen = EmbeddingGeneratorRAG(model_name=RAG_EMBEDDING_MODEL, device=device)
        train_ocr = [s['ocr_text'] for s in train_data]; train_qwen = [s['qwen_reasoning'] for s in train_data]
        rag_embeddings_db = temp_embed_gen.generate_fused_embeddings(train_ocr, train_qwen)

        if rag_embeddings_db is not None:
            logger.info("  Building RAG FAISS index...")
            rag_retriever = RAGRetrieverFAISS(rag_embeddings_db, top_k=RETRIEVAL_K)
            if rag_retriever.index:
                 logger.info("  Generating RAG context for train/val/test splits...")
                 rag_prompt_constructor = PromptConstructorRAG(train_data) # Needs train data for lookup
                 # Process splits in batches for embedding generation efficiency
                 for split_data, split_name in [(train_data, "Train"), (val_data, "Val"), (test_data, "Test")]:
                     if not split_data: continue
                     logger.info(f"  Processing {split_name} for RAG context ({len(split_data)} samples)...")
                     split_ocr = [s['ocr_text'] for s in split_data]; split_qwen = [s['qwen_reasoning'] for s in split_data]
                     query_embeddings = temp_embed_gen.generate_fused_embeddings(split_ocr, split_qwen)
                     if query_embeddings is None: logger.warning(f"Failed query embed for {split_name}. No RAG context."); continue

                     retrieved_indices_batch = rag_retriever.retrieve_similar(query_embeddings)
                     if retrieved_indices_batch is None: logger.warning(f"Retrieval failed for {split_name}. No RAG context."); continue

                     for i, sample in enumerate(tqdm(split_data, desc=f"  Format {split_name} RAG", ncols=100)):
                         sample_id = sample['id']
                         valid_indices = retrieved_indices_batch[i].tolist() # retrieve_similar already handles k+1
                         # format_rag_examples handles self-exclusion
                         rag_context_map[sample_id] = rag_prompt_constructor.format_rag_examples(valid_indices, sample_id)
                     del query_embeddings, retrieved_indices_batch # Free memory
                 logger.info(f"Generated RAG context for {len(rag_context_map)} samples total.")
            else: logger.warning("RAG index build failed. RAG disabled.")
        else: logger.warning("RAG DB embedding failed. RAG disabled.")
    except Exception as e: logger.error(f"RAG setup error: {e}", exc_info=True)
    finally:
        del rag_embeddings_db, rag_retriever # Keep rag_context_map
        if temp_embed_gen: del temp_embed_gen
        gc.collect(); torch.cuda.empty_cache()
    check_cuda_memory("After RAG Setup")


    # --- 5. Tokenizer, Datasets, DataLoaders ---
    logger.info("--- Step 5: Loading Tokenizer ---")
    try: bart_tokenizer = BartTokenizer.from_pretrained(MENTALBART_MODEL_NAME)
    except Exception as e: logger.error(f"Failed tokenizer load: {e}", exc_info=True); return

    logger.info("--- Step 6: Creating Datasets ---")
    try:
        train_dataset = MentalHealthMemeDatasetRAG(train_data, bart_tokenizer, label_encoder, clip_features_map, is_multilabel, MAX_LEN_BART, rag_context_map, VISUAL_FEATURE_DIM)
        val_dataset = MentalHealthMemeDatasetRAG(val_data, bart_tokenizer, label_encoder, clip_features_map, is_multilabel, MAX_LEN_BART, rag_context_map, VISUAL_FEATURE_DIM)
        test_dataset = MentalHealthMemeDatasetRAG(test_data, bart_tokenizer, label_encoder, clip_features_map, is_multilabel, MAX_LEN_BART, rag_context_map, VISUAL_FEATURE_DIM) if test_data else None
    except Exception as e: logger.error(f"Dataset creation failed: {e}", exc_info=True); return

    logger.info("--- Step 7: Creating DataLoaders ---")
    try:
        nw = 2 if torch.cuda.is_available() else 0; pm = True if device == torch.device('cuda') else False
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, pin_memory=pm, drop_last=(len(train_dataset)%BATCH_SIZE==1))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=nw, pin_memory=pm) # Larger batch for eval
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=nw, pin_memory=pm) if test_dataset else None
        if not train_loader or not val_loader: raise ValueError("Empty DataLoader")
    except Exception as e: logger.error(f"DataLoader creation failed: {e}", exc_info=True); return
    check_cuda_memory("After Dataloaders")


    # --- 6. Ensemble Training ---
    logger.info(f"--- Step 8: Starting Ensemble Training ({num_ensemble_runs} models) ---")
    trained_model_paths = []
    # ... [Ensemble loop: Instantiate M3H_CrossAttentionFusionModel, setup opt/sched, call train_and_evaluate] ...
    for i in range(num_ensemble_runs):
        run_seed = seed + i; set_seed(run_seed)
        model_run_output_dir = os.path.join(pipeline_output_dir, f"run_{run_seed}")
        logger.info(f"--- Training Model {i + 1}/{num_ensemble_runs} (Seed: {run_seed}) ---")
        model=None; optimizer=None; scheduler=None # Scope
        try:
            model = M3H_CrossAttentionFusionModel(num_labels, MENTALBART_MODEL_NAME, is_multilabel, VISUAL_FEATURE_DIM, NUM_ATTENTION_HEADS, DROPOUT_PROB).to(device)
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
            total_steps = len(train_loader) * NUM_EPOCHS; warmup_steps = int(0.1 * total_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

            history, best_path = train_and_evaluate( train_loader, val_loader, model, optimizer, scheduler, device, NUM_EPOCHS, model_run_output_dir, label_encoder, is_multilabel)
            trained_model_paths.append(best_path)
            # plot_training_history(history, model_run_output_dir, f"_run_{run_seed}") # Add plot function if defined/needed

        except Exception as train_e: logger.error(f"Training run {i+1} failed: {train_e}", exc_info=True) # Save traceback if needed
        finally: del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache(); check_cuda_memory(f"End run {i+1}")

    if not trained_model_paths: logger.error("No models trained."); return
    logger.info(f"--- Ensemble Training Finished. Trained {len(trained_model_paths)} models. ---")

    # --- 7. Final Evaluation ---
    logger.info("--- Step 9: Evaluating Ensemble ---")
    # ... [Call evaluate_ensemble for val/test, using MENTALBART_MODEL_NAME] ...
    val_metrics = evaluate_ensemble(trained_model_paths, val_loader, device, label_encoder, is_multilabel, num_labels, pipeline_output_dir, "validation_ensemble", MENTALBART_MODEL_NAME)
    if test_loader: test_metrics = evaluate_ensemble(trained_model_paths, test_loader, device, label_encoder, is_multilabel, num_labels, pipeline_output_dir, "test_ensemble", MENTALBART_MODEL_NAME)

    # --- 8. Save Artifacts ---
    logger.info("--- Step 10: Saving Label Encoder ---")
    le_path = os.path.join(pipeline_output_dir, "label_encoder.pkl")
    try: 
        with open(le_path, 'wb') as f: pickle.dump(label_encoder, f); logger.info(f"LE saved: {le_path}")
    except Exception as e: logger.error(f"LE save fail: {e}")

    pipeline_duration = time.time() - pipeline_start_time
    logger.info(f"===== Pipeline COMPLETED for {dataset_type} in {pipeline_duration/60:.2f} mins =====")
    check_cuda_memory("Pipeline End")


# --- Execution ---
if __name__ == "__main__":
    logger.info("Script execution started.")
    set_seed(BASE_SEED)

    # --- CHOOSE DATASET TYPE ---
    # dataset = 'anxiety'
    dataset = 'depression'
    # ---------------------------
    logger.info(f"Running pipeline for dataset: {dataset}")

    try:
        run_mental_rag_ensemble_pipeline(dataset_type=dataset)
    except Exception as main_e:
        logger.critical(f"Unhandled exception in main execution block: {main_e}", exc_info=True)
        # Save traceback
        tb_path = os.path.join(OUTPUT_DIR_BASE, f"{dataset}_CRASH_traceback.txt")
        os.makedirs(os.path.dirname(tb_path), exist_ok=True)
        with open(tb_path, "w") as f: traceback.print_exc(file=f)
        logger.info(f"Crash traceback saved to: {tb_path}")

    logger.info("Script execution finished.")