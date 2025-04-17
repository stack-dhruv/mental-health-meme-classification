# hybrid_pipeline.py (save inside the 'shared' folder)

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    # For MentalBART
    BartTokenizer,
    BartModel,
    # For LXMERT
    LxmertTokenizer,
    LxmertModel, # Base LXMERT model
    LxmertConfig, # To get hidden size if needed
    # General HF utilities
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
# Sentence Transformer for RAG
from sentence_transformers import SentenceTransformer
# FAISS for RAG Index
import faiss
# General utilities
import matplotlib.pyplot as plt
from tqdm import tqdm # Use standard if running as script
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
logger = logging.getLogger("HybridMemePipeline")

# --- Constants & Hyperparameters ---

# Model Identifiers
LXMERT_MODEL_NAME = "unc-nlp/lxmert-base-uncased" # Standard LXMERT
# MENTALBART_MODEL_NAME = "mental/mental-bart-base-cased" # MentalBART
MENTALBART_MODEL_NAME = 'Tianlin668/MentalBART'
RAG_EMBEDDING_MODEL = "BAAI/bge-m3" # Sentence Transformer for RAG

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
print(project_root)
DATASET_DIR = os.path.join(project_root, "dataset")
MODELS_DIR = os.path.join(project_root, "models")
# Paths (Relative to this script in 'shared' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Should be the 'shared' directory
SRC_DIR = os.path.abspath(os.path.join(script_dir, "..")) # Get the 'src' directory
# DATASET_DIR = os.path.join(SRC_DIR, "../dataset") # Go up one level from src to project root, then dataset
# MODELS_DIR = os.path.join(SRC_DIR, "../models")   # Go up one level from src to project root, then models
# OUTPUT_DIR_BASE will be set dynamically in the pipeline function

ANXIETY_DATA_DIR = os.path.join(DATASET_DIR, "Anxiety_Data")
DEPRESSION_DATA_DIR = os.path.join(DATASET_DIR, "Depressive_Data")
REGION_FEATURES_DIR = os.path.join(MODELS_DIR, "region_visual_features")

# Training Settings
MAX_LEN_LXMERT = 80      # Max sequence length for LXMERT text input (OCR) - typically shorter
MAX_LEN_BART = 512     # Max sequence length for MentalBART input (Reasoning + RAG)
BATCH_SIZE = 16         # START SMALL - This model is memory intensive
NUM_EPOCHS = 10         # Epochs per ensemble member
LEARNING_RATE = 2e-5   # Common for fine-tuning transformers
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
DROPOUT_PROB = 0.1

# RAG Settings
RETRIEVAL_K = 3          # Number of examples to retrieve

# Ensemble Settings
NUM_ENSEMBLE_MODELS = 3
BASE_SEED = 42

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# --- Seed Function ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True # Optional
    # torch.backends.cudnn.benchmark = False   # Optional
    logger.info(f"Seed set to {seed}")

# --- Custom Collate Function ---
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized visual features and positions.
    This handles the "Trying to resize storage that is not resizable" error during batching.
    """
    elem = batch[0]
    batch_dict = {key: [] for key in elem}
    
    # Collect all items by key
    for b in batch:
        for key in elem:
            batch_dict[key].append(b[key])
    
    # Special handling for visual features and positions
    max_boxes = max([feat.size(0) for feat in batch_dict['visual_feats']])
    feat_dim = batch_dict['visual_feats'][0].size(1)
    
    # Create padded batches for visual features
    padded_feats = []
    padded_pos = []
    padded_vis_mask = []
    
    for i in range(len(batch)):
        feat = batch_dict['visual_feats'][i]
        pos = batch_dict['visual_pos'][i]
        num_boxes = feat.size(0)
        
        # Pad features if needed
        if num_boxes < max_boxes:
            padding = torch.zeros(max_boxes - num_boxes, feat_dim, dtype=feat.dtype, device=feat.device)
            padded_feat = torch.cat([feat, padding], dim=0)
            
            # Pad positions (4-dimensional coordinates)
            pos_padding = torch.zeros(max_boxes - num_boxes, 4, dtype=pos.dtype, device=pos.device)
            padded_position = torch.cat([pos, pos_padding], dim=0)
            
            # Create mask with 1s for real boxes, 0s for padding
            mask = torch.cat([
                batch_dict['vis_attention_mask'][i], 
                torch.zeros(max_boxes - num_boxes, dtype=torch.long)
            ])
        else:
            padded_feat = feat
            padded_position = pos
            mask = batch_dict['vis_attention_mask'][i]
        
        padded_feats.append(padded_feat)
        padded_pos.append(padded_position)
        padded_vis_mask.append(mask)
    
    # Stack the padded tensors
    batch_dict['visual_feats'] = torch.stack(padded_feats)
    batch_dict['visual_pos'] = torch.stack(padded_pos)
    batch_dict['vis_attention_mask'] = torch.stack(padded_vis_mask)
    
    # Handle other tensors normally
    result = {}
    for key in batch_dict:
        if key in ['visual_feats', 'visual_pos', 'vis_attention_mask']:
            # Already processed
            result[key] = batch_dict[key]
        else:
            # Use default PyTorch stacking
            result[key] = torch.stack(batch_dict[key]) if torch.is_tensor(batch_dict[key][0]) else batch_dict[key]
    
    return result

# --- Data Loading Function ---
def load_qwen_data(file_path: str) -> List[Dict]:
    """Loads data from JSON containing combined figurative reasoning results."""
    logger.info(f"Loading figurative reasoning data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        # Attempt line-by-line loading for jsonl format
        data = []
        try:
             with open(file_path, 'r', encoding='utf-8') as f:
                 for line in f:
                     try: data.append(json.loads(line.strip()))
                     except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
             if not data: logger.error("File appears empty or fully invalid after jsonl attempt."); return []
             logger.info(f"Successfully loaded {len(data)} lines as JSON Lines.")
        except Exception as e_jsonl:
             logger.error(f"Failed to load as JSON Lines: {e_jsonl}"); return []

    is_anxiety = "Anxiety_Data" in file_path
    filtered_data = []
    processed_ids = set()

    required_keys = ['sample_id', 'ocr_text'] # Base keys - modified for new format
    label_key_anxiety = 'meme_anxiety_category'
    label_key_depression = 'meme_depressive_categories'

    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            logger.warning(f"Skipping non-dict item at index {idx} in {file_path}")
            continue

        sample_id = sample.get('sample_id')
        if not sample_id: sample_id = sample.get('id') # Fallback id
        if not sample_id: logger.warning(f"Skipping sample {idx} due to missing ID."); continue
        if sample_id in processed_ids: logger.warning(f"Skipping duplicate sample ID: {sample_id}"); continue
        processed_ids.add(sample_id)

        # Check for essential keys
        if not all(k in sample for k in required_keys):
             missing = [k for k in required_keys if k not in sample]
             logger.warning(f"Skipping sample {sample_id} due to missing keys: {missing}")
             continue

        # Extract core data - using figurative_reasoning instead of qwen_reasoning
        item = {
            'id': sample_id, # Standardize to 'id'
            'image_id': sample.get('image_id', sample_id), # Fallback to sample_id if image_id missing
            'ocr_text': sample.get('ocr_text', '') or "", # Ensure empty string if None or empty
            'qwen_reasoning': sample.get('figurative_reasoning', '') or "" # Use figurative_reasoning field
        }

        # Process labels
        if is_anxiety:
            label = sample.get(label_key_anxiety)
            if label is None: logger.warning(f"Anxiety sample {sample_id} missing label '{label_key_anxiety}'. Skipping."); continue
            # Standardize
            if label == 'Irritatbily': label = 'Irritability'
            elif label == 'Unknown': label = 'Unknown Anxiety'
            item['original_labels'] = label # Single label
            item['stratify_label'] = label
        else: # Depression
            labels = sample.get(label_key_depression)
            if labels is None: logger.warning(f"Depression sample {sample_id} missing label '{label_key_depression}'. Assigning ['Unknown Depression']."); labels = ["Unknown Depression"]
            # Ensure list format
            if isinstance(labels, str): labels = [l.strip() for l in labels.split(',') if l.strip()]
            if not isinstance(labels, list): logger.warning(f"Unexpected label format for {sample_id}: {type(labels)}. Treating as empty."); labels = []
            if not labels: labels = ["Unknown Depression"] # Default if empty list

            processed_labels = [str(lbl).strip() for lbl in labels if str(lbl).strip()]
            processed_labels = [lbl if lbl != 'Unknown' else 'Unknown Depression' for lbl in processed_labels]
            if not processed_labels: processed_labels = ["Unknown Depression"] # Final fallback

            item['original_labels'] = processed_labels # List of labels
            item['stratify_label'] = processed_labels[0] # Use first for stratification

        filtered_data.append(item)

    logger.info(f"Loaded and filtered {len(filtered_data)} samples from {file_path}.")
    return filtered_data

# --- Feature Loading Function ---
def load_region_features(feature_dir: str, dataset_name: str, split: str) -> Dict[str, Dict[str, Any]]:
    """Loads pre-computed region features (.pt file)."""
    # Construct filename (e.g., anxiety_train_region_features.pt)
    filename = f"{dataset_name}_{split}_region_features.pt"
    filepath = os.path.join(feature_dir, filename)
    logger.info(f"Attempting to load region features from: {filepath}")
    if os.path.exists(filepath):
        try:
            features_map = torch.load(filepath, map_location='cpu')
            logger.info(f"Successfully loaded {len(features_map)} entries from {filename}.")
            # Basic validation: Check if first item looks like expected dict
            if features_map:
                first_key = next(iter(features_map))
                first_val = features_map[first_key]
                if isinstance(first_val, dict) and 'features' in first_val and 'boxes' in first_val:
                     logger.info("Feature map structure looks valid (contains 'features' and 'boxes').")
                else:
                     logger.warning(f"Feature map structure might be invalid. First entry type: {type(first_val)}")
            return features_map
        except Exception as e:
            logger.error(f"Error loading region features from {filepath}: {e}", exc_info=True)
            return {}
    else:
        logger.error(f"Region feature file not found: {filepath}")
        return {}

# --- RAG Components (Adapted) ---
class EmbeddingGeneratorRAG:
    """Generates fused embeddings for OCR + Qwen Reasoning for RAG."""
    def __init__(self, model_name: str = RAG_EMBEDDING_MODEL, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing SentenceTransformer for RAG: {model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"RAG ST model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}", exc_info=True)
            raise

    def generate_fused_embeddings(self, ocr_texts: List[str], qwen_texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """Generates embeddings for OCR & Qwen, then fuses (e.g., averaging)."""
        if len(ocr_texts) != len(qwen_texts):
            logger.error(f"Length mismatch: OCR={len(ocr_texts)}, Qwen={len(qwen_texts)}")
            return None
        logger.info(f"Generating fused RAG embeddings for {len(ocr_texts)} items...")

        # Combine OCR and Qwen text with a separator for context
        combined_texts = [f"OCR: {o}\nReasoning: {q}" for o, q in zip(ocr_texts, qwen_texts)]

        try:
            embeddings = self.model.encode(
                combined_texts, batch_size=batch_size, show_progress_bar=len(combined_texts)>1000,
                convert_to_numpy=True, device=self.device
            )
            logger.info(f"Generated RAG embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error during RAG embedding generation: {e}", exc_info=True)
            return None

class RAGRetrieverFAISS:
    """Handles FAISS index for text similarity retrieval."""
    def __init__(self, embeddings: Optional[np.ndarray], top_k: int = RETRIEVAL_K):
        self.top_k = top_k
        self.index = None
        self.dimension = 0
        if embeddings is not None and embeddings.size > 0:
            if embeddings.dtype != np.float32: embeddings = embeddings.astype(np.float32)
            self.build_index(embeddings)
        else: logger.warning("RAGRetriever initialized with no embeddings.")

    def build_index(self, embeddings: np.ndarray):
        """Builds FAISS IndexFlatL2."""
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] == 0:
             logger.error(f"Invalid embeddings for FAISS index: Shape={embeddings.shape}, Type={type(embeddings)}")
             return
        self.dimension = embeddings.shape[1]
        n_samples = embeddings.shape[0]
        logger.info(f"Building FAISS IndexFlatL2 (Dim: {self.dimension}, N: {n_samples})")
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            logger.info(f"FAISS index built successfully. Size: {self.index.ntotal}")
        except Exception as e: logger.error(f"Error building FAISS index: {e}", exc_info=True); self.index = None

    def retrieve_similar(self, query_embeddings: np.ndarray) -> Optional[np.ndarray]:
        """Retrieves indices of top_k most similar items."""
        if self.index is None: logger.error("FAISS index not built."); return None
        if query_embeddings is None or query_embeddings.size == 0: logger.warning("Empty query embeddings."); return None
        if query_embeddings.dtype != np.float32: query_embeddings = query_embeddings.astype(np.float32)
        if query_embeddings.ndim == 1: query_embeddings = np.expand_dims(query_embeddings, axis=0)
        if query_embeddings.shape[1] != self.dimension: logger.error(f"Query dim {query_embeddings.shape[1]} != index dim {self.dimension}."); return None

        k = min(self.top_k + 1, self.index.ntotal) # +1 for self-retrieval
        if k <= 0: logger.warning("Index empty or k=0."); return np.array([[] for _ in range(query_embeddings.shape[0])], dtype=int)

        logger.debug(f"Searching FAISS for k={k} neighbors for {query_embeddings.shape[0]} queries.")
        try:
            distances, indices = self.index.search(query_embeddings, k=k)
            return indices
        except Exception as e: logger.error(f"FAISS search error: {e}", exc_info=True); return None

class PromptConstructorRAG:
    """Constructs context strings including RAG examples."""
    def __init__(self, train_data: List[Dict], label_encoder: LabelEncoder):
        self.train_data = train_data
        self.label_encoder = label_encoder
        self.is_multilabel = isinstance(train_data[0]['original_labels'], list) if train_data else False
        logger.info(f"PromptConstructorRAG initialized with {len(train_data)} examples. Multilabel: {self.is_multilabel}")

    def format_rag_examples(self, sample_id: str, retrieved_indices: Optional[List[int]]) -> str:
        """Formats retrieved examples into a text block."""
        if not retrieved_indices: return " [No similar examples retrieved] "

        context_str = " [Similar Examples Start] "
        num_added = 0
        for idx in retrieved_indices:
            try:
                ex = self.train_data[idx]
                ex_id = ex['id']
                # Avoid self-retrieval
                if ex_id == sample_id: continue

                ocr = ex.get('ocr_text', 'N/A')
                reasoning = ex.get('qwen_reasoning', 'N/A')
                labels = ex.get('original_labels', 'N/A')

                if isinstance(labels, list): label_str = ", ".join(labels) if labels else "None"
                else: label_str = str(labels) if labels is not None else "N/A"

                context_str += f"\n<Example {num_added+1}>\n"
                context_str += f" OCR: {ocr}\n Reasoning: {reasoning}\n Label: {label_str}\n</Example {num_added+1}>"
                num_added += 1
                if num_added >= RETRIEVAL_K: break # Limit added examples

            except IndexError: logger.warning(f"RAG index {idx} out of bounds for train_data."); continue
            except Exception as e: logger.error(f"Error formatting RAG example index {idx}: {e}"); continue

        if num_added == 0: context_str += " [No valid similar examples found] "
        context_str += " [Similar Examples End] "
        return context_str

# --- Dataset Class ---
class MentalHealthMemeDataset(Dataset):
    def __init__(self,
                 samples: List[Dict],
                 lxmert_tokenizer: LxmertTokenizer,
                 bart_tokenizer: BartTokenizer,
                 label_encoder: LabelEncoder,
                 region_features_map: Dict[str, Dict],
                 is_multilabel: bool,
                 max_len_lxmert: int,
                 max_len_bart: int,
                 rag_context_map: Optional[Dict[str, str]] = None): # Map sample_id -> RAG context string

        self.samples = samples
        self.lxmert_tokenizer = lxmert_tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.label_encoder = label_encoder
        self.region_features_map = region_features_map
        self.is_multilabel = is_multilabel
        self.num_labels = len(label_encoder.classes_)
        self.max_len_lxmert = max_len_lxmert
        self.max_len_bart = max_len_bart
        self.rag_context_map = rag_context_map if rag_context_map else {}

        logger.info(f"Dataset created. Samples: {len(samples)}, Multilabel: {is_multilabel}")
        self._validate_features()

    def _validate_features(self):
        """Basic check for missing region features."""
        missing_count = 0
        found_count = 0
        checked_ids = set()
        for i in range(min(20, len(self.samples))): # Check more samples
            sample = self.samples[i]
            sample_id = sample['id']
            image_id = sample.get('image_id', sample_id)
            if sample_id in checked_ids: continue
            checked_ids.add(sample_id)

            # Try both image_id and sample_id for features
            if image_id not in self.region_features_map and sample_id not in self.region_features_map:
                missing_count += 1
                # logger.debug(f"Missing features for sample_id: {sample_id}, image_id: {image_id}")
            else:
                found_count +=1
        if missing_count > 0:
            logger.warning(f"First {len(checked_ids)} unique samples check: {missing_count} missing region features, {found_count} found.")
        else:
             logger.info(f"First {len(checked_ids)} unique samples check: All region features found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id']
        image_id = sample['image_id'] # Use this if different from sample_id for features
        ocr_text = sample['ocr_text']
        figurative_reasoning = sample['qwen_reasoning']  # This is now the figurative_reasoning field
        original_labels = sample['original_labels']

        # --- Prepare Labels ---
        if self.is_multilabel:
            label_tensor = torch.zeros(self.num_labels, dtype=torch.float)
            try:
                if original_labels: # Should be a list
                    label_indices = self.label_encoder.transform(original_labels)
                    for label_idx in label_indices:
                         if 0 <= label_idx < self.num_labels: label_tensor[label_idx] = 1.0
            except ValueError as e: logger.warning(f"Label encoding error for {sample_id}: {e}. Labels: {original_labels}")
            except Exception as e: logger.error(f"Unexpected label error for {sample_id}: {e}")
        else: # Single label
            label_tensor = torch.tensor(-1, dtype=torch.long) # Default invalid
            try:
                 if original_labels is not None: # Should be a string
                      label_idx = self.label_encoder.transform([original_labels])[0]
                      label_tensor = torch.tensor(label_idx, dtype=torch.long)
            except ValueError as e: logger.warning(f"Label encoding error for {sample_id}: {e}. Label: {original_labels}")
            except Exception as e: logger.error(f"Unexpected label error for {sample_id}: {e}")

        # --- Prepare LXMERT Input (OCR + Visual) ---
        lxmert_encoding = self.lxmert_tokenizer(
            ocr_text,
            padding="max_length",
            max_length=self.max_len_lxmert,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        lxmert_input_ids = lxmert_encoding["input_ids"].squeeze(0)
        lxmert_attention_mask = lxmert_encoding["attention_mask"].squeeze(0)
        lxmert_token_type_ids = lxmert_encoding["token_type_ids"].squeeze(0)

        # Load Region Features
        visual_feats = torch.zeros((1, 1)) # Dummy default
        visual_pos = torch.zeros((1, 4)) # Dummy default
        num_boxes = 0
        vis_attention_mask = torch.tensor([0], dtype=torch.long) # Dummy

        region_data = self.region_features_map.get(image_id, self.region_features_map.get(sample_id)) # Try both IDs
        if region_data and isinstance(region_data, dict):
             try:
                 # Expecting keys like 'features', 'boxes', potentially 'num_boxes'
                 # Adjust keys based on your actual .pt file structure
                 visual_feats = region_data['features'] # Shape: (num_boxes, feature_dim, e.g., 2048)
                 visual_pos = region_data['boxes']     # Shape: (num_boxes, 4) - normalized coordinates?
                 num_boxes = visual_feats.shape[0]

                 # LXMERT might need a visual attention mask (all 1s for present boxes)
                 vis_attention_mask = torch.ones(num_boxes, dtype=torch.long)

                 # Ensure correct types
                 if not isinstance(visual_feats, torch.Tensor): visual_feats = torch.tensor(visual_feats)
                 if not isinstance(visual_pos, torch.Tensor): visual_pos = torch.tensor(visual_pos)
                 visual_feats = visual_feats.float()
                 visual_pos = visual_pos.float()

             except KeyError as e:
                 logger.warning(f"Missing key {e} in region feature data for {image_id}/{sample_id}. Using defaults.")
                 visual_feats = torch.zeros((1, 2048)) # Default size for LXMERT features often 2048
                 visual_pos = torch.zeros((1, 4))
                 num_boxes = 0
                 vis_attention_mask = torch.tensor([0], dtype=torch.long)
             except Exception as e:
                  logger.error(f"Error processing region features for {image_id}/{sample_id}: {e}. Using defaults.")
                  visual_feats = torch.zeros((1, 2048)); visual_pos = torch.zeros((1, 4)); num_boxes = 0; vis_attention_mask = torch.tensor([0], dtype=torch.long)
        else:
             logger.debug(f"Region features not found for {image_id}/{sample_id}. Using defaults.")
             visual_feats = torch.zeros((1, 2048)); visual_pos = torch.zeros((1, 4)); num_boxes = 0; vis_attention_mask = torch.tensor([0], dtype=torch.long)


        # --- Prepare MentalBART Input (Figurative Reasoning + RAG Context) ---
        rag_context = self.rag_context_map.get(sample_id, " [No RAG Context] ")
        bart_input_text = f"Reasoning: {figurative_reasoning}{rag_context}" # Combine reasoning and RAG

        bart_encoding = self.bart_tokenizer(
            bart_input_text,
            padding="max_length",
            max_length=self.max_len_bart,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        bart_input_ids = bart_encoding["input_ids"].squeeze(0)
        bart_attention_mask = bart_encoding["attention_mask"].squeeze(0)

        return {
            # LXMERT inputs
            "lxmert_input_ids": lxmert_input_ids,
            "lxmert_attention_mask": lxmert_attention_mask,
            "lxmert_token_type_ids": lxmert_token_type_ids,
            "visual_feats": visual_feats,
            "visual_pos": visual_pos,
            "vis_attention_mask": vis_attention_mask, # Visual attention mask

            # MentalBART inputs
            "bart_input_ids": bart_input_ids,
            "bart_attention_mask": bart_attention_mask,

            # Label
            "labels": label_tensor
        }

# --- Hybrid Model Architecture ---
class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for enhancing interaction between modalities."""
    def __init__(self, query_dim, key_dim, dropout_prob=0.1):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, key_dim)
        self.key_projection = nn.Linear(key_dim, key_dim)
        self.value_projection = nn.Linear(key_dim, key_dim)
        self.scale = key_dim ** 0.5
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, query, key_value):
        """
        Compute cross-attention: query attends to key_value
        Args:
            query: tensor of shape [batch_size, query_dim]
            key_value: tensor of shape [batch_size, key_dim]
        Returns:
            attended_features: tensor of shape [batch_size, key_dim]
        """
        # Project query to match key dimensions
        query_proj = self.query_projection(query).unsqueeze(1)  # [batch_size, 1, key_dim]
        
        # Project keys and values
        key_proj = self.key_projection(key_value).unsqueeze(1)  # [batch_size, 1, key_dim]
        value_proj = self.value_projection(key_value).unsqueeze(1)  # [batch_size, 1, key_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(query_proj, key_proj.transpose(-1, -2)) / self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        attended_features = torch.matmul(attention_probs, value_proj).squeeze(1)  # [batch_size, key_dim]
        
        return attended_features

class HybridFusionModel(nn.Module):
    def __init__(self,
                 num_labels: int,
                 lxmert_model_name: str = LXMERT_MODEL_NAME,
                 bart_model_name: str = MENTALBART_MODEL_NAME,
                 is_multilabel: bool = False,
                 dropout_prob: float = DROPOUT_PROB):
        super().__init__()
        self.num_labels = num_labels
        self.is_multilabel = is_multilabel
        logger.info("Initializing HybridFusionModel...")
        logger.info(f"  LXMERT base: {lxmert_model_name}")
        logger.info(f"  BART base: {bart_model_name}")
        logger.info(f"  Num Labels: {num_labels}, Multilabel: {is_multilabel}")

        # Load LXMERT base model
        try:
            self.lxmert_config = LxmertConfig.from_pretrained(lxmert_model_name)
            self.lxmert = LxmertModel.from_pretrained(lxmert_model_name)
            self.lxmert_hidden_size = self.lxmert_config.hidden_size # Usually 768
            logger.info(f"  LXMERT loaded. Hidden size: {self.lxmert_hidden_size}")
        except Exception as e: logger.error(f"Failed to load LXMERT: {e}", exc_info=True); raise

        # Load MentalBART base model (only need encoder)
        try:
            self.bart = BartModel.from_pretrained(bart_model_name)
            self.bart_hidden_size = self.bart.config.hidden_size # Usually 768 for base
            logger.info(f"  MentalBART loaded. Hidden size: {self.bart_hidden_size}")
        except Exception as e: logger.error(f"Failed to load MentalBART: {e}", exc_info=True); raise

        # Cross-modal attention layers
        self.lxmert_to_bart_attention = CrossModalAttention(
            query_dim=self.lxmert_hidden_size, 
            key_dim=self.bart_hidden_size,
            dropout_prob=dropout_prob
        )
        self.bart_to_lxmert_attention = CrossModalAttention(
            query_dim=self.bart_hidden_size, 
            key_dim=self.lxmert_hidden_size,
            dropout_prob=dropout_prob
        )
        
        # Gating mechanisms to control information flow
        self.lxmert_gate = nn.Sequential(
            nn.Linear(self.lxmert_hidden_size * 2, self.lxmert_hidden_size),
            nn.Sigmoid()
        )
        self.bart_gate = nn.Sequential(
            nn.Linear(self.bart_hidden_size * 2, self.bart_hidden_size),
            nn.Sigmoid()
        )
        
        # Feature enhancement layers
        self.lxmert_enhance = nn.Sequential(
            nn.Linear(self.lxmert_hidden_size * 2, self.lxmert_hidden_size),
            nn.LayerNorm(self.lxmert_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.bart_enhance = nn.Sequential(
            nn.Linear(self.bart_hidden_size * 2, self.bart_hidden_size),
            nn.LayerNorm(self.bart_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Fusion Layer - now handles enhanced representations
        self.fusion_dim = self.lxmert_hidden_size + self.bart_hidden_size
        logger.info(f"  Fusion input dimension: {self.fusion_dim}")
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4) # Further reduce dim
        )
        self.classifier_input_dim = self.fusion_dim // 4

        # Final Classifier
        self.classifier = nn.Linear(self.classifier_input_dim, num_labels)
        logger.info(f"  Classifier input dim: {self.classifier_input_dim}, output dim: {num_labels}")

        # Loss Function
        self.loss_fct = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
        logger.info(f"  Using loss: {'BCEWithLogitsLoss' if is_multilabel else 'CrossEntropyLoss'}")
        logger.info(f"  Cross-Modal Attention added for enhanced modality interaction")


    def forward(self,
                lxmert_input_ids: torch.Tensor,
                lxmert_attention_mask: torch.Tensor,
                lxmert_token_type_ids: torch.Tensor,
                visual_feats: torch.Tensor,
                visual_pos: torch.Tensor,
                vis_attention_mask: torch.Tensor, 
                bart_input_ids: torch.Tensor,
                bart_attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):

        # 1. Process with LXMERT
        try:
            lxmert_outputs = self.lxmert(
                input_ids=lxmert_input_ids,
                attention_mask=lxmert_attention_mask,
                visual_feats=visual_feats,
                visual_pos=visual_pos,
                token_type_ids=lxmert_token_type_ids,
                visual_attention_mask=vis_attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
            H_lxmert = lxmert_outputs.pooled_output
        except Exception as e:
            logger.error(f"Error in LXMERT forward pass: {e}", exc_info=True)
            H_lxmert = torch.zeros((lxmert_input_ids.shape[0], self.lxmert_hidden_size), device=lxmert_input_ids.device)

        # 2. Process with MentalBART Encoder
        try:
            bart_encoder_outputs = self.bart.encoder(
                input_ids=bart_input_ids,
                attention_mask=bart_attention_mask,
                return_dict=True
            )
            H_context = bart_encoder_outputs.last_hidden_state[:, 0, :]
        except Exception as e:
            logger.error(f"Error in MentalBART encoder forward pass: {e}", exc_info=True)
            H_context = torch.zeros((bart_input_ids.shape[0], self.bart_hidden_size), device=bart_input_ids.device)

        # 3. Cross-Modal Attention - Let each modality attend to the other
        # LXMERT attends to BART
        H_lxmert_attends_to_bart = self.lxmert_to_bart_attention(H_lxmert, H_context)
        # BART attends to LXMERT
        H_bart_attends_to_lxmert = self.bart_to_lxmert_attention(H_context, H_lxmert)
        
        # 4. Gating Mechanism - Control information flow
        lxmert_gate_values = self.lxmert_gate(torch.cat([H_lxmert, H_lxmert_attends_to_bart], dim=1))
        bart_gate_values = self.bart_gate(torch.cat([H_context, H_bart_attends_to_lxmert], dim=1))
        
        # 5. Apply gates and enhance features
        H_lxmert_enhanced_input = torch.cat([
            H_lxmert,
            lxmert_gate_values * H_lxmert_attends_to_bart
        ], dim=1)
        
        H_bart_enhanced_input = torch.cat([
            H_context,
            bart_gate_values * H_bart_attends_to_lxmert
        ], dim=1)
        
        # Apply enhancement layers
        H_lxmert_enhanced = self.lxmert_enhance(H_lxmert_enhanced_input)
        H_bart_enhanced = self.bart_enhance(H_bart_enhanced_input)

        # 6. Fuse enhanced representations
        combined_features = torch.cat((H_lxmert_enhanced, H_bart_enhanced), dim=1)
        fused_output = self.fusion_layer(combined_features)

        # 7. Classify
        logits = self.classifier(fused_output)

        # 8. Calculate Loss
        loss = None
        if labels is not None:
             # Ensure labels are on the same device as logits
             labels = labels.to(logits.device)
             if self.is_multilabel:
                 loss = self.loss_fct(logits, labels.float())
             else: # Single-label
                 # Ensure labels are long type for CrossEntropy
                 loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        # Return results
        class HybridOutput: pass
        output = HybridOutput()
        output.loss = loss
        output.logits = logits
        return output


# --- Training Function (Adapted) ---
def train_and_evaluate(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: HybridFusionModel, # Use the correct model type hint
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int,
    output_dir: str,
    label_encoder: LabelEncoder,
    is_multilabel: bool
) -> Tuple[Dict[str, List], str]:
    """Trains and evaluates a single HybridFusionModel instance."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting training run. Output: {output_dir}")
    logger.info(f"Task Type: {'Multi-Label' if is_multilabel else 'Single-Label'}")

    best_val_f1_macro = 0.0
    best_model_path = os.path.join(output_dir, "best_model_macro_f1.pt")
    last_model_path = os.path.join(output_dir, "last_model.pt")
    log_file = os.path.join(output_dir, "training_log.csv")
    training_logs = []
    history = defaultdict(list)
    total_steps = len(train_dataloader) * num_epochs

    try:
        for epoch in range(num_epochs):
            epoch_num = epoch + 1
            logger.info(f"--- Epoch {epoch_num}/{num_epochs} ---")
            model.train()
            total_train_loss = 0.0
            all_train_logits, all_train_labels_raw = [], []
            train_progress = tqdm(train_dataloader, desc=f"Train Epoch {epoch_num}", leave=False)

            for batch_idx, batch in enumerate(train_progress):
                optimizer.zero_grad()
                # Move all input tensors to the device
                # LXMERT inputs
                lxmert_input_ids = batch["lxmert_input_ids"].to(device)
                lxmert_attention_mask = batch["lxmert_attention_mask"].to(device)
                lxmert_token_type_ids = batch["lxmert_token_type_ids"].to(device)
                visual_feats = batch["visual_feats"].to(device)
                visual_pos = batch["visual_pos"].to(device)
                vis_attention_mask = batch["vis_attention_mask"].to(device)
                # BART inputs
                bart_input_ids = batch["bart_input_ids"].to(device)
                bart_attention_mask = batch["bart_attention_mask"].to(device)
                # Labels (handle potential placement issues in Dataset)
                labels = batch["labels"].to(device)

                try:
                    outputs = model(
                        lxmert_input_ids=lxmert_input_ids,
                        lxmert_attention_mask=lxmert_attention_mask,
                        lxmert_token_type_ids=lxmert_token_type_ids,
                        visual_feats=visual_feats,
                        visual_pos=visual_pos,
                        vis_attention_mask=vis_attention_mask,
                        bart_input_ids=bart_input_ids,
                        bart_attention_mask=bart_attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits

                    if loss is None: logger.error(f"Epoch {epoch_num}, Batch {batch_idx}: Loss is None."); continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    total_train_loss += loss.item()

                    all_train_logits.append(logits.detach().cpu())
                    all_train_labels_raw.append(batch["labels"].cpu()) # Get original labels from batch before moving

                    train_progress.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

                except Exception as e:
                    logger.error(f"Error during training batch {batch_idx} epoch {epoch_num}: {e}", exc_info=True)
                    # Log details about the batch that failed if possible
                    logger.error(f"Failing Batch Keys: {list(batch.keys())}")
                    logger.error(f"Failing Batch Shapes (example): lxmert_input_ids={batch['lxmert_input_ids'].shape if 'lxmert_input_ids' in batch else 'N/A'}")
                    # Consider breaking or continuing based on severity
                    # break # Stop epoch on error
                    continue # Continue epoch

            avg_train_loss = total_train_loss / len(train_dataloader) if train_dataloader else 0.0
            # --- Calculate training metrics ---
            train_f1_micro, train_f1_macro, train_accuracy, train_hamming = 0.0, 0.0, 0.0, 0.0
            if all_train_logits and all_train_labels_raw:
                all_train_logits_cat = torch.cat(all_train_logits, dim=0)
                all_train_labels_raw_cat = torch.cat(all_train_labels_raw, dim=0)
                if is_multilabel:
                    train_probs = torch.sigmoid(all_train_logits_cat).numpy(); train_preds = (train_probs > 0.5).astype(int); train_labels = all_train_labels_raw_cat.numpy().astype(int)
                    train_f1_micro = f1_score(train_labels, train_preds, average="micro", zero_division=0); train_f1_macro = f1_score(train_labels, train_preds, average="macro", zero_division=0)
                    train_accuracy = accuracy_score(train_labels, train_preds); train_hamming = hamming_loss(train_labels, train_preds)
                    logger.info(f"Epoch {epoch_num} Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}, Hamming: {train_hamming:.4f}, MicroF1: {train_f1_micro:.4f}, MacroF1: {train_f1_macro:.4f}")
                else:
                    train_preds = torch.argmax(all_train_logits_cat, dim=1).numpy(); train_labels = all_train_labels_raw_cat.numpy()
                    train_f1_micro = f1_score(train_labels, train_preds, average="micro", zero_division=0); train_f1_macro = f1_score(train_labels, train_preds, average="macro", zero_division=0)
                    train_accuracy = accuracy_score(train_labels, train_preds); train_hamming = 0.0 # Or hamming_loss(train_labels, train_preds)
                    logger.info(f"Epoch {epoch_num} Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}, MacroF1: {train_f1_macro:.4f}")


            # --- Validation Phase ---
            model.eval()
            total_val_loss = 0.0
            all_val_logits, all_val_labels_raw = [], []
            val_progress = tqdm(val_dataloader, desc=f"Val Epoch {epoch_num}", leave=False)
            with torch.no_grad():
                for batch in val_progress:
                    lxmert_input_ids = batch["lxmert_input_ids"].to(device); lxmert_attention_mask = batch["lxmert_attention_mask"].to(device)
                    lxmert_token_type_ids = batch["lxmert_token_type_ids"].to(device); visual_feats = batch["visual_feats"].to(device)
                    visual_pos = batch["visual_pos"].to(device); vis_attention_mask = batch["vis_attention_mask"].to(device)
                    bart_input_ids = batch["bart_input_ids"].to(device); bart_attention_mask = batch["bart_attention_mask"].to(device)
                    labels = batch["labels"].to(device) # Move labels for loss calculation if needed by model

                    try:
                         outputs = model( # Pass all required args
                             lxmert_input_ids=lxmert_input_ids, lxmert_attention_mask=lxmert_attention_mask, lxmert_token_type_ids=lxmert_token_type_ids,
                             visual_feats=visual_feats, visual_pos=visual_pos, vis_attention_mask=vis_attention_mask,
                             bart_input_ids=bart_input_ids, bart_attention_mask=bart_attention_mask,
                             labels=labels # Pass labels for potential loss calculation within model
                         )
                         if outputs.loss is not None: total_val_loss += outputs.loss.item()
                         all_val_logits.append(outputs.logits.cpu())
                         all_val_labels_raw.append(batch["labels"].cpu()) # Store original CPU labels
                    except Exception as e: logger.error(f"Validation batch error epoch {epoch_num}: {e}", exc_info=True); continue

            avg_val_loss = total_val_loss / len(val_dataloader) if val_dataloader else 0.0
            # --- Calculate validation metrics ---
            val_f1_micro, val_f1_macro, val_accuracy, val_hamming = 0.0, 0.0, 0.0, 0.0
            if all_val_logits and all_val_labels_raw:
                all_val_logits_cat = torch.cat(all_val_logits, dim=0)
                all_val_labels_raw_cat = torch.cat(all_val_labels_raw, dim=0)
                if is_multilabel:
                    val_probs = torch.sigmoid(all_val_logits_cat).numpy(); val_preds = (val_probs > 0.5).astype(int); val_labels = all_val_labels_raw_cat.numpy().astype(int)
                    val_f1_micro = f1_score(val_labels, val_preds, average="micro", zero_division=0); val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)
                    val_accuracy = accuracy_score(val_labels, val_preds); val_hamming = hamming_loss(val_labels, val_preds)
                    logger.info(f"Epoch {epoch_num} Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}, Hamming: {val_hamming:.4f}, MicroF1: {val_f1_micro:.4f}, MacroF1: {val_f1_macro:.4f}")
                else:
                    val_preds = torch.argmax(all_val_logits_cat, dim=1).numpy(); val_labels = all_val_labels_raw_cat.numpy()
                    val_f1_micro = f1_score(val_labels, val_preds, average="micro", zero_division=0); val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)
                    val_accuracy = accuracy_score(val_labels, val_preds); val_hamming = hamming_loss(val_labels, val_preds) # = 1 - accuracy
                    logger.info(f"Epoch {epoch_num} Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}, MacroF1: {val_f1_macro:.4f}")

            # --- Logging and Saving ---
            history['epoch'].append(epoch_num); history['train_loss'].append(avg_train_loss); history['val_loss'].append(avg_val_loss)
            history['train_f1_macro'].append(train_f1_macro); history['val_f1_macro'].append(val_f1_macro); # Add others as needed
            history['val_accuracy'].append(val_accuracy); history['val_hamming'].append(val_hamming)

            current_log = { 'epoch': epoch_num, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_f1_macro': val_f1_macro, 'val_accuracy': val_accuracy }
            training_logs.append(current_log)

            if val_f1_macro > best_val_f1_macro:
                best_val_f1_macro = val_f1_macro
                logger.info(f"Epoch {epoch_num}: New best validation Macro F1: {best_val_f1_macro:.4f}. Saving model...")
                torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), last_model_path)
            pd.DataFrame(training_logs).to_csv(log_file, index=False)

            # Clean up
            del all_train_logits, all_train_labels_raw, all_val_logits, all_val_labels_raw
            if 'all_train_logits_cat' in locals(): del all_train_logits_cat, all_train_labels_raw_cat
            if 'all_val_logits_cat' in locals(): del all_val_logits_cat, all_val_labels_raw_cat
            gc.collect()
            if device == torch.device('cuda'): torch.cuda.empty_cache()

        logger.info(f"Training finished. Best Val Macro F1: {best_val_f1_macro:.4f}")

    except KeyboardInterrupt: logger.warning("Training interrupted."); torch.save(model.state_dict(), last_model_path); pd.DataFrame(training_logs).to_csv(log_file, index=False)
    except Exception as e: logger.error(f"Unexpected error during training: {e}", exc_info=True); pd.DataFrame(training_logs).to_csv(log_file, index=False); raise

    return history, best_model_path

# --- Split Data Function ---
def split_data(data, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    Uses stratified split based on 'stratify_label' when available.
    """
    if not data:
        logger.warning("Empty data provided to split_data")
        return [], [], []
    
    # Get stratify values if available
    stratify = [sample.get('stratify_label') for sample in data if 'stratify_label' in sample]
    stratify = stratify if len(stratify) == len(data) else None
    
    if test_size > 0:
        # First split: train+val vs test
        train_val, test = train_test_split(
            data, test_size=test_size, 
            random_state=random_state,
            stratify=stratify
        )
        
        # Update stratify for the second split
        if stratify:
            stratify_train_val = [sample.get('stratify_label') for sample in train_val]
        else:
            stratify_train_val = None
        
        # Second split: train vs val
        adjusted_val_size = val_size / (1 - test_size)  # Adjust val_size relative to train_val size
        train, val = train_test_split(
            train_val, test_size=adjusted_val_size,
            random_state=random_state,
            stratify=stratify_train_val
        )
        
        return train, val, test
    else:
        # Only split train vs val
        train, val = train_test_split(
            data, test_size=val_size,
            random_state=random_state,
            stratify=stratify
        )
        return train, val, []

# --- Ensemble Evaluation Function ---
def evaluate_ensemble(
    model_paths: List[str],
    data_loader: DataLoader,
    device: torch.device,
    label_encoder: LabelEncoder,
    is_multilabel: bool,
    num_labels: int,
    output_dir: str,
    output_prefix: str,
    lxmert_model_name: str = LXMERT_MODEL_NAME,
    bart_model_name: str = MENTALBART_MODEL_NAME
) -> Dict[str, float]:
    """
    Evaluates an ensemble of models on the provided data loader.
    Loads each model, gets predictions, ensembles them, and calculates metrics.
    
    Args:
        model_paths: List of paths to trained model checkpoints
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        label_encoder: Label encoder for classes
        is_multilabel: Whether task is multilabel or single-label
        num_labels: Number of label classes
        output_dir: Directory to save results
        output_prefix: Prefix for output files
        lxmert_model_name: LXMERT model name/path
        bart_model_name: BART model name/path
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating ensemble of {len(model_paths)} models")
    if not model_paths:
        logger.error("No model paths provided for ensemble evaluation")
        return {}
    
    all_logits = []
    all_labels = []
    
    # Get predictions from each model
    for model_idx, model_path in enumerate(model_paths):
        logger.info(f"Loading model {model_idx+1}/{len(model_paths)} from {model_path}")
        try:
            # Initialize model
            model = HybridFusionModel(
                num_labels=num_labels,
                lxmert_model_name=lxmert_model_name,
                bart_model_name=bart_model_name,
                is_multilabel=is_multilabel
            ).to(device)
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Get predictions
            model_logits = []
            if model_idx == 0:  # Only collect labels once
                with torch.no_grad():
                    for batch in tqdm(data_loader, desc=f"Eval Model {model_idx+1}"):
                        # Move inputs to device
                        lxmert_input_ids = batch["lxmert_input_ids"].to(device)
                        lxmert_attention_mask = batch["lxmert_attention_mask"].to(device)
                        lxmert_token_type_ids = batch["lxmert_token_type_ids"].to(device)
                        visual_feats = batch["visual_feats"].to(device)
                        visual_pos = batch["visual_pos"].to(device)
                        vis_attention_mask = batch["vis_attention_mask"].to(device)
                        bart_input_ids = batch["bart_input_ids"].to(device)
                        bart_attention_mask = batch["bart_attention_mask"].to(device)
                        labels = batch["labels"].cpu()  # Keep labels on CPU
                        
                        # Forward pass
                        outputs = model(
                            lxmert_input_ids=lxmert_input_ids,
                            lxmert_attention_mask=lxmert_attention_mask,
                            lxmert_token_type_ids=lxmert_token_type_ids,
                            visual_feats=visual_feats,
                            visual_pos=visual_pos,
                            vis_attention_mask=vis_attention_mask,
                            bart_input_ids=bart_input_ids,
                            bart_attention_mask=bart_attention_mask
                        )
                        
                        model_logits.append(outputs.logits.cpu())
                        all_labels.append(labels)
            else:
                with torch.no_grad():
                    for batch in tqdm(data_loader, desc=f"Eval Model {model_idx+1}"):
                        # Move inputs to device
                        lxmert_input_ids = batch["lxmert_input_ids"].to(device)
                        lxmert_attention_mask = batch["lxmert_attention_mask"].to(device)
                        lxmert_token_type_ids = batch["lxmert_token_type_ids"].to(device)
                        visual_feats = batch["visual_feats"].to(device)
                        visual_pos = batch["visual_pos"].to(device)
                        vis_attention_mask = batch["vis_attention_mask"].to(device)
                        bart_input_ids = batch["bart_input_ids"].to(device)
                        bart_attention_mask = batch["bart_attention_mask"].to(device)
                        
                        # Forward pass
                        outputs = model(
                            lxmert_input_ids=lxmert_input_ids,
                            lxmert_attention_mask=lxmert_attention_mask,
                            lxmert_token_type_ids=lxmert_token_type_ids,
                            visual_feats=visual_feats,
                            visual_pos=visual_pos,
                            vis_attention_mask=vis_attention_mask,
                            bart_input_ids=bart_input_ids,
                            bart_attention_mask=bart_attention_mask
                        )
                        
                        model_logits.append(outputs.logits.cpu())
            
            # Combine logits for this model
            all_logits.append(torch.cat(model_logits, dim=0))
            
            # Clean up to save memory
            del model, model_logits
            gc.collect()
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error evaluating model {model_idx+1}: {e}", exc_info=True)
            continue
    
    if not all_logits:
        logger.error("No models successfully evaluated")
        return {}
    
    # Combine predictions from all models
    ensemble_logits = torch.stack(all_logits).mean(dim=0)  # Average logits
    all_labels_cat = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    metrics = {}
    try:
        if is_multilabel:
            ensemble_probs = torch.sigmoid(ensemble_logits).numpy()
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            labels_np = all_labels_cat.numpy().astype(int)
            
            metrics['accuracy'] = accuracy_score(labels_np, ensemble_preds)
            metrics['f1_micro'] = f1_score(labels_np, ensemble_preds, average='micro', zero_division=0)
            metrics['f1_macro'] = f1_score(labels_np, ensemble_preds, average='macro', zero_division=0)
            metrics['hamming_loss'] = hamming_loss(labels_np, ensemble_preds)
            
            # Generate classification report
            class_names = label_encoder.classes_
            report = classification_report(
                labels_np, ensemble_preds, 
                target_names=class_names, 
                zero_division=0,
                output_dict=True
            )
            
            # Save confusion matrices for each class
            mcm = multilabel_confusion_matrix(labels_np, ensemble_preds)
            
        else:  # Single-label
            ensemble_preds = torch.argmax(ensemble_logits, dim=1).numpy()
            labels_np = all_labels_cat.numpy()
            
            metrics['accuracy'] = accuracy_score(labels_np, ensemble_preds)
            metrics['f1_micro'] = f1_score(labels_np, ensemble_preds, average='micro', zero_division=0)
            metrics['f1_macro'] = f1_score(labels_np, ensemble_preds, average='macro', zero_division=0)
            
            # Generate classification report
            class_names = label_encoder.classes_
            report = classification_report(
                labels_np, ensemble_preds, 
                target_names=class_names, 
                zero_division=0,
                output_dict=True
            )
        
        # Log and save results
        logger.info(f"Ensemble Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save detailed report
        results_file = os.path.join(output_dir, f"{output_prefix}_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'classification_report': report
            }, f, indent=2)
        
        # Save predictions
        preds_file = os.path.join(output_dir, f"{output_prefix}_predictions.npz")
        np.savez(
            preds_file,
            ensemble_logits=ensemble_logits.numpy(),
            ensemble_preds=ensemble_preds,
            true_labels=labels_np
        )
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Predictions saved to {preds_file}")
        
    except Exception as e:
        logger.error(f"Error calculating ensemble metrics: {e}", exc_info=True)
    
    return metrics

# --- Main Pipeline Function (Adapted) ---
def run_hybrid_ensemble_pipeline(
    dataset_type: str = 'anxiety',
    num_ensemble_runs: int = NUM_ENSEMBLE_MODELS,
    seed: int = BASE_SEED,
    # Add other parameters if needed (batch size, epochs, lr etc. Use constants for now)
):
    is_multilabel = (dataset_type == 'depression')
    # Dynamically set output directory base
    output_dir_base = os.path.join(SRC_DIR, dataset_type, "output", "hybrid_pipeline_output")
    pipeline_output_dir = output_dir_base # Use this as the main output for this run
    os.makedirs(pipeline_output_dir, exist_ok=True)
    logger.info(f"===== STARTING HYBRID PIPELINE for: {dataset_type} =====")
    logger.info(f"Output directory: {pipeline_output_dir}")

    # --- 1. Load Data (using combined JSONs with figurative reasoning) ---
    if dataset_type == 'anxiety':
        # Updated paths to the cleaned data directory
        data_dir = os.path.join(ANXIETY_DATA_DIR, "final", "cleaned")
        train_file = os.path.join(data_dir, "anxiety_train_combined_preprocessed.json")
        # No separate val file for anxiety as per structure
        val_file = None # Explicitly set to None
        test_file = os.path.join(data_dir, "anxiety_test_combined_preprocessed.json")
    else: # depression
        # Updated paths to the cleaned data directory
        data_dir = os.path.join(DEPRESSION_DATA_DIR, "final", "cleaned")
        train_file = os.path.join(data_dir, "depressive_train_combined_preprocessed.json")
        val_file = os.path.join(data_dir, "depressive_val_combined_preprocessed.json")
        test_file = os.path.join(data_dir, "depressive_test_combined_preprocessed.json")

    logger.info("--- Step 1: Loading Data ---")
    train_data_full = load_qwen_data(train_file) # Load the full training data first
    test_data = load_qwen_data(test_file) if os.path.exists(test_file) else []

    if not train_data_full: logger.error("Failed to load train data. Aborting."); return

    # Handle validation data loading/splitting
    if val_file and os.path.exists(val_file):
        logger.info(f"Loading validation data from specified file: {val_file}")
        val_data = load_qwen_data(val_file)
        train_data = train_data_full # Use the full loaded training data
    elif dataset_type == 'anxiety':
        logger.info("Anxiety dataset: No validation file specified. Splitting train data (82% train, 18% val).")
        train_data, val_data, _ = split_data(train_data_full, val_size=0.18, test_size=0, random_state=seed)
    else:
        # Default split if val file doesn't exist for other types (e.g., depression)
        logger.warning(f"Validation file not found at {val_file}. Splitting train data (default 85% train, 15% val).")
        train_data, val_data, _ = split_data(train_data_full, val_size=0.15, test_size=0, random_state=seed)

    logger.info(f"Data sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    if not train_data or not val_data: logger.error("Train or Val data empty after loading/split. Aborting."); return

    # --- 2. Load Region Features ---
    logger.info("--- Step 2: Loading Region Features ---")
    features_train = load_region_features(REGION_FEATURES_DIR, dataset_type, "train")
    features_test = load_region_features(REGION_FEATURES_DIR, dataset_type, "test") if test_data else {}

    # Load validation features if the file exists, otherwise validation split will use training features
    val_feature_file_exists = os.path.exists(os.path.join(REGION_FEATURES_DIR, f"{dataset_type}_val_region_features.pt"))
    if val_feature_file_exists:
        logger.info("Loading separate validation region features.")
        features_val = load_region_features(REGION_FEATURES_DIR, dataset_type, "val")
    else:
        logger.info("No separate validation region features file found. Validation samples will use features from the training set map.")
        features_val = {} # Initialize empty, train features will cover val IDs if split from train

    # Combine features: Test first, then Val, then Train to ensure precedence if keys overlap (though unlikely with good IDs)
    region_features_map = {**features_test, **features_val, **features_train}
    logger.info(f"Total unique region features loaded into map: {len(region_features_map)}")

    # --- 3. Label Encoding ---
    logger.info("--- Step 3: Encoding Labels ---")
    all_labels = set()
    for d in [train_data, val_data, test_data]:
        for s in d:
            lbls = s['original_labels']
            if isinstance(lbls, list): all_labels.update(l for l in lbls if l)
            elif isinstance(lbls, str) and lbls: all_labels.add(lbls)
    if not all_labels: logger.error("No labels found."); return
    label_encoder = LabelEncoder().fit(sorted(list(all_labels)))
    num_labels = len(label_encoder.classes_)
    logger.info(f"Labels encoded. Num classes: {num_labels}. Classes: {label_encoder.classes_}")

    # --- 4. RAG Setup ---
    logger.info("--- Step 4: Setting up RAG ---")
    rag_embeddings = None; rag_retriever = None; rag_context_map = {}
    try:
        # Define embedding file path
        rag_embed_file = os.path.join(pipeline_output_dir, f"{dataset_type}_rag_embeddings.pt")
        logger.info(f"RAG embedding file path: {rag_embed_file}")

        # Check if embeddings file already exists
        if os.path.exists(rag_embed_file):
            logger.info(f"Attempting to load pre-computed RAG embeddings from {rag_embed_file}")
            try:
                loaded_data = torch.load(rag_embed_file, map_location='cpu')
                # Check if loaded data is a tensor or numpy array
                if isinstance(loaded_data, torch.Tensor):
                    rag_embeddings = loaded_data.numpy() # Convert to numpy for FAISS
                elif isinstance(loaded_data, np.ndarray):
                    rag_embeddings = loaded_data
                else:
                    logger.warning(f"Loaded RAG embeddings file contains unexpected type: {type(loaded_data)}. Regenerating.")
                    rag_embeddings = None # Force regeneration

                if rag_embeddings is not None:
                    logger.info(f"Loaded RAG embeddings with shape: {rag_embeddings.shape}")

            except Exception as e:
                logger.error(f"Failed to load pre-computed RAG embeddings from {rag_embed_file}: {e}", exc_info=True)
                rag_embeddings = None # Reset to None so we regenerate

        # Generate embeddings if not loaded from file
        if rag_embeddings is None:
            logger.info("Generating new RAG DB embeddings (Train OCR + Figurative Reasoning)...")
            rag_embed_generator = EmbeddingGeneratorRAG(model_name=RAG_EMBEDDING_MODEL, device=device)
            train_ocr = [s['ocr_text'] for s in train_data]
            train_figurative = [s['qwen_reasoning'] for s in train_data] # Using the figurative reasoning
            rag_embeddings = rag_embed_generator.generate_fused_embeddings(train_ocr, train_figurative)

            # Save embeddings for future use if generation was successful
            if rag_embeddings is not None:
                logger.info(f"Saving generated RAG embeddings to {rag_embed_file}")
                try:
                    # Ensure saving as a Tensor for consistency and potential efficiency
                    if isinstance(rag_embeddings, np.ndarray):
                        rag_embeddings_tensor = torch.from_numpy(rag_embeddings)
                    elif isinstance(rag_embeddings, torch.Tensor):
                         rag_embeddings_tensor = rag_embeddings # Already a tensor
                    else:
                         logger.error(f"Generated embeddings are of unexpected type {type(rag_embeddings)}. Cannot save.")
                         rag_embeddings_tensor = None # Prevent saving

                    if rag_embeddings_tensor is not None:
                        torch.save(rag_embeddings_tensor, rag_embed_file)
                        logger.info(f"RAG embeddings saved successfully.")

                except Exception as e:
                    logger.error(f"Failed to save RAG embeddings to {rag_embed_file}: {e}", exc_info=True)
                    # Continue even if saving fails, but log the error

            # Clean up generator immediately after use
            del rag_embed_generator; gc.collect(); torch.cuda.empty_cache()

        # Proceed with RAG index building and context generation if embeddings are available
        if rag_embeddings is not None:
            logger.info("Building RAG FAISS index...")
            rag_retriever = RAGRetrieverFAISS(rag_embeddings, top_k=RETRIEVAL_K)
            if rag_retriever.index:
                 # Generate RAG context for Val and Test sets
                 logger.info("Generating RAG query embeddings and context...")
                 rag_prompt_constructor = PromptConstructorRAG(train_data, label_encoder)
                 # Use a temporary generator for query embeddings to manage memory
                 temp_embed_gen = EmbeddingGeneratorRAG(model_name=RAG_EMBEDDING_MODEL, device=device)
                 for split_data_, split_name in [(val_data, "Val"), (test_data, "Test")]:
                     if not split_data_: continue
                     logger.info(f"Processing {split_name} for RAG context...")
                     split_ocr = [s['ocr_text'] for s in split_data_]
                     split_figurative = [s['qwen_reasoning'] for s in split_data_] # Using figurative reasoning
                     query_embeddings = temp_embed_gen.generate_fused_embeddings(split_ocr, split_figurative)
                     if query_embeddings is not None:
                         retrieved_indices_batch = rag_retriever.retrieve_similar(query_embeddings)
                         if retrieved_indices_batch is not None:
                             for i, sample in enumerate(tqdm(split_data_, desc=f"Formatting {split_name} RAG")):
                                 sample_id = sample['id']
                                 # Exclude self-retrieval (won't happen here as queries are from val/test)
                                 # Ensure indices are valid before accessing train_data
                                 valid_indices = [idx for idx in retrieved_indices_batch[i][:RETRIEVAL_K] if 0 <= idx < len(train_data)]
                                 rag_context_map[sample_id] = rag_prompt_constructor.format_rag_examples(sample_id, valid_indices)
                         del query_embeddings, retrieved_indices_batch; gc.collect(); torch.cuda.empty_cache()
                 del temp_embed_gen; gc.collect(); torch.cuda.empty_cache() # Clean up temp generator
                 logger.info(f"Generated RAG context for {len(rag_context_map)} samples.")
            else: logger.warning("RAG index build failed. RAG disabled.")
        else: logger.warning("RAG embeddings not available (failed load/generation). RAG disabled.")

    except Exception as e: logger.error(f"Error during RAG setup: {e}", exc_info=True); logger.warning("Proceeding without RAG.")
    finally: # Ensure cleanup even on error
        # Keep rag_context_map, clear embeddings/retriever which might be large
        del rag_embeddings, rag_retriever
        if 'rag_embed_generator' in locals(): del rag_embed_generator
        if 'temp_embed_gen' in locals(): del temp_embed_gen
        gc.collect(); torch.cuda.empty_cache()


    # --- 5. Tokenizers, Datasets, DataLoaders ---
    logger.info("--- Step 5: Loading Tokenizers ---")
    try:
        lxmert_tokenizer = LxmertTokenizer.from_pretrained(LXMERT_MODEL_NAME)
        bart_tokenizer = BartTokenizer.from_pretrained(MENTALBART_MODEL_NAME)
    except Exception as e: logger.error(f"Failed to load tokenizers: {e}", exc_info=True); return

    logger.info("--- Step 6: Creating Datasets ---")
    try:
        train_dataset = MentalHealthMemeDataset(train_data, lxmert_tokenizer, bart_tokenizer, label_encoder, region_features_map, is_multilabel, MAX_LEN_LXMERT, MAX_LEN_BART, rag_context_map={}) # No RAG context for train itself
        val_dataset = MentalHealthMemeDataset(val_data, lxmert_tokenizer, bart_tokenizer, label_encoder, region_features_map, is_multilabel, MAX_LEN_LXMERT, MAX_LEN_BART, rag_context_map)
        test_dataset = MentalHealthMemeDataset(test_data, lxmert_tokenizer, bart_tokenizer, label_encoder, region_features_map, is_multilabel, MAX_LEN_LXMERT, MAX_LEN_BART, rag_context_map) if test_data else None
    except Exception as e: logger.error(f"Failed to create Datasets: {e}", exc_info=True); return

    logger.info("--- Step 7: Creating DataLoaders ---")
    try:
        nw = 0  # Set num_workers to 0 to avoid potential multiprocessing issues
        pm = True if device == torch.device('cuda') else False
        
        # Use the custom_collate_fn to handle variable-sized features
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=nw, 
            pin_memory=pm,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=nw, 
            pin_memory=pm,
            collate_fn=custom_collate_fn
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=nw, 
                pin_memory=pm,
                collate_fn=custom_collate_fn
            )
            
        if not train_loader or not val_loader: 
            raise ValueError("Dataloader creation resulted in empty loader.")
    except Exception as e: 
        logger.error(f"Failed to create DataLoaders: {e}", exc_info=True); 
        return


    # --- 6. Ensemble Training ---
    logger.info(f"--- Step 8: Starting Ensemble Training ({num_ensemble_runs} models) ---")
    trained_model_paths = []
    all_histories = []

    for i in range(num_ensemble_runs):
        run_seed = seed + i
        set_seed(run_seed)
        # Output for individual runs goes into subdirectories of the main pipeline output
        model_run_output_dir = os.path.join(pipeline_output_dir, f"run_{run_seed}")
        logger.info(f"--- Training Model {i + 1}/{num_ensemble_runs} (Seed: {run_seed}) ---")

        model=None; optimizer=None; scheduler=None # Define scope
        try:
            model = HybridFusionModel(num_labels, LXMERT_MODEL_NAME, MENTALBART_MODEL_NAME, is_multilabel, DROPOUT_PROB).to(device)
            optimizer = AdamW(
                model.parameters(), 
                lr=LEARNING_RATE,
                betas=(ADAM_BETA1, ADAM_BETA2),
                eps=ADAM_EPSILON,
                weight_decay=WEIGHT_DECAY
            )
            total_steps = len(train_loader) * NUM_EPOCHS
            warmup_steps = int(0.1 * total_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

            history, best_path = train_and_evaluate(
                train_loader, val_loader, model, optimizer, scheduler, device,
                NUM_EPOCHS, model_run_output_dir, label_encoder, is_multilabel
            )
            trained_model_paths.append(best_path)
            all_histories.append(history)
            # plot_training_history(history, model_run_output_dir, f"_run_{run_seed}") # Add plotting function if needed

        except Exception as train_e:
            logger.error(f"Training failed run {i+1}: {train_e}", exc_info=True)
            if model_run_output_dir: # Save traceback if dir exists
                 tb_path = os.path.join(model_run_output_dir, "error_traceback.txt")
                 try:
                     with open(tb_path, "w") as f: traceback.print_exc(file=f)
                     logger.info(f"Traceback saved: {tb_path}")
                 except: pass # Ignore if cannot save traceback

        finally: # Ensure cleanup
            del model, optimizer, scheduler
            if 'history' in locals(): del history
            gc.collect()
            if device == torch.device('cuda'): torch.cuda.empty_cache()

    logger.info(f"--- Ensemble Training Finished. Trained {len(trained_model_paths)} models. ---")
    if not trained_model_paths: logger.error("No models trained successfully."); return

    # --- 7. Final Evaluation ---
    logger.info("--- Step 9: Evaluating Ensemble ---")
    # ... (Call evaluate_ensemble for val_loader and test_loader if exists) ...
    logger.info("--- Final Validation Eval (Ensemble) ---")
    val_metrics = evaluate_ensemble(
        trained_model_paths, val_loader, device, label_encoder, is_multilabel, num_labels,
        pipeline_output_dir, "validation_ensemble", LXMERT_MODEL_NAME, MENTALBART_MODEL_NAME
    )
    if test_loader:
        logger.info("--- Final Test Eval (Ensemble) ---")
        test_metrics = evaluate_ensemble(
            trained_model_paths, test_loader, device, label_encoder, is_multilabel, num_labels,
            pipeline_output_dir, "test_ensemble", LXMERT_MODEL_NAME, MENTALBART_MODEL_NAME
        )


    # --- 8. Save Artifacts ---
    logger.info("--- Step 10: Saving Label Encoder ---")
    le_path = os.path.join(pipeline_output_dir, "label_encoder.pkl")
    try:
        with open(le_path, 'wb') as f: pickle.dump(label_encoder, f)
        logger.info(f"Label encoder saved: {le_path}")
    except Exception as e: logger.error(f"Failed label encoder save: {e}")

    logger.info(f"===== HYBRID PIPELINE COMPLETED for: {dataset_type} =====")


# --- Execution ---
if __name__ == "__main__":
    logger.info("Script started execution.")
    set_seed(BASE_SEED)

    # Default to depression dataset since that's what the user provided examples for
    dataset = 'anxiety' 
    
    try:
        run_hybrid_ensemble_pipeline(dataset_type=dataset)
    except Exception as main_e:
        logger.critical(f"Unhandled exception in main execution: {main_e}", exc_info=True)

    logger.info("Script finished.")