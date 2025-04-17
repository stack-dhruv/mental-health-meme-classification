import argparse
import json
import logging
import os
import pickle
import random
import re
import time
import traceback
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                           hamming_loss, multilabel_confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from transformers import (BartModel, BartTokenizer,
                          get_linear_schedule_with_warmup)
import gc
import matplotlib.pyplot as plt
import sys
import faiss

# --- Configuration ---
BASE_TRANSFORMER_NAME = 'Tianlin668/MentalBART'  # Changed from 'mental/mental-bart-base-cased'
TOKENIZER_NAME = 'Tianlin668/MentalBART'         # Changed to match base model

REGION_FEATURE_DIM = 2048
MAX_REGIONS = 36

MAX_TEXT_LEN = 128

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
NUM_ENSEMBLE_MODELS = 3
BASE_SEED = 42
ANXIETY_VAL_SPLIT_RATIO = 0.15

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("HybridCrossAttention")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed set to {seed}")

def check_cuda_memory(step_name=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.debug(f"CUDA Memory ({step_name}): Allocated={allocated:.3f} GB, Reserved={reserved:.3f} GB")
    else:
        logger.debug(f"CUDA not available ({step_name}).")

def load_json_data(file_path: str) -> Optional[List[Dict]]:
    logger.info(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records as JSON object.")
        return data
    except json.JSONDecodeError:
        logger.warning(f"Failed to load as JSON object, trying JSON Lines: {file_path}")
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as line_e:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {file_path}: {line_e}")
            if data:
                logger.info(f"Successfully loaded {len(data)} records as JSON Lines.")
                return data
            else:
                logger.error(f"File {file_path} could not be parsed as JSON or JSON Lines.")
                return None
        except Exception as e_jsonl:
            logger.error(f"Error reading file {file_path} as JSON Lines: {e_jsonl}", exc_info=True)
            return None

def clean_triples(text):
    """
    Clean the figurative reasoning text while preserving important structural elements
    like 'Cause-Effect', 'Figurative Understanding', 'Mental State' headers.
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip(): return ""
    
    # Remove numbering but preserve the headers marked with asterisks
    cleaned = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Replace the bold markers with cleaner format but keep the headers
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    return cleaned.strip()

def process_loaded_data(data: List[Dict], is_anxiety: bool) -> List[Dict]:
    filtered_data = []
    processed_ids = set()
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict): continue
        sample_id = sample.get('sample_id', sample.get('id', f"genid_{idx}"))
        if sample_id in processed_ids: continue
        processed_ids.add(sample_id)

        ocr_text = sample.get('ocr_text', "")
        reasoning = sample.get('figurative_reasoning', "")
        image_id = sample.get('image_id', sample_id)

        # Clean the reasoning while preserving important structure
        cleaned_reasoning = clean_triples(reasoning)
        
        # Combine OCR and reasoning with a special separator
        # that helps model distinguish between them
        combined_text = f"{ocr_text} [SEP] {cleaned_reasoning}".strip()
        if combined_text == "[SEP]": combined_text = ""

        if not combined_text:
             logger.warning(f"Skipping sample {sample_id}: missing combined 'ocr_text' and 'reasoning'.")
             continue

        # Add the original text components for potential separate processing
        sample_out = {
            'sample_id': sample_id, 
            'image_id': image_id, 
            'text': combined_text,
            'ocr_text': ocr_text,
            'reasoning_text': cleaned_reasoning
        }

        if is_anxiety:
            label = sample.get('meme_anxiety_category')
            if label is None: logger.warning(f"Anxiety sample {sample_id} missing label."); continue
            if label == 'Irritatbily': label = 'Irritability'
            elif label == 'Unknown': label = 'Unknown Anxiety'
            sample_out['original_labels'] = label
            sample_out['stratify_label'] = label
        else:
            labels = sample.get('meme_depressive_categories')
            if labels is None: logger.warning(f"Depression sample {sample_id} missing labels."); continue
            if isinstance(labels, str):
                processed_labels = [lbl.strip() for lbl in labels.split(',') if lbl.strip()]
            elif isinstance(labels, list):
                processed_labels = [str(lbl).strip() for lbl in labels if str(lbl).strip()]
            else: processed_labels = []

            processed_labels = [lbl if lbl != 'Unknown' else 'Unknown Depression' for lbl in processed_labels]
            if not processed_labels: processed_labels = ["Unknown Depression"]

            sample_out['original_labels'] = processed_labels
            sample_out['stratify_label'] = processed_labels[0]

        filtered_data.append(sample_out)

    if filtered_data:
        example = filtered_data[0]
        logger.info(f"Example processed data sample_id: {example['sample_id']}")
        logger.info(f"Combined text sample: {example['text'][:150]}...")

    logger.info(f"Processed {len(filtered_data)} valid samples.")
    return filtered_data

def split_data(data: List[Dict], val_size: float, test_size: float, random_state: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if not data: return [], [], []
    n_samples = len(data)
    try: stratify_labels = [d['stratify_label'] for d in data]
    except KeyError: logger.error("Missing 'stratify_label'. Cannot stratify."); return data, [], []

    unique_labels, counts = np.unique(stratify_labels, return_counts=True)
    labels_with_one_sample = unique_labels[counts == 1]
    if len(labels_with_one_sample) > 0: logger.warning(f"Labels with 1 sample found: {list(labels_with_one_sample)}. Stratification might fail.")

    train_data, val_data, test_data = [], [], []

    if test_size > 0:
        try:
            train_val_indices, test_indices = train_test_split(range(n_samples), test_size=test_size, random_state=random_state, stratify=stratify_labels)
            train_val_data = [data[i] for i in train_val_indices]; test_data = [data[i] for i in test_indices]
            train_val_labels = [stratify_labels[i] for i in train_val_indices]
        except ValueError as e:
            logger.warning(f"Stratified test split failed: {e}. Non-stratified fallback."); train_val_indices, test_indices = train_test_split(range(n_samples), test_size=test_size, random_state=random_state)
            train_val_data = [data[i] for i in train_val_indices]; test_data = [data[i] for i in test_indices]; train_val_labels = [stratify_labels[i] for i in train_val_indices]
    else:
        train_val_data = data; train_val_labels = stratify_labels; test_data = []

    if val_size > 0 and len(train_val_data) > 0:
        relative_val_size = val_size / (1.0 - test_size) if test_size > 0 else val_size
        if relative_val_size <= 0 or relative_val_size >= 1 or len(train_val_data) < 2:
            logger.warning(f"Invalid relative val size or insufficient data ({len(train_val_data)}). No validation split."); train_data = train_val_data; val_data = []
        else:
            try:
                train_indices, val_indices = train_test_split(range(len(train_val_data)), test_size=relative_val_size, random_state=random_state, stratify=train_val_labels)
                train_data = [train_val_data[i] for i in train_indices]; val_data = [train_val_data[i] for i in val_indices]
            except ValueError as e:
                logger.warning(f"Stratified val split failed: {e}. Non-stratified fallback."); train_indices, val_indices = train_test_split(range(len(train_val_data)), test_size=relative_val_size, random_state=random_state)
                train_data = [train_val_data[i] for i in train_indices]; val_data = [train_val_data[i] for i in val_indices]
    else:
        train_data = train_val_data; val_data = []

    logger.info(f"Final split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

def load_feature_file(path, description):
    features = {}
    if path and os.path.exists(path):
        try:
            features = torch.load(path, map_location='cpu')
            logger.info(f"Loaded {len(features)} {description} features from {os.path.basename(path)}.")
        except Exception as e: logger.error(f"Error loading {description} features from {path}: {e}", exc_info=True)
    elif path: logger.warning(f"{description.capitalize()} feature file not found: {path}")
    return features

def tokenize_text(text: str, tokenizer: BartTokenizer, max_len: int):
    encoding = tokenizer.encode_plus(
        text, add_special_tokens=True, truncation=True, max_length=max_len,
        padding='max_length', return_attention_mask=True, return_tensors='pt'
    )
    return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

class SimpleOutput:
    """Simple container for model outputs with optional loss and logits attributes."""
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

class KnowledgeRetriever:
    """
    Simple wrapper for a FAISS-based vector DB for retrieval-augmented knowledge fusion.
    """
    def __init__(self, embedding_dim, db_path=None):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.sample_id_map = []
        self.text_map = []
        if db_path and os.path.exists(db_path):
            self.load(db_path)

    def build(self, embeddings, sample_ids, texts):
        self.index.add(embeddings.astype('float32'))
        self.sample_id_map = list(sample_ids)
        self.text_map = list(texts)

    def query(self, embedding, top_k=3, exclude_id=None):
        embedding = embedding.astype('float32').reshape(1, -1)
        D, I = self.index.search(embedding, top_k + 1)
        results = []
        for idx in I[0]:
            if idx < 0 or (exclude_id is not None and self.sample_id_map[idx] == exclude_id):
                continue
            results.append({
                'sample_id': self.sample_id_map[idx],
                'text': self.text_map[idx],
                'embedding': self.index.reconstruct(idx)
            })
            if len(results) == top_k:
                break
        return results

    def save(self, db_path):
        faiss.write_index(self.index, db_path + ".faiss")
        with open(db_path + ".meta.pkl", "wb") as f:
            pickle.dump({'sample_id_map': self.sample_id_map, 'text_map': self.text_map}, f)

    def load(self, db_path):
        self.index = faiss.read_index(db_path + ".faiss")
        with open(db_path + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)
            self.sample_id_map = meta['sample_id_map']
            self.text_map = meta['text_map']

class HybridCrossAttentionClassifier(nn.Module):
    def __init__(self,
                 num_labels: int,
                 base_model_name: str,
                 is_multilabel: bool,
                 visual_feature_dim: int,
                 max_regions: int,
                 dropout_prob: float,
                 knowledge_dim: int = None,
                 top_k_knowledge: int = 3):
        super().__init__()
        self.num_labels = num_labels
        self.is_multilabel = is_multilabel

        logger.info(f"--- Initializing HybridCrossAttentionClassifier ---")
        logger.info(f"  Base Model: {base_model_name} (MentalBART)")
        logger.info(f"  Task: {'Multi' if is_multilabel else 'Single'}-Label ({num_labels} classes)")
        logger.info(f"  Visual Dim: {visual_feature_dim}, Max Regions: {max_regions}")

        try:
            # Load MentalBART model
            self.base_model = BartModel.from_pretrained(base_model_name)
            self.text_embeddings = self.base_model.shared
            self.hidden_dim = self.base_model.config.hidden_size
            
            # VL Stream - Visual projection
            if visual_feature_dim != self.hidden_dim:
                self.visual_projection = nn.Linear(visual_feature_dim, self.hidden_dim)
            else:
                self.visual_projection = nn.Identity()

            # Spatial position encoding for region features
            self.box_projection = nn.Linear(4, self.hidden_dim)
            
            # Modality segment embeddings (0=text, 1=visual, 2=reasoning)
            self.segment_embeddings = nn.Embedding(3, self.hidden_dim)
            
            # VL stream encoder (MentalBART encoder)
            self.vl_encoder = self.base_model.encoder
            
            # Context stream for figurative reasoning
            self.context_encoder = BartModel.from_pretrained(base_model_name).encoder
            
            # Enhanced cross-modal attention modules
            self.text_to_visual_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout_prob
            )
            
            self.visual_to_text_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout_prob
            )
            
            # NEW: OCR-to-reasoning cross-attention
            self.ocr_to_reasoning_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout_prob
            )
            
            # NEW: Reasoning-to-OCR cross-attention
            self.reasoning_to_ocr_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout_prob
            )
            
            # NEW: Trimodal attention for joint reasoning across modalities
            self.trimodal_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=12,  # More heads for complex relationships
                dropout=dropout_prob
            )
            
            # Layer normalization for cross-attended features
            self.text_layer_norm = nn.LayerNorm(self.hidden_dim)
            self.visual_layer_norm = nn.LayerNorm(self.hidden_dim)
            self.reasoning_layer_norm = nn.LayerNorm(self.hidden_dim)
            self.trimodal_layer_norm = nn.LayerNorm(self.hidden_dim)
            
            # NEW: Gated multimodal fusion units for better information flow
            self.gated_text_visual_fusion = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid()
            )
            
            self.gated_text_reasoning_fusion = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid()
            )
            
            # Knowledge dimension setup
            self.knowledge_dim = knowledge_dim or self.hidden_dim
            self.top_k_knowledge = top_k_knowledge
            
            # NEW: Enhanced fusion layer using gated attention
            self.fusion_proj = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.Sigmoid()
            )
            
            # NEW: Knowledge attention layer
            self.knowledge_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=4,
                dropout=dropout_prob
            )
            
            # Final fusion layer combining all streams with knowledge
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.hidden_dim * 2 + self.knowledge_dim * self.top_k_knowledge, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),  # Changed from ReLU to GELU for better gradient flow
                nn.Dropout(dropout_prob)
            )
            
            # Classification MLP with a more gradual reduction in dimensions
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout_prob / 2),  # Reduce dropout in deeper layers
                nn.Linear(self.hidden_dim // 4, num_labels)
            )

        except Exception as e:
            logger.error(f"Model Init Error: {e}", exc_info=True)
            raise

        # Loss function based on task type
        if self.is_multilabel:
            self.loss_fct = nn.BCEWithLogitsLoss()
            logger.info("  Using BCEWithLogitsLoss for multi-label classification")
        else:
            self.loss_fct = nn.CrossEntropyLoss()
            logger.info("  Using CrossEntropyLoss for single-label classification")
        
        logger.info("--- HybridCrossAttentionClassifier Initialized ---")

    def forward(self,
                input_ids, attention_mask,
                visual_features, visual_boxes, visual_attention_mask,
                context_input_ids=None, context_attention_mask=None,
                knowledge_embeddings=None,
                labels=None):

        batch_size = input_ids.shape[0]
        text_seq_len = input_ids.shape[1]
        vis_seq_len = visual_features.shape[1]

        try:
            # Process text input (OCR)
            text_embeds = self.text_embeddings(input_ids)
            text_segment_ids = torch.zeros(batch_size, text_seq_len, dtype=torch.long, device=input_ids.device)
            text_segment_embeds = self.segment_embeddings(text_segment_ids)
            text_input_embeds = text_embeds + text_segment_embeds
            
            # Process visual input (region features)
            projected_vis_embeds = self.visual_projection(visual_features)
            box_embeds = self.box_projection(visual_boxes)
            visual_embeds = projected_vis_embeds + box_embeds
            visual_segment_ids = torch.ones(batch_size, vis_seq_len, dtype=torch.long, device=input_ids.device)
            visual_segment_embeds = self.segment_embeddings(visual_segment_ids)
            visual_input_embeds = visual_embeds + visual_segment_embeds
            
            # Process reasoning/context input
            has_context = context_input_ids is not None and context_attention_mask is not None
            if has_context:
                context_embeds = self.text_embeddings(context_input_ids)
                context_seq_len = context_input_ids.shape[1]
                # Use segment ID 2 for reasoning text
                context_segment_ids = torch.full((batch_size, context_seq_len), 2, 
                                               dtype=torch.long, device=input_ids.device)
                context_segment_embeds = self.segment_embeddings(context_segment_ids)
                context_input_embeds = context_embeds + context_segment_embeds
                
                # Process context through its encoder
                context_outputs = self.context_encoder(
                    inputs_embeds=context_input_embeds,
                    attention_mask=context_attention_mask,
                    return_dict=True
                )
                context_states = context_outputs.last_hidden_state
            else:
                context_states = None
                context_attention_mask = None
            
            # 1. VL Stream Processing - Concatenate text and visual embeddings
            combined_embeds = torch.cat([text_input_embeds, visual_input_embeds], dim=1)
            combined_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            
            vl_outputs = self.vl_encoder(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                return_dict=True
            )
            vl_hidden_states = vl_outputs.last_hidden_state
            
            # 2. Apply enhanced bidirectional cross-attention
            text_states = vl_hidden_states[:, :text_seq_len]
            visual_states = vl_hidden_states[:, text_seq_len:]
            
            # Text -> Visual attention
            text_to_visual, _ = self.text_to_visual_attention(
                query=text_states.transpose(0, 1),
                key=visual_states.transpose(0, 1),
                value=visual_states.transpose(0, 1),
                key_padding_mask=~visual_attention_mask.bool()
            )
            text_to_visual = text_to_visual.transpose(0, 1)
            
            # Visual -> Text attention
            visual_to_text, _ = self.visual_to_text_attention(
                query=visual_states.transpose(0, 1),
                key=text_states.transpose(0, 1),
                value=text_states.transpose(0, 1),
                key_padding_mask=~attention_mask.bool()
            )
            visual_to_text = visual_to_text.transpose(0, 1)
            
            # 3. Apply OCR-Reasoning cross-attention if context is available
            if has_context:
                # OCR -> Reasoning attention
                ocr_to_reasoning, _ = self.ocr_to_reasoning_attention(
                    query=text_states.transpose(0, 1),
                    key=context_states.transpose(0, 1),
                    value=context_states.transpose(0, 1),
                    key_padding_mask=~context_attention_mask.bool()
                )
                ocr_to_reasoning = ocr_to_reasoning.transpose(0, 1)
                
                # Reasoning -> OCR attention
                reasoning_to_ocr, _ = self.reasoning_to_ocr_attention(
                    query=context_states.transpose(0, 1),
                    key=text_states.transpose(0, 1),
                    value=text_states.transpose(0, 1),
                    key_padding_mask=~attention_mask.bool()
                )
                reasoning_to_ocr = reasoning_to_ocr.transpose(0, 1)
                
                # Apply layer normalization to cross-attended reasoning
                context_states = self.reasoning_layer_norm(context_states + reasoning_to_ocr)
            
            # 4. Combine and normalize the cross-attended features
            cross_attended_text = self.text_layer_norm(text_states + text_to_visual)
            if has_context:
                cross_attended_text = self.text_layer_norm(cross_attended_text + ocr_to_reasoning)
                
            cross_attended_visual = self.visual_layer_norm(visual_states + visual_to_text)
            
            # 5. NEW: Apply trimodal attention if context is available
            if has_context:
                # Concatenate all modalities for joint processing
                all_modalities = torch.cat([
                    cross_attended_text, 
                    cross_attended_visual, 
                    context_states
                ], dim=1)
                
                # Create combined attention mask
                all_attention_mask = torch.cat([
                    attention_mask,
                    visual_attention_mask,
                    context_attention_mask
                ], dim=1)
                
                # Apply self-attention across all modalities jointly
                all_modalities_T = all_modalities.transpose(0, 1)
                trimodal_attended, _ = self.trimodal_attention(
                    query=all_modalities_T,
                    key=all_modalities_T,
                    value=all_modalities_T,
                    key_padding_mask=~all_attention_mask.bool()
                )
                trimodal_attended = trimodal_attended.transpose(0, 1)
                
                # Split back to individual modalities
                total_len = trimodal_attended.size(1)
                text_end = text_seq_len
                visual_end = text_end + vis_seq_len
                
                tri_text = trimodal_attended[:, :text_end]
                tri_visual = trimodal_attended[:, text_end:visual_end]
                tri_context = trimodal_attended[:, visual_end:]
                
                # Layer normalization
                tri_text = self.trimodal_layer_norm(tri_text)
                tri_visual = self.trimodal_layer_norm(tri_visual)
                tri_context = self.trimodal_layer_norm(tri_context)
                
                # Get [CLS] representations from each modality
                text_cls = tri_text[:, 0]
                visual_cls = torch.mean(tri_visual, dim=1)  # Average pool visual features
                context_cls = tri_context[:, 0]
                
                # Gated fusion of modalities
                combined_features = torch.cat([text_cls, visual_cls, context_cls], dim=1)
                fusion_projection = self.fusion_proj(combined_features)
                fusion_gate = self.fusion_gate(combined_features)
                
                # Apply gated fusion
                multimodal_representation = fusion_projection * fusion_gate
            else:
                # Without context, use simpler fusion of text and visual
                text_cls = cross_attended_text[:, 0]
                visual_cls = torch.mean(cross_attended_visual, dim=1)
                
                # Gated fusion between text and visual only
                text_visual_combined = torch.cat([text_cls, visual_cls], dim=1)
                multimodal_representation = torch.cat([text_cls, visual_cls], dim=1)
                multimodal_representation = text_cls + visual_cls  # Simple addition
            
            # 6. Process knowledge if available
            if knowledge_embeddings is not None:
                knowledge_flat = knowledge_embeddings.view(batch_size, self.top_k_knowledge, -1)
                
                # Create a query from our multimodal representation
                query = multimodal_representation.unsqueeze(0)  # [1, B, H]
                
                # Reshape knowledge for attention
                knowledge_for_attn = knowledge_flat.transpose(0, 1)  # [K, B, H]
                
                # Apply attention over knowledge items
                attended_knowledge, attention_weights = self.knowledge_attention(
                    query=query,
                    key=knowledge_for_attn,
                    value=knowledge_for_attn
                )
                
                attended_knowledge = attended_knowledge.squeeze(0)  # [B, H]
                
                # Combine with multimodal representation
                fusion_inputs = [multimodal_representation, attended_knowledge, 
                                 knowledge_embeddings.view(batch_size, -1)]
            else:
                fusion_inputs = [
                    multimodal_representation, 
                    torch.zeros_like(multimodal_representation),
                    torch.zeros((batch_size, self.knowledge_dim * self.top_k_knowledge), 
                                device=input_ids.device)
                ]
            
            # Final fusion of all inputs
            fused_representation = self.fusion_layer(torch.cat(fusion_inputs, dim=1))

            # 7. Classification
            logits = self.classifier(fused_representation)

        except Exception as e:
            logger.error(f"Forward pass error: {e}", exc_info=True)
            fused_representation = torch.zeros(batch_size, self.hidden_dim, device=input_ids.device)
            logits = torch.zeros(batch_size, self.num_labels, device=input_ids.device)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            try:
                if self.is_multilabel:
                    # Multi-label classification (BCEWithLogitsLoss)
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # Single-label classification (CrossEntropyLoss)
                    loss = self.loss_fct(logits, labels)
            except Exception as loss_e:
                logger.error(f"Loss calculation error: {loss_e}", exc_info=True)

        return SimpleOutput(loss=loss, logits=logits)

class MemeDataset(Dataset):
    def __init__(self, data_samples, tokenizer, image_features_map, label_encoder, 
                 max_text_len=MAX_TEXT_LEN, max_regions=MAX_REGIONS, 
                 is_multilabel=False, knowledge_retriever=None, encoder_model=None, device=None, top_k_knowledge=3):
        self.data = data_samples
        self.tokenizer = tokenizer
        self.image_features_map = image_features_map
        self.label_encoder = label_encoder
        self.max_text_len = max_text_len
        self.max_regions = max_regions
        self.is_multilabel = is_multilabel
        self.default_box_feature = np.array([0, 0, 1, 1])  # Default box coordinates [x1, y1, x2, y2]
        self.knowledge_retriever = knowledge_retriever
        self.encoder_model = encoder_model  # Should be a torch model (e.g., BartModel)
        self.device = device
        self.top_k_knowledge = top_k_knowledge
        self.embedding_dim = encoder_model.config.hidden_size if encoder_model else 768
        
    def __len__(self):
        return len(self.data)
        
    def get_text_embedding(self, text):
        # Use encoder_model to get embedding for text
        if not self.encoder_model or not text:
            return np.zeros(self.embedding_dim)
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_text_len)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.encoder_model.encoder(**inputs, return_dict=True)
            emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().squeeze(0)
        return emb

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Process text (combined OCR and reasoning)
        text = sample.get('text', '')
        input_ids, attention_mask = tokenize_text(text, self.tokenizer, self.max_text_len)
        
        # For context input, we can use the reasoning part separately if needed
        # This allows the model to process the structured reasoning in the context stream
        context_text = sample.get('reasoning_text', '')
        if context_text:
            context_input_ids, context_attention_mask = tokenize_text(
                context_text, self.tokenizer, self.max_text_len
            )
        else:
            # Create empty tensors if no context available
            context_input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            context_attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)
        
        # Process image features
        image_id = sample.get('image_id', sample.get('sample_id', ''))
        if image_id in self.image_features_map:
            img_features = self.image_features_map[image_id]['features']
            img_boxes = self.image_features_map[image_id]['boxes']
        else:
            # If no image features found, use zeros
            img_features = np.zeros((self.max_regions, 2048))
            img_boxes = np.tile(self.default_box_feature, (self.max_regions, 1))
            
        # Ensure we have the correct number of regions
        if len(img_features) > self.max_regions:
            img_features = img_features[:self.max_regions]
            img_boxes = img_boxes[:self.max_regions]
        elif len(img_features) < self.max_regions:
            padding_size = self.max_regions - len(img_features)
            feature_padding = np.zeros((padding_size, img_features.shape[1]))
            box_padding = np.tile(self.default_box_feature, (padding_size, 1))
            img_features = np.vstack([img_features, feature_padding])
            img_boxes = np.vstack([img_boxes, box_padding])
            
        # Create visual attention mask (1 for actual regions, 0 for padding)
        visual_attention_mask = np.zeros(self.max_regions)
        visual_attention_mask[:min(len(self.image_features_map.get(image_id, {}).get('features', [])), self.max_regions)] = 1
        
        # Knowledge retrieval
        knowledge_embeddings = np.zeros((self.top_k_knowledge, self.embedding_dim))
        if self.knowledge_retriever and self.encoder_model:
            query_text = sample.get('text', '')
            query_emb = self.get_text_embedding(query_text)
            retrieved = self.knowledge_retriever.query(query_emb, top_k=self.top_k_knowledge, exclude_id=sample.get('sample_id', ''))
            for i, r in enumerate(retrieved):
                knowledge_embeddings[i] = r['embedding']
        
        # Process labels
        if self.is_multilabel:
            labels = sample.get('original_labels', [])
            label_vector = np.zeros(len(self.label_encoder.classes_))
            if labels:
                indices = self.label_encoder.transform([l for l in labels if l in self.label_encoder.classes_])
                label_vector[indices] = 1
        else:
            label = sample.get('original_labels', None)
            if label in self.label_encoder.classes_:
                label_idx = self.label_encoder.transform([label])[0]
            else:
                # If label not in our classes, use a default
                label_idx = 0
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'visual_features': torch.tensor(img_features, dtype=torch.float),
            'visual_boxes': torch.tensor(img_boxes, dtype=torch.float),
            'visual_attention_mask': torch.tensor(visual_attention_mask, dtype=torch.long),
            'knowledge_embeddings': torch.tensor(knowledge_embeddings, dtype=torch.float),
            'labels': torch.tensor(label_vector if self.is_multilabel else label_idx),
            'sample_id': sample.get('sample_id', '')
        }

def calculate_metrics(y_true, y_pred, is_multilabel):
    """Calculate multiple metrics for evaluation."""
    if is_multilabel:
        # For multi-label classification
        preds_binary = (y_pred >= 0.5).astype(float)
        f1_macro = f1_score(y_true, preds_binary, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, preds_binary, average='weighted', zero_division=0)
        hl = hamming_loss(y_true, preds_binary)
    else:
        # For single-label classification
        preds_classes = np.argmax(y_pred, axis=1)
        f1_macro = f1_score(y_true, preds_classes, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, preds_classes, average='weighted', zero_division=0)
        # For hamming loss in single-label case, convert to one-hot
        n_classes = y_pred.shape[1]
        true_one_hot = np.eye(n_classes)[y_true]
        pred_one_hot = np.eye(n_classes)[preds_classes]
        hl = hamming_loss(true_one_hot, pred_one_hot)
    
    return {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'hamming_loss': hl
    }

def log_metrics(epoch, metrics_dict, prefix=""):
    """Log metrics with clear formatting."""
    logger.info(f"{prefix} Epoch {epoch}:")
    for metric_name, value in metrics_dict.items():
        logger.info(f"  {metric_name}: {value:.4f}")

class MetricsTracker:
    """Helper class to track and update metrics during training."""
    def __init__(self, is_multilabel):
        self.is_multilabel = is_multilabel
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1_macro': [], 'val_f1_macro': [],
            'train_f1_weighted': [], 'val_f1_weighted': [],
            'train_hamming_loss': [], 'val_hamming_loss': []
        }
        self.current_epoch_metrics = {}
        
    def update_train_metrics(self, loss, y_true, y_pred):
        """Update metrics for training data"""
        metrics = calculate_metrics(y_true, y_pred, self.is_multilabel)
        self.current_epoch_metrics['train_loss'] = loss
        self.current_epoch_metrics['train_f1_macro'] = metrics['f1_macro']
        self.current_epoch_metrics['train_f1_weighted'] = metrics['f1_weighted']
        self.current_epoch_metrics['train_hamming_loss'] = metrics['hamming_loss']
        
    def update_val_metrics(self, loss, y_true, y_pred):
        """Update metrics for validation data"""
        metrics = calculate_metrics(y_true, y_pred, self.is_multilabel)
        self.current_epoch_metrics['val_loss'] = loss
        self.current_epoch_metrics['val_f1_macro'] = metrics['f1_macro']
        self.current_epoch_metrics['val_f1_weighted'] = metrics['f1_weighted']
        self.current_epoch_metrics['val_hamming_loss'] = metrics['hamming_loss']
    
    def end_epoch(self):
        """Save current metrics to history and reset current metrics."""
        for key, value in self.current_epoch_metrics.items():
            self.history[key].append(value)
        self.current_epoch_metrics = {}
        
    def log_current_metrics(self, epoch):
        """Log current epoch metrics."""
        # Format train metrics
        train_metrics = {k.replace('train_', ''): v for k, v in self.current_epoch_metrics.items() 
                         if k.startswith('train_')}
        log_metrics(epoch, train_metrics, prefix="TRAIN")
        
        # Format validation metrics if available
        val_metrics = {k.replace('val_', ''): v for k, v in self.current_epoch_metrics.items() 
                       if k.startswith('val_')}
        if val_metrics:
            log_metrics(epoch, val_metrics, prefix="VALIDATION")

def plot_training_metrics(metrics_history, output_path, model_name):
    """Create plots for all tracked metrics over epochs."""
    epochs = list(range(1, len(metrics_history['train_loss']) + 1))
    
    plt.figure(figsize=(20, 15))
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in metrics_history and metrics_history['val_loss']:
        plt.plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 Macro
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics_history['train_f1_macro'], 'b-', label='Training F1 Macro')
    if 'val_f1_macro' in metrics_history and metrics_history['val_f1_macro']:
        plt.plot(epochs, metrics_history['val_f1_macro'], 'r-', label='Validation F1 Macro')
    plt.title('F1 Macro Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 Weighted
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics_history['train_f1_weighted'], 'b-', label='Training F1 Weighted')
    if 'val_f1_weighted' in metrics_history and metrics_history['val_f1_weighted']:
        plt.plot(epochs, metrics_history['val_f1_weighted'], 'r-', label='Validation F1 Weighted')
    plt.title('F1 Weighted Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot Hamming Loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics_history['train_hamming_loss'], 'b-', label='Training Hamming Loss')
    if 'val_hamming_loss' in metrics_history and metrics_history['val_hamming_loss']:
        plt.plot(epochs, metrics_history['val_hamming_loss'], 'r-', label='Validation Hamming Loss')
    plt.title('Hamming Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Hamming Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    file_path = os.path.join(output_path, f"{model_name}_metrics.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Metrics plot saved to {file_path}")

def generate_classification_metrics(y_true, y_pred, is_multilabel, label_encoder=None, phase="Evaluation"):
    """
    Generate and format classification report, accuracy score, and multilabel confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        is_multilabel: Whether this is a multilabel classification task
        label_encoder: LabelEncoder with classes_ attribute for class names
        phase: String descriptor for the evaluation phase (e.g., "Training", "Validation")
        
    Returns:
        dict: Dictionary containing formatted metrics
    """
    target_names = None
    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
        target_names = label_encoder.classes_
        
    results = {
        'phase': phase,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if is_multilabel:
        # Convert probabilities to binary predictions
        preds_binary = (y_pred >= 0.5).astype(int)
        
        # Calculate accuracy - exact match for all labels
        results['accuracy'] = accuracy_score(y_true, preds_binary)
        
        # Generate classification report
        try:
            report = classification_report(
                y_true, preds_binary, 
                target_names=target_names,
                zero_division=0,
                output_dict=True
            )
            results['classification_report'] = report
            results['classification_report_text'] = classification_report(
                y_true, preds_binary,
                target_names=target_names,
                zero_division=0
            )
        except Exception as e:
            logger.warning(f"Error generating classification report: {e}")
            results['classification_report'] = None
            results['classification_report_text'] = f"Error: {e}"
        
        # Generate multilabel confusion matrix
        try:
            mcm = multilabel_confusion_matrix(y_true, preds_binary)
            results['confusion_matrix'] = mcm.tolist()  # Convert to list for JSON serialization
            
            # Create a text representation of confusion matrices
            cm_text = []
            for i, cm in enumerate(mcm):
                label_name = target_names[i] if target_names is not None else f"Label {i}"
                cm_text.append(f"Confusion Matrix for {label_name}:\n{cm}")
            results['confusion_matrix_text'] = "\n\n".join(cm_text)
        except Exception as e:
            logger.warning(f"Error generating confusion matrix: {e}")
            results['confusion_matrix'] = None
            results['confusion_matrix_text'] = f"Error: {e}"
            
    else:
        # Single-label classification
        preds_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        results['accuracy'] = accuracy_score(y_true, preds_classes)
        
        # Generate classification report
        try:
            report = classification_report(
                y_true, preds_classes, 
                target_names=target_names,
                zero_division=0,
                output_dict=True
            )
            results['classification_report'] = report
            results['classification_report_text'] = classification_report(
                y_true, preds_classes,
                target_names=target_names,
                zero_division=0
            )
        except Exception as e:
            logger.warning(f"Error generating classification report: {e}")
            results['classification_report'] = None
            results['classification_report_text'] = f"Error: {e}"
        
        # Generate multilabel confusion matrix (for single-label, convert to one-hot first)
        try:
            n_classes = len(target_names) if target_names is not None else y_pred.shape[1]
            true_one_hot = np.eye(n_classes)[y_true]
            pred_one_hot = np.eye(n_classes)[preds_classes]
            mcm = multilabel_confusion_matrix(true_one_hot, pred_one_hot)
            results['confusion_matrix'] = mcm.tolist()
            
            # Create a text representation of confusion matrices
            cm_text = []
            for i, cm in enumerate(mcm):
                label_name = target_names[i] if target_names is not None else f"Class {i}"
                cm_text.append(f"Confusion Matrix for {label_name}:\n{cm}")
            results['confusion_matrix_text'] = "\n\n".join(cm_text)
        except Exception as e:
            logger.warning(f"Error generating confusion matrix: {e}")
            results['confusion_matrix'] = None
            results['confusion_matrix_text'] = f"Error: {e}"

    return results

def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs, device, 
                is_multilabel, output_dir, model_name):
    """
    Train model with comprehensive metrics tracking and logging.
    """
    metrics_tracker = MetricsTracker(is_multilabel)
    best_val_f1 = 0.0
    
    logger.info(f"Starting training for {model_name} ({num_epochs} epochs)")
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual_features = batch['visual_features'].to(device)
            visual_boxes = batch['visual_boxes'].to(device)
            visual_attention_mask = batch['visual_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Include context inputs for the figurative reasoning stream
            context_input_ids = batch['context_input_ids'].to(device)
            context_attention_mask = batch['context_attention_mask'].to(device)
            
            # Include knowledge embeddings
            knowledge_embeddings = batch.get('knowledge_embeddings', None)
            if knowledge_embeddings is not None:
                knowledge_embeddings = knowledge_embeddings.to(device)
            
            # Forward pass with context inputs
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                visual_features=visual_features, 
                visual_boxes=visual_boxes, 
                visual_attention_mask=visual_attention_mask,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                knowledge_embeddings=knowledge_embeddings,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Collect metrics
            train_losses.append(loss.item())
            
            # Move predictions and labels to CPU for metric calculation
            if is_multilabel:
                preds = torch.sigmoid(logits).detach().cpu().numpy()
            else:
                preds = F.softmax(logits, dim=1).detach().cpu().numpy()
                
            all_train_preds.append(preds)
            all_train_labels.append(labels.detach().cpu().numpy())
        
        # Concatenate all batches
        all_train_preds = np.vstack(all_train_preds)
        all_train_labels = np.vstack(all_train_labels) if is_multilabel else np.concatenate(all_train_labels)
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Update train metrics
        metrics_tracker.update_train_metrics(avg_train_loss, all_train_labels, all_train_preds)
        
        # Generate and log detailed classification metrics for training
        train_metrics = generate_classification_metrics(
            all_train_labels, all_train_preds, 
            is_multilabel=is_multilabel,
            label_encoder=train_loader.dataset.label_encoder, 
            phase=f"Training Epoch {epoch}"
        )
        
        # Log training classification report
        logger.info(f"Training Classification Report (Epoch {epoch}):\n{train_metrics['classification_report_text']}")
        logger.info(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Save training metrics
        train_metrics_path = os.path.join(output_dir, f"{model_name}_epoch_{epoch}_train_metrics.json")
        with open(train_metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)
        
        # Validation phase
        if val_loader:
            model.eval()
            val_losses = []
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    visual_features = batch['visual_features'].to(device)
                    visual_boxes = batch['visual_boxes'].to(device)
                    visual_attention_mask = batch['visual_attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Include context inputs for the figurative reasoning stream
                    context_input_ids = batch['context_input_ids'].to(device)
                    context_attention_mask = batch['context_attention_mask'].to(device)
                    
                    # Include knowledge embeddings
                    knowledge_embeddings = batch.get('knowledge_embeddings', None)
                    if knowledge_embeddings is not None:
                        knowledge_embeddings = knowledge_embeddings.to(device)
                    
                    # Forward pass with context inputs
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        visual_features=visual_features, 
                        visual_boxes=visual_boxes, 
                        visual_attention_mask=visual_attention_mask,
                        context_input_ids=context_input_ids,
                        context_attention_mask=context_attention_mask,
                        knowledge_embeddings=knowledge_embeddings,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    # Collect metrics
                    val_losses.append(loss.item())
                    
                    # Move predictions and labels to CPU
                    if is_multilabel:
                        preds = torch.sigmoid(logits).detach().cpu().numpy()
                    else:
                        preds = F.softmax(logits, dim=1).detach().cpu().numpy()
                        
                    all_val_preds.append(preds)
                    all_val_labels.append(labels.detach().cpu().numpy())
            
            # Concatenate all batches
            all_val_preds = np.vstack(all_val_preds)
            all_val_labels = np.vstack(all_val_labels) if is_multilabel else np.concatenate(all_val_labels)
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            # Update validation metrics
            metrics_tracker.update_val_metrics(avg_val_loss, all_val_labels, all_val_preds)
            
            # Generate and log detailed classification metrics for validation
            val_metrics = generate_classification_metrics(
                all_val_labels, all_val_preds, 
                is_multilabel=is_multilabel,
                label_encoder=val_loader.dataset.label_encoder, 
                phase=f"Validation Epoch {epoch}"
            )
            
            # Log validation classification report
            logger.info(f"Validation Classification Report (Epoch {epoch}):\n{val_metrics['classification_report_text']}")
            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save validation metrics
            val_metrics_path = os.path.join(output_dir, f"{model_name}_epoch_{epoch}_val_metrics.json")
            with open(val_metrics_path, 'w') as f:
                json.dump(val_metrics, f, indent=2)
            
            # Check if this is the best model so far
            current_val_f1 = metrics_tracker.current_epoch_metrics['val_f1_weighted']
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                model_path = os.path.join(output_dir, f"{model_name}_best.pt")
                torch.save(model.state_dict(), model_path)
                logger.info(f"New best model saved with F1 weighted: {best_val_f1:.4f}")
        
        # Log metrics for this epoch
        metrics_tracker.log_current_metrics(epoch)
        metrics_tracker.end_epoch()
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(output_dir, f"{model_name}_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics_tracker.history
        }, checkpoint_path)
    
    # Create and save plots
    plot_training_metrics(metrics_tracker.history, output_dir, model_name)
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    
    # Save metrics history as JSON
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics_history.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_tracker.history, f, indent=2)
    
    return model, metrics_tracker.history

def evaluate_model(model, test_loader, device, is_multilabel, output_dir, model_name, label_encoder=None):
    """
    Evaluate a trained model on test data and generate comprehensive metrics.
    """
    model.eval()
    all_test_preds = []
    all_test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual_features = batch['visual_features'].to(device)
            visual_boxes = batch['visual_boxes'].to(device)
            visual_attention_mask = batch['visual_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Include context inputs for the figurative reasoning stream
            context_input_ids = batch['context_input_ids'].to(device)
            context_attention_mask = batch['context_attention_mask'].to(device)
            
            # Include knowledge embeddings
            knowledge_embeddings = batch.get('knowledge_embeddings', None)
            if knowledge_embeddings is not None:
                knowledge_embeddings = knowledge_embeddings.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                visual_features=visual_features, 
                visual_boxes=visual_boxes, 
                visual_attention_mask=visual_attention_mask,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                knowledge_embeddings=knowledge_embeddings
            )
            
            logits = outputs.logits
            
            # Move predictions and labels to CPU
            if is_multilabel:
                preds = torch.sigmoid(logits).detach().cpu().numpy()
            else:
                preds = F.softmax(logits, dim=1).detach().cpu().numpy()
                
            all_test_preds.append(preds)
            all_test_labels.append(labels.detach().cpu().numpy())
    
    # Concatenate all batches
    all_test_preds = np.vstack(all_test_preds)
    all_test_labels = np.vstack(all_test_labels) if is_multilabel else np.concatenate(all_test_labels)
    
    # Generate detailed metrics
    test_metrics = generate_classification_metrics(
        all_test_labels, all_test_preds, 
        is_multilabel=is_multilabel,
        label_encoder=label_encoder or test_loader.dataset.label_encoder, 
        phase="Test Set Evaluation"
    )
    
    # Log test metrics
    logger.info(f"Test Classification Report:\n{test_metrics['classification_report_text']}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save test metrics
    test_metrics_path = os.path.join(output_dir, f"{model_name}_test_metrics.json")
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
        
    return test_metrics

class HybridModelPipeline:
    def __init__(self, dataset_type='depression', config=None):
        self.dataset_type = dataset_type
        self.is_anxiety = dataset_type.lower() == 'anxiety'
        self.config = config or {}
        
        # Set default configuration parameters
        self.set_default_config()
        
        # Initialize directories and paths
        self.initialize_paths()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.knowledge_retriever = None
        self.encoder_model = None
        
    def set_default_config(self):
        defaults = {
            'base_model_name': BASE_TRANSFORMER_NAME,
            'tokenizer_name': TOKENIZER_NAME,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'adam_beta1': ADAM_BETA1,
            'adam_beta2': ADAM_BETA2, 
            'adam_epsilon': ADAM_EPSILON,
            'weight_decay': WEIGHT_DECAY,
            'dropout': DROPOUT,
            'num_ensemble_models': NUM_ENSEMBLE_MODELS,
            'base_seed': BASE_SEED,
            'anxiety_val_split_ratio': ANXIETY_VAL_SPLIT_RATIO,
            'max_text_len': MAX_TEXT_LEN,
            'max_regions': MAX_REGIONS,
            'region_feature_dim': REGION_FEATURE_DIM
        }
        
        # Apply any custom config, using defaults for missing values
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def initialize_paths(self):
        # Create necessary directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        self.input_dir = os.path.join(base_dir, 'dataset')
        self.output_dir = os.path.join(base_dir, 'models', 'output')
        self.feature_dir = os.path.join(base_dir, 'models', 'region_visual_features')
        
        # Create output dir if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_data(self):
        """Load and process the dataset"""
        # Define file paths based on dataset choice
        if self.is_anxiety:
            data_file = os.path.join(self.input_dir, "anxiety_train.json")
            test_file = os.path.join(self.input_dir, "anxiety_test.json")
        else:
            data_file = os.path.join(self.input_dir, "depression_train.json")
            test_file = os.path.join(self.input_dir, "depression_test.json")
            
        # Load and process training data
        raw_data = load_json_data(data_file)
        if not raw_data:
            logger.error(f"Failed to load data from {data_file}. Exiting.")
            return False
            
        self.processed_data = process_loaded_data(raw_data, self.is_anxiety)
            
        # Load test data if available
        raw_test_data = load_json_data(test_file)
        self.test_data = None
        if raw_test_data:
            self.test_data = process_loaded_data(raw_test_data, self.is_anxiety)
        
        # Load image features
        image_features_file = os.path.join(self.feature_dir, "region_features.pt")
        self.image_features_map = load_feature_file(image_features_file, "region")
        
        return True
    
    def setup_label_encoding(self):
        """Create the label encoder based on dataset"""
        if self.is_anxiety:
            unique_labels = sorted(set([d['original_labels'] for d in self.processed_data if 'original_labels' in d]))
            self.is_multilabel = False
            logger.info(f"Found {len(unique_labels)} unique anxiety labels")
        else:
            # Flatten the list of lists
            all_labels = []
            for d in self.processed_data:
                if 'original_labels' in d and isinstance(d['original_labels'], list):
                    all_labels.extend(d['original_labels'])
            unique_labels = sorted(set(all_labels))
            self.is_multilabel = True
            logger.info(f"Found {len(unique_labels)} unique depression labels")
            
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(unique_labels)
        logger.info(f"Labels: {', '.join(self.label_encoder.classes_)}")
        self.num_labels = len(self.label_encoder.classes_)
        
    def split_dataset(self):
        """Split the data into train/val/test sets"""
        val_size = self.config['anxiety_val_split_ratio'] if self.is_anxiety else 0.15
        self.train_data, self.val_data, _ = split_data(
            self.processed_data, 
            val_size=val_size, 
            test_size=0, 
            random_state=self.config['base_seed']
        )
        
    def build_knowledge_db(self):
        """
        Build the vector DB for knowledge fusion using all training samples.
        """
        logger.info("Building knowledge fusion vector DB...")
        # Use the base BART encoder for embeddings
        self.encoder_model = BartModel.from_pretrained(self.config['base_model_name']).to(self.device)
        self.encoder_model.eval()
        embeddings = []
        sample_ids = []
        texts = []
        with torch.no_grad():
            for sample in self.processed_data:
                text = sample.get('text', '')
                if not text: continue
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.config['max_text_len'])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.encoder_model.encoder(**inputs, return_dict=True)
                emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().squeeze(0)
                embeddings.append(emb)
                sample_ids.append(sample.get('sample_id', ''))
                texts.append(text)
        embeddings = np.stack(embeddings)
        self.knowledge_retriever = KnowledgeRetriever(embedding_dim=embeddings.shape[1])
        self.knowledge_retriever.build(embeddings, sample_ids, texts)
        logger.info(f"Knowledge DB built with {len(sample_ids)} samples.")
        
    def prepare_dataset(self):
        """Create PyTorch datasets and dataloaders"""
        # Load tokenizer
        try:
            self.tokenizer = BartTokenizer.from_pretrained(self.config['tokenizer_name'])
            logger.info(f"Loaded tokenizer: {self.config['tokenizer_name']}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return False
            
        # Build knowledge DB and encoder model if not already done
        if self.knowledge_retriever is None or self.encoder_model is None:
            self.build_knowledge_db()
            
        # Create datasets
        self.train_dataset = MemeDataset(
            self.train_data, 
            self.tokenizer, 
            self.image_features_map, 
            self.label_encoder,
            max_text_len=self.config['max_text_len'],
            max_regions=self.config['max_regions'],
            is_multilabel=self.is_multilabel,
            knowledge_retriever=self.knowledge_retriever,
            encoder_model=self.encoder_model,
            device=self.device,
            top_k_knowledge=3
        )
        
        if self.val_data:
            self.val_dataset = MemeDataset(
                self.val_data, 
                self.tokenizer, 
                self.image_features_map, 
                self.label_encoder,
                max_text_len=self.config['max_text_len'],
                max_regions=self.config['max_regions'],
                is_multilabel=self.is_multilabel,
                knowledge_retriever=self.knowledge_retriever,
                encoder_model=self.encoder_model,
                device=self.device,
                top_k_knowledge=3
            )
        else:
            self.val_dataset = None
            
        if self.test_data:
            self.test_dataset = MemeDataset(
                self.test_data, 
                self.tokenizer, 
                self.image_features_map, 
                self.label_encoder,
                max_text_len=self.config['max_text_len'],
                max_regions=self.config['max_regions'],
                is_multilabel=self.is_multilabel,
                knowledge_retriever=self.knowledge_retriever,
                encoder_model=self.encoder_model,
                device=self.device,
                top_k_knowledge=3
            )
        else:
            self.test_dataset = None
            
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=2
            )
        else:
            self.val_loader = None
            
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=2
            )
        else:
            self.test_loader = None
            
        logger.info(f"Created dataloaders: Train={len(self.train_loader)}, "
                    f"Val={'None' if self.val_loader is None else len(self.val_loader)}, "
                    f"Test={'None' if self.test_loader is None else len(self.test_loader)}")
        return True
    
    def train_ensemble(self):
        """Train multiple models for ensemble"""
        self.ensemble_models = []
        
        for model_idx in range(self.config['num_ensemble_models']):
            logger.info(f"Training ensemble model {model_idx+1}/{self.config['num_ensemble_models']}")
            
            # Set seed for reproducibility, but different for each model
            model_seed = self.config['base_seed'] + model_idx
            set_seed(model_seed)
            
            # Initialize model
            model = HybridCrossAttentionClassifier(
                num_labels=self.num_labels,
                base_model_name=self.config['base_model_name'],
                is_multilabel=self.is_multilabel,
                visual_feature_dim=self.config['region_feature_dim'],
                max_regions=self.config['max_regions'],
                dropout_prob=self.config['dropout'],
                knowledge_dim=self.encoder_model.config.hidden_size,
                top_k_knowledge=3
            ).to(self.device)
            
            # Setup optimizer and scheduler
            optimizer = AdamW(
                model.parameters(),
                lr=self.config['learning_rate'],
                betas=(self.config['adam_beta1'], self.config['adam_beta2']),
                eps=self.config['adam_epsilon'],
                weight_decay=self.config['weight_decay']
            )
            
            total_steps = len(self.train_loader) * self.config['num_epochs']
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Set model name for this ensemble member
            model_name = f"{self.dataset_type}_hybrid_model_{model_idx+1}"
            
            # Train the model
            trained_model, metrics = train_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=self.config['num_epochs'],
                device=self.device,
                is_multilabel=self.is_multilabel,
                output_dir=self.output_dir,
                model_name=model_name
            )
            
            self.ensemble_models.append({
                'model': trained_model,
                'metrics': metrics,
                'name': model_name
            })
            
            # Clear some memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        logger.info(f"Ensemble training complete. Trained {len(self.ensemble_models)} models.")
        return True
        
    def run_pipeline(self):
        """Run the full pipeline"""
        logger.info(f"====== Starting Hybrid Cross Attention Pipeline for {self.dataset_type} ======")
        
        # Load and process data
        if not self.load_data():
            return False
            
        # Setup label encoding
        self.setup_label_encoding()
        
        # Split dataset 
        self.split_dataset()
        
        # Prepare datasets and dataloaders
        if not self.prepare_dataset():
            return False
        
        # Train ensemble models
        self.train_ensemble()
        
        logger.info(f"====== Completed Hybrid Cross Attention Pipeline for {self.dataset_type} ======")
        return True

def run_hybrid_ensemble_pipeline(dataset_type='depression', custom_config=None):
    """
    Run the full pipeline for training an ensemble of hybrid cross-attention models.
    
    Args:
        dataset_type (str): The type of dataset to use ('anxiety' or 'depression')
        custom_config (dict, optional): Custom configuration parameters to override defaults
        
    Returns:
        bool: True if the pipeline completed successfully, False otherwise
    """
    try:
        # Initialize the pipeline
        pipeline = HybridModelPipeline(dataset_type=dataset_type, config=custom_config)
        
        # Run the pipeline
        success = pipeline.run_pipeline()
        
        return success
    except Exception as e:
        logger.error(f"Error running hybrid ensemble pipeline: {e}", exc_info=True)
        return False

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