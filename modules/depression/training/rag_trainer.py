# -*- coding: utf-8 -*- # Add this line for better encoding support

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm # Use tqdm directly
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from collections import defaultdict
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
# Import AutoConfig to potentially modify dropout, although it might conflict with pretrained settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import re
import ast # To safely evaluate string representation of lists

# --- Configuration ---

# Set device (use GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If using DataParallel, specify target devices
DEVICE_IDS = [0,1] # Adjust GPU IDs if needed and using DataParallel
PRIMARY_DEVICE = f'cuda:{DEVICE_IDS[0]}' if DEVICE == torch.device("cuda") and DEVICE_IDS else DEVICE
print(f"Using primary device: {PRIMARY_DEVICE}")

# --- Hyperparameters from Paper ---
LEARNING_RATE = 5e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 1e-2 # Paper specified 1*10^-2
DROPOUT_RATE = 0.2 # From paper - Applying this via config might be complex/risky, using default for now
BATCH_SIZE = 16
MAX_LENGTH = 512 # Capped at 512 as per paper
GRADIENT_CLIP_THRESHOLD = 1.0

# --- Multi-label specific parameters ---
# Threshold for binary classification decisions per label
THRESHOLD = 0.5  # Can be adjusted or optimized per class

# --- Other Configurations ---
TOP_K = 3
N_SHOT = 1 # Number of few-shot examples to include in the prompt (can be <= TOP_K)
EPOCHS = 10 # You might need more epochs depending on convergence
FREEZE_LAYERS_BELOW = 10 # Keep freezing strategy unless LoRA is explicitly implemented

# Model Name
MODEL_NAME = 'Tianlin668/MentalBART'

# DATASET DIRECTORY CONFIGURATION
try:
    script_dir = os.path.dirname(__file__)
    # Go up three levels from script dir to project root
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # Adjust if needed
    # Simple check if the detected root seems plausible
    if project_root == '/' or not os.path.exists(os.path.join(project_root, 'README.md')): # Heuristic check
         print("Warning: Project root auto-detection might be incorrect. Adjust path joining if necessary.")
         # Fallback or specific path might be needed depending on structure
         project_root = os.path.abspath(".") # Assume script is run from near project root

    DATASET_DIRECTORY = os.path.join(project_root, "dataset")
    print(f"Attempting to use DATASET_DIRECTORY: {DATASET_DIRECTORY}")
    if not os.path.exists(DATASET_DIRECTORY):
         print(f"ERROR: Cannot find dataset directory at {DATASET_DIRECTORY}.")
         # Provide guidance if path is wrong
         print("Please ensure the 'dataset' directory exists at the root of your project.")
         sys.exit(1)

except NameError:
    # This block runs if __file__ is not defined (e.g., in an interactive environment)
    print("Warning: __file__ not defined. Using hardcoded relative paths from CWD.")
    # Assume CWD is project root for this fallback
    project_root = os.path.abspath(".")
    DATASET_DIRECTORY = os.path.join(project_root, "dataset")
    if not os.path.exists(DATASET_DIRECTORY):
        print(f"ERROR: Cannot find dataset directory at {DATASET_DIRECTORY} relative to CWD.")
        print("Please run the script from the project root directory or adjust paths.")
        sys.exit(1)

DEPRESSION_DATASET_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Depressive_Data")

# --- Input File Paths ---
# Original data files
TARGET_TRAIN_FILE_PATH = os.path.join(DEPRESSION_DATASET_DIRECTORY, "final", "cleaned", "depressive_train_combined_preprocessed.json")
TARGET_TEST_FILE_PATH = os.path.join(DEPRESSION_DATASET_DIRECTORY, "final", "cleaned", "depressive_test_combined_preprocessed.json")
TARGET_VAL_FILE_PATH = os.path.join(DEPRESSION_DATASET_DIRECTORY, "final", "cleaned", "depressive_val_combined_preprocessed.json")

# Embedding directory and retrieval results
EMBEDDING_DIR = os.path.join(project_root, "depressive_embeddings_output") # Adjust if needed
print(f"Expecting embedding/retrieval files in: {EMBEDDING_DIR}")
RETRIEVED_TEST_INDICES_FILE = os.path.join(EMBEDDING_DIR, f"test_top_{TOP_K}_similar_indices.csv") # Using CSV
RETRIEVED_TRAIN_INDICES_FILE = os.path.join(EMBEDDING_DIR, f"train_top_{TOP_K}_similar_indices.csv") # Using CSV
RETRIEVED_VAL_INDICES_FILE = os.path.join(EMBEDDING_DIR, f"val_top_{TOP_K}_similar_indices.csv")

# --- Helper Functions ---

def split_ocr(text, limit):
    """Splits text by words up to a specified limit."""
    if not isinstance(text, str): text = str(text)
    words = text.split()
    if len(words) > limit: return ' '.join(words[:limit]) + '...'
    return text

def process_triples(triples):
    """
    Robustly processes the figurative reasoning string to extract relevant sections.
    Prioritizes finding all three key sections ('cause-effect', 'figurative understanding', 'mental state').
    If all three sections are reliably found and extracted, returns the structured, cleaned text.
    Otherwise, returns the entire original reasoning string, lowercased, to avoid information loss.

    Args:
        triples (str): The raw figurative reasoning string.

    Returns:
        str: The processed reasoning string or the lowercased original string.
    """
    if not isinstance(triples, str) or not triples.strip():
        return "Missing or empty reasoning" # Handle None, empty strings

    text_lower = triples.lower() # Lowercase the entire input first

    sections_to_find = {
        'cause-effect': None,
        'figurative understanding': None,
        'mental state': None
    }

    # Pattern to find section markers like "1. **title:**", "**title:**", "### title" etc.
    # Captures the title itself (group 1). Uses re.MULTILINE.
    # Make it more robust to variations in spacing and markdown around the title.
    pattern_section_start = r'(?:^\s*\d+\.\s*)?(?:[\#\*]*)\s*(cause-effect|figurative understanding|mental state)\b'

    matches = list(re.finditer(pattern_section_start, text_lower, re.MULTILINE))

    # If no potential section markers are found at all, fallback immediately
    if not matches:
        # print(f"Debug: No section markers found for input: {text_lower[:100]}...") # Optional debug
        return text_lower # Fallback to original lowercased text

    # Store found sections and their start/end points
    found_sections_pos = {}
    for match in matches:
        title = match.group(1).strip()
        # Find the end of the marker (e.g., after the colon and stars if present)
        marker_end_match = re.search(r'[:\*]+', text_lower[match.end():match.end()+10])
        marker_end_pos = match.end() + marker_end_match.end() if marker_end_match else match.end()
        found_sections_pos[title] = {'start': match.start(), 'marker_end': marker_end_pos}

    # Determine content boundaries and extract raw content
    sorted_titles = sorted(found_sections_pos.keys(), key=lambda k: found_sections_pos[k]['start'])

    raw_content = {}
    for i, title in enumerate(sorted_titles):
        start_content = found_sections_pos[title]['marker_end']
        # End content at the start of the *next* found section marker, or end of text
        next_section_start = len(text_lower) # Default to end of string
        if (i + 1) < len(sorted_titles):
            next_title = sorted_titles[i+1]
            next_section_start = found_sections_pos[next_title]['start']

        content = text_lower[start_content:next_section_start].strip()
        raw_content[title] = content


    # Clean the extracted raw content for each required section
    for title in sections_to_find.keys():
        if title in raw_content:
            content_to_clean = raw_content[title]
            # Basic cleaning: split lines, remove leading markers/whitespace, join
            lines = content_to_clean.splitlines()
            cleaned_lines = []
            for line in lines:
                # Remove leading list markers, hyphens, stars, digits, colons, whitespace
                cleaned_line = re.sub(r'^\s*[-\*\u2022\d\.\:]+\s*', '', line).strip()
                # Remove potential leftover sub-titles (simple version)
                cleaned_line = re.sub(r'^\s*\w+\s*:', '', cleaned_line).strip()

                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
            cleaned_content = ' '.join(cleaned_lines) # Join with spaces

            if cleaned_content: # Only store if cleaning didn't result in empty string
                 sections_to_find[title] = cleaned_content


    # --- Validation and Fallback Logic ---
    # Check if *all three* required sections were found and have content
    all_sections_found = all(content is not None for content in sections_to_find.values())

    if all_sections_found:
        # Format the output string
        output_lines = [f"{title}: {content}" for title, content in sections_to_find.items()]
        return '\n'.join(output_lines)
    else:
        # If any section is missing or empty after cleaning, return the original lowercased text
        # print(f"Debug: Fallback triggered. Found sections: { {k:v is not None for k,v in sections_to_find.items()} }") # Optional Debug
        return text_lower

def prompt_it(current_df, original_train_df, mlb, n_shot=1):
    """Formats data into prompts including few-shot examples for multi-label classification."""
    prompts = []
    ocr_word_limit = 75

    original_train_lookup = {idx: row for idx, row in original_train_df.iterrows()}
    class_names = mlb.classes_
    class_names_str = str(list(class_names))

    print(f"Generating prompts with {n_shot} examples...")
    for idx, row in tqdm(current_df.iterrows(), total=len(current_df), desc="Generating Prompts"):
        current_ocr = row['ocr_text']
        current_reasoning = row['processed_reasoning']
        retrieved_indices_str = row[f'top_{TOP_K}_train_indices']

        try:
            retrieved_indices = ast.literal_eval(retrieved_indices_str)
            if not isinstance(retrieved_indices, list): raise ValueError("Parsed indices not list")
        except (ValueError, SyntaxError, TypeError) as e:
            retrieved_indices = []

        # --- Build Few-Shot Examples ---
        few_shot_examples_str = ""
        indices_to_use = retrieved_indices[:n_shot]
        example_count = 0
        for i, train_index in enumerate(indices_to_use):
            try:
                if train_index in original_train_lookup:
                    example_row = original_train_lookup[train_index]
                    shot_ocr = example_row['ocr_text']
                    shot_reasoning = process_triples(example_row['figurative_reasoning'])
                    
                    # For multi-label, get all labels that apply
                    shot_labels = example_row['meme_depressive_categories']
                    if isinstance(shot_labels, str):
                        # Convert string representation to actual list
                        try:
                            shot_labels = ast.literal_eval(shot_labels)
                        except:
                            shot_labels = [shot_labels]  # Fallback to single-item list
                    
                    if not isinstance(shot_labels, list):
                        shot_labels = [shot_labels]
                    
                    # Format the labels as comma-separated string
                    shot_labels_str = ", ".join(shot_labels)

                    shot_ocr_limited = split_ocr(shot_ocr, ocr_word_limit)

                    few_shot_examples_str += f"##Example {example_count + 1}:\n"
                    few_shot_examples_str += f"<|ocr_text|>{shot_ocr_limited}\n"
                    few_shot_examples_str += f"<|commonsense figurative explanation|>{shot_reasoning}\n"
                    few_shot_examples_str += f"The mental health disorders of the person for this post are: {shot_labels_str}\n\n"
                    example_count += 1
            except Exception as e:
                 print(f"Warning: Error processing few-shot example index {train_index} for {row['sample_id']}: {e}.")

        # --- Build Final Prompt ---
        current_ocr_limited = split_ocr(current_ocr, ocr_word_limit)

        prompt = f"#System: You specialize in analyzing mental health behaviors through social media posts. Your task is to identify all mental health issues depicted in a person's post from the following categories: {class_names_str}. A post may exhibit multiple mental health issues.\n\n"
        prompt += few_shot_examples_str
        prompt += "###Your_turn:\n"
        prompt += f"<|ocr_text|>{current_ocr_limited}\n"
        prompt += f"<|commonsense figurative explanation|>{current_reasoning}\n"
        prompt += "The mental health disorders of the person for this post are: "

        prompts.append(prompt)

    current_df['prompt'] = prompts
    return current_df


# --- Dataset Class ---
class MemeMultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx] if isinstance(self.texts[idx], str) else str(self.texts[idx])
        label = self.labels[idx]
        try:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            return {
                'input_ids': input_ids.to(torch.long),
                'attention_mask': attention_mask.to(torch.long),
                'targets': torch.tensor(label, dtype=torch.float)  # Float for BCE loss
            }
        except Exception as e:
             print(f"Error tokenizing text at index {idx}: {e}")
             dummy_encoding = self.tokenizer.encode_plus("Error", max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
             return {
                'input_ids': dummy_encoding['input_ids'].squeeze(0).to(torch.long),
                'attention_mask': dummy_encoding['attention_mask'].squeeze(0).to(torch.long),
                'targets': torch.ones(len(self.labels[0]), dtype=torch.float) * -1  # Invalid marker for multi-label
             }

# --- Training and Evaluation Functions ---
def train_model(training_loader, model, optimizer, criterion, device, gradient_clip_val):
    model.train()
    train_losses = []
    loop = tqdm(training_loader, total=len(training_loader), leave=False, colour='cyan', desc="Training")
    for batch_idx, data in enumerate(loop):
        # For multi-label, check if any example has valid labels (not all -1)
        valid_mask = (data['targets'][:, 0] != -1)  # Check first label column
        if not valid_mask.any(): continue

        ids = data['input_ids'][valid_mask].to(device)
        mask = data['attention_mask'][valid_mask].to(device)
        targets = data['targets'][valid_mask].to(device)

        optimizer.zero_grad()
        outputs = model(ids, attention_mask=mask).logits
        loss = criterion(outputs, targets)

        if torch.isnan(loss): continue

        loss.backward()
        # Apply Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        optimizer.step()

        train_losses.append(loss.item())
        loop.set_postfix(loss=loss.item())

    return np.mean(train_losses) if train_losses else 0

def eval_model(validation_loader, model, criterion, device, threshold, class_names):
    model.eval()
    val_targets = []
    val_outputs = []
    val_losses = []
    all_probs = []
    
    loop = tqdm(validation_loader, total=len(validation_loader), leave=False, colour='magenta', desc="Evaluating")
    with torch.no_grad():
        for batch_idx, data in enumerate(loop):
            valid_mask = (data['targets'][:, 0] != -1)  # Check first label column
            if not valid_mask.any(): continue

            ids = data['input_ids'][valid_mask].to(device)
            mask = data['attention_mask'][valid_mask].to(device)
            targets = data['targets'][valid_mask].to(device)

            outputs = model(ids, attention_mask=mask).logits
            loss = criterion(outputs, targets)

            if torch.isnan(loss): continue

            val_losses.append(loss.item())
            
            # Store actual probability values
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            
            # Apply threshold for binary predictions
            preds = (probs >= threshold).astype(int)
            
            val_targets.extend(targets.cpu().numpy())
            val_outputs.extend(preds)

    if not val_targets:
        print("Warning: No valid validation samples processed.")
        return {
            'accuracy': 0.0,
            'macro avg': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 0},
            'weighted avg': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 0}
        }, [], [], 0.0, 0.0, []

    val_targets = np.array(val_targets)
    val_outputs = np.array(val_outputs)
    
    # Calculate metrics
    # For multi-label, we use sample-average metrics
    f1_macro = f1_score(val_targets, val_outputs, average='macro', zero_division=0)
    f1_micro = f1_score(val_targets, val_outputs, average='micro', zero_division=0)
    f1_weighted = f1_score(val_targets, val_outputs, average='weighted', zero_division=0)
    precision_macro = precision_score(val_targets, val_outputs, average='macro', zero_division=0)
    recall_macro = recall_score(val_targets, val_outputs, average='macro', zero_division=0)
    
    # Calculate per-class metrics
    report = classification_report(
        val_targets, val_outputs,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Add additional multi-label metrics to report
    report['f1_macro'] = f1_macro
    report['f1_micro'] = f1_micro
    report['f1_weighted'] = f1_weighted
    report['precision_macro'] = precision_macro
    report['recall_macro'] = recall_macro
    
    # Calculate accuracy - for multi-label this is subset accuracy
    # (percentage of samples where all labels match)
    accuracy = np.mean(np.all(val_targets == val_outputs, axis=1))
    report['accuracy'] = accuracy

    return report, val_targets, val_outputs, np.mean(val_losses) if val_losses else 0, f1_macro, all_probs


# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # 1. Load Data
    print("Loading data...")
    try:
        train_orig_df = pd.read_json(TARGET_TRAIN_FILE_PATH)
        test_orig_df = pd.read_json(TARGET_TEST_FILE_PATH)
        val_orig_df = pd.read_json(TARGET_VAL_FILE_PATH)
        train_retrieval_df = pd.read_csv(RETRIEVED_TRAIN_INDICES_FILE)
        test_retrieval_df = pd.read_csv(RETRIEVED_TEST_INDICES_FILE)
        val_retrieval_df = pd.read_csv(RETRIEVED_VAL_INDICES_FILE)
        
        print(f"Loaded train df shape: {train_orig_df.shape}")
        print(f"Loaded test df shape: {test_orig_df.shape}")
        print(f"Loaded val df shape: {val_orig_df.shape}")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please check paths.")
        sys.exit(1)

    # 2. Process the multi-label data
    label_col = 'meme_depressive_categories'  # Column with list of depression categories
    print(f"Original train samples: {len(train_orig_df)}")
    print(f"Original test samples: {len(test_orig_df)}")
    print(f"Original val samples: {len(val_orig_df)}")
    
    # Convert string representation of lists to actual lists
    def parse_label_lists(df, column_name):
        def parse_labels(labels):
            # Check if labels is None or NaN (scalar check)
            if labels is None or (isinstance(labels, float) and pd.isna(labels)):
                return []
            
            # If it's already a list, return it directly
            if isinstance(labels, list):
                return labels
            
            # Try to parse string representation of a list
            try:
                parsed = ast.literal_eval(labels)
                if isinstance(parsed, list):
                    return parsed
                else:
                    return [parsed]  # Handle single item that's not a list
            except (ValueError, SyntaxError, TypeError):
                # If it can't be parsed as a list, treat as a single label
                return [labels]
        
        df[column_name] = df[column_name].apply(parse_labels)
        return df
    
    train_orig_df = parse_label_lists(train_orig_df, label_col)
    test_orig_df = parse_label_lists(test_orig_df, label_col)
    val_orig_df = parse_label_lists(val_orig_df, label_col)
    
    # Reset index after any processing
    train_orig_df.reset_index(drop=True, inplace=True)
    test_orig_df.reset_index(drop=True, inplace=True)
    val_orig_df.reset_index(drop=True, inplace=True)

    # 3. Merge DataFrames
    print("Merging dataframes...")
    train_orig_df['sample_id'] = train_orig_df['sample_id'].astype(str)
    test_orig_df['sample_id'] = test_orig_df['sample_id'].astype(str)
    val_orig_df['sample_id'] = val_orig_df['sample_id'].astype(str)
    train_retrieval_df['sample_id'] = train_retrieval_df['sample_id'].astype(str)
    test_retrieval_df['sample_id'] = test_retrieval_df['sample_id'].astype(str)
    val_retrieval_df['sample_id'] = val_retrieval_df['sample_id'].astype(str)

    # Merge filtered data with retrieval data
    train_df = pd.merge(train_orig_df, train_retrieval_df, on='sample_id', how='left')
    test_df = pd.merge(test_orig_df, test_retrieval_df, on='sample_id', how='left')
    val_df = pd.merge(val_orig_df, val_retrieval_df, on='sample_id', how='left')

    indices_col = f'top_{TOP_K}_train_indices'
    if train_df[indices_col].isnull().any():
        missing_count = train_df[indices_col].isnull().sum()
        print(f"Warning: {missing_count} training samples missing retrieval indices after merge. Filling with '[]'.")
        train_df[indices_col].fillna('[]', inplace=True)
    if test_df[indices_col].isnull().any():
        missing_count = test_df[indices_col].isnull().sum()
        print(f"Warning: {missing_count} test samples missing retrieval indices after merge. Filling with '[]'.")
        test_df[indices_col].fillna('[]', inplace=True)
    if val_df[indices_col].isnull().any():
        missing_count = val_df[indices_col].isnull().sum()
        print(f"Warning: {missing_count} validation samples missing retrieval indices after merge. Filling with '[]'.")
        val_df[indices_col].fillna('[]', inplace=True)


    # 4. Process Figurative Reasoning
    reasoning_col = 'figurative_reasoning'
    print(f"Processing reasoning column: {reasoning_col}")
    train_df['processed_reasoning'] = train_df[reasoning_col].apply(process_triples)
    test_df['processed_reasoning'] = test_df[reasoning_col].apply(process_triples)
    val_df['processed_reasoning'] = val_df[reasoning_col].apply(process_triples)

    # --- Example Check after processing ---
    print("\n--- Checking Processed Reasoning (Example) ---")
    if not train_df.empty:
        example_idx_check = 0
        print(f"Original Reasoning (Sample 0):\n{train_df.loc[example_idx_check, reasoning_col]}\n")
        print(f"Processed Reasoning (Sample 0):\n{train_df.loc[example_idx_check, 'processed_reasoning']}")
    print("-------------------------------------------\n")
    # --- Check another random one from test ---
    if not test_df.empty:
        print(f"Processed Reasoning (Test Sample 0):\n{test_df.loc[0, 'processed_reasoning']}")
        print("-------------------------------------------\n")


    # 5. Multi-label Encoding
    print("Encoding multi-labels...")
    mlb = MultiLabelBinarizer()
    
    # Extract all unique labels across train, test, and val sets
    all_labels = []
    for labels_list in train_df[label_col]:
        all_labels.extend(labels_list)
    for labels_list in test_df[label_col]:
        all_labels.extend(labels_list)
    for labels_list in val_df[label_col]:
        all_labels.extend(labels_list)
    
    # Fit the binarizer on all unique labels
    mlb.fit([list(set(all_labels))])
    
    # Transform labels to multi-hot encoding
    train_df['labels'] = list(mlb.transform(train_df[label_col]))
    test_df['labels'] = list(mlb.transform(test_df[label_col]))
    val_df['labels'] = list(mlb.transform(val_df[label_col]))
    
    num_labels = len(mlb.classes_)
    class_names = list(mlb.classes_)
    print(f"Found {num_labels} unique labels: {class_names}")
    
    # Print sample transform
    sample_labels = train_df[label_col].iloc[0]
    sample_encoded = mlb.transform([sample_labels])[0]
    print(f"Sample multi-label encoding:\nLabels: {sample_labels}\nEncoded: {sample_encoded}")


    # 6. Generate Prompts
    train_orig_df_for_lookup = pd.read_json(TARGET_TRAIN_FILE_PATH)
    train_orig_df_for_lookup = parse_label_lists(train_orig_df_for_lookup, label_col)
    train_orig_df_for_lookup.reset_index(drop=True, inplace=True)

    train_df = prompt_it(train_df, train_orig_df_for_lookup, mlb, n_shot=N_SHOT)
    test_df = prompt_it(test_df, train_orig_df_for_lookup, mlb, n_shot=N_SHOT)
    val_df = prompt_it(val_df, train_orig_df_for_lookup, mlb, n_shot=N_SHOT)


    # 7. Tokenizer and Model Loading
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


    # 8. Layer Freezing (remains the same)
    print(f"Freezing layers below {FREEZE_LAYERS_BELOW}...")
    layer_prefix = "model."
    frozen_count = 0
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        do_freeze = False
        if name.startswith(layer_prefix + "encoder.layers.") or \
           name.startswith(layer_prefix + "decoder.layers."):
           try:
               layer_num = int(name.split('.')[3])
               if layer_num < FREEZE_LAYERS_BELOW:
                   param.requires_grad = False
                   do_freeze = True
                   frozen_count +=1
           except (IndexError, ValueError): pass
        if param.requires_grad: trainable_params += 1
    print(f"Froze {frozen_count}/{total_params} param groups. Trainable: {trainable_params}")


    # Move model to device(s)
    if torch.cuda.device_count() > 1 and len(DEVICE_IDS) > 1:
        print(f"Using {len(DEVICE_IDS)} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)
    model.to(PRIMARY_DEVICE)


    # 9. Create Datasets and DataLoaders
    print("Creating datasets and dataloaders...")
    train_dataset = MemeMultiLabelDataset(train_df['prompt'].tolist(), train_df['labels'].tolist(), tokenizer, MAX_LENGTH)
    val_dataset = MemeMultiLabelDataset(val_df['prompt'].tolist(), val_df['labels'].tolist(), tokenizer, MAX_LENGTH)
    test_dataset = MemeMultiLabelDataset(test_df['prompt'].tolist(), test_df['labels'].tolist(), tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


    # 10. Optimizer and Loss Function with Paper Hyperparameters
    print("Setting up optimizer with specified hyperparameters...")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY
    )
    # Use Binary Cross Entropy With Logits Loss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()

    # 11. Training Loop
    print("Starting training...")
    history = defaultdict(list)
    best_f1_macro = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f'\n--- Epoch {epoch}/{EPOCHS} ---')
        # Pass gradient clip value to training function
        train_loss = train_model(train_loader, model, optimizer, criterion, PRIMARY_DEVICE, GRADIENT_CLIP_THRESHOLD)
        # Pass final class names to eval function
        val_report, val_targets, val_outputs, val_loss, val_f1_macro, val_probs = eval_model(
            val_loader, model, criterion, PRIMARY_DEVICE, THRESHOLD, class_names
        )

        # Extract metrics from report
        val_f1_weighted = val_report.get('f1_weighted', 0.0)
        val_precision = val_report.get('precision_macro', 0.0)
        val_recall = val_report.get('recall_macro', 0.0)
        val_accuracy = val_report.get('accuracy', 0.0)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Subset Accuracy: {val_accuracy:.4f}")
        print(f"  Val F1 Macro: {val_f1_macro:.4f}")
        print(f"  Val F1 Weighted: {val_f1_weighted:.4f}")
        print(f"  Val Precision Macro: {val_precision:.4f}")
        print(f"  Val Recall Macro: {val_recall:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)

        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            save_dir = "mental_bart_depression_finetuned_multilabel"
            os.makedirs(save_dir, exist_ok=True)
            model_to_save.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
            # Save label binarizer and threshold
            import pickle
            with open(os.path.join(save_dir, "multilabel_binarizer.pkl"), "wb") as f:
                pickle.dump(mlb, f)
            with open(os.path.join(save_dir, "threshold.txt"), "w") as f:
                f.write(str(THRESHOLD))
                
            print(f"  -> New best model saved with Macro F1: {best_f1_macro:.4f} in '{save_dir}'")

            # Print detailed classification report for best model
            if val_targets is not None and val_outputs is not None and len(val_targets) > 0:
                print("\nClassification Report (Best Epoch):")
                for label_idx, label_name in enumerate(class_names):
                    label_precision = val_report.get(label_name, {}).get('precision', 0.0)
                    label_recall = val_report.get(label_name, {}).get('recall', 0.0)
                    label_f1 = val_report.get(label_name, {}).get('f1-score', 0.0)
                    label_support = val_report.get(label_name, {}).get('support', 0)
                    print(f"  {label_name}: Precision={label_precision:.4f}, Recall={label_recall:.4f}, F1={label_f1:.4f}, Support={label_support}")
                
                print(f"\n  Macro avg: Precision={val_precision:.4f}, Recall={val_recall:.4f}, F1={val_f1_macro:.4f}")
                print(f"  Weighted avg: F1={val_f1_weighted:.4f}")

    # 12. Final Evaluation on Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    # Load best model for final evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(save_dir, num_labels=num_labels)
    best_model.to(PRIMARY_DEVICE)
    
    test_report, test_targets, test_outputs, test_loss, test_f1_macro, test_probs = eval_model(
        test_loader, best_model, criterion, PRIMARY_DEVICE, THRESHOLD, class_names
    )
    
    test_f1_weighted = test_report.get('f1_weighted', 0.0)
    test_precision = test_report.get('precision_macro', 0.0)
    test_recall = test_report.get('recall_macro', 0.0)
    test_accuracy = test_report.get('accuracy', 0.0)
    
    print(f"\nTest Set Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Subset Accuracy: {test_accuracy:.4f}")
    print(f"  Test F1 Macro: {test_f1_macro:.4f}")
    print(f"  Test F1 Weighted: {test_f1_weighted:.4f}")
    print(f"  Test Precision Macro: {test_precision:.4f}")
    print(f"  Test Recall Macro: {test_recall:.4f}")
    
    # Save test results
    with open(os.path.join(save_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Subset Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test F1 Macro: {test_f1_macro:.4f}\n")
        f.write(f"Test F1 Weighted: {test_f1_weighted:.4f}\n")
        f.write(f"Test Precision Macro: {test_precision:.4f}\n")
        f.write(f"Test Recall Macro: {test_recall:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        for label_idx, label_name in enumerate(class_names):
            label_precision = test_report.get(label_name, {}).get('precision', 0.0)
            label_recall = test_report.get(label_name, {}).get('recall', 0.0)
            label_f1 = test_report.get(label_name, {}).get('f1-score', 0.0)
            label_support = test_report.get(label_name, {}).get('support', 0)
            f.write(f"{label_name}: Precision={label_precision:.4f}, Recall={label_recall:.4f}, F1={label_f1:.4f}, Support={label_support}\n")

    # 13. Plotting Results
    print("\nPlotting training history...")
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Losses')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(history['val_accuracy'], label='Validation Subset Accuracy', color='green')
    axs[1].set_title('Validation Subset Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(history['val_f1_macro'], label='Validation F1 Macro', color='orange')
    axs[2].plot(history['val_f1_weighted'], label='Validation F1 Weighted', color='red')
    axs[2].set_title('Validation F1 Scores')
    axs[2].set_ylabel('F1 Score')
    axs[2].set_xlabel('Epoch')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(history['val_precision'], label='Validation Precision', color='purple')
    axs[3].plot(history['val_recall'], label='Validation Recall', color='brown')
    axs[3].set_title('Validation Precision and Recall')
    axs[3].set_ylabel('Score')
    axs[3].set_xlabel('Epoch')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(EMBEDDING_DIR, "training_history_multilabel_depression.png")
    try:
        plt.savefig(plot_filename)
        print(f"Training history plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # 14. Class-wise Performance Analysis
    print("\nAnalyzing class-wise performance...")
    
    # Create a dataframe to analyze performance across classes
    class_metrics = []
    for label_idx, label_name in enumerate(class_names):
        metrics = {
            'Class': label_name,
            'Precision': test_report.get(label_name, {}).get('precision', 0.0),
            'Recall': test_report.get(label_name, {}).get('recall', 0.0),
            'F1-Score': test_report.get(label_name, {}).get('f1-score', 0.0),
            'Support': test_report.get(label_name, {}).get('support', 0)
        }
        class_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(class_metrics)
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    
    # Save class metrics
    metrics_df.to_csv(os.path.join(save_dir, "class_performance_metrics.csv"), index=False)
    
    # Plot class performance
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(metrics_df))
    width = 0.25
    
    ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='blue', alpha=0.7)
    ax.bar(x, metrics_df['Recall'], width, label='Recall', color='green', alpha=0.7)
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='red', alpha=0.7)
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Class'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    class_plot_filename = os.path.join(save_dir, "class_performance_plot.png")
    try:
        plt.savefig(class_plot_filename)
        print(f"Class performance plot saved to {class_plot_filename}")
    except Exception as e:
        print(f"Error saving class performance plot: {e}")

    # 15. Threshold Analysis (for future optimization)
    print("\nSaving probability distributions for threshold optimization...")
    
    # Create a dataframe with all prediction probabilities for future threshold tuning
    probs_df = pd.DataFrame(test_probs, columns=class_names)
    probs_df['true_labels'] = [str(label) for label in test_targets.tolist()]
    probs_df.to_csv(os.path.join(save_dir, "test_probabilities.csv"), index=False)
    
    print("\n--- Training and Evaluation Complete ---")