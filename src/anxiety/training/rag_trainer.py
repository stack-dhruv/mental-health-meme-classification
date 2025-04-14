import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm # Use tqdm directly
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import ast # To safely evaluate string representation of lists

# --- Configuration ---

# Set device (use GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If using DataParallel, specify target devices
DEVICE_IDS = [0] # Adjust GPU IDs if needed and using DataParallel
PRIMARY_DEVICE = f'cuda:{DEVICE_IDS[0]}' if DEVICE == torch.device("cuda") and DEVICE_IDS else DEVICE
print(f"Using primary device: {PRIMARY_DEVICE}")

# Number of top similar examples to retrieve (should match the previous script)
TOP_K = 3
N_SHOT = 1 # Number of few-shot examples to include in the prompt (can be <= TOP_K)

# Model and Training Params
MODEL_NAME = 'Tianlin668/MentalBART'
MAX_LENGTH = 512 # Max sequence length for tokenizer (adjust based on token analysis if needed)
BATCH_SIZE = 16 # Adjust based on GPU memory
EPOCHS = 10
LEARNING_RATE = 5e-5
FREEZE_LAYERS_BELOW = 10 # Freeze layers < 10 in BART encoder/decoder

# DATASET DIRECTORY CONFIGURATION
try:
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # Adjust if needed
    DATASET_DIRECTORY = os.path.join(project_root, "dataset")
except NameError:
    print("Warning: __file__ not defined. Using current working directory structure.")
    relative_dataset_path = os.path.join("..", "..", "..", "dataset") # Adjust if CWD is different
    DATASET_DIRECTORY = os.path.abspath(relative_dataset_path)
    print(DATASET_DIRECTORY)
    if not os.path.exists(DATASET_DIRECTORY):
        relative_dataset_path = os.path.join("dataset") # Try alternate structure
        DATASET_DIRECTORY = os.path.abspath(relative_dataset_path)

ANXIETY_DATASET_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Anxiety_Data")

# --- Input File Paths ---
# Original data files
TARGET_TRAIN_FILE_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "final", "cleaned", "anxiety_train_combined_preprocessed.json")
TARGET_TEST_FILE_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "final", "cleaned", "anxiety_test_combined_preprocessed.json")

# Embedding directory and retrieval results
EMBEDDING_DIR = os.path.join(project_root, "anxiety_embeddings_output") # Adjust if you used a different name
RETRIEVED_TEST_INDICES_FILE = os.path.join(EMBEDDING_DIR, f"test_top_{TOP_K}_similar_indices.csv") # Using CSV
RETRIEVED_TRAIN_INDICES_FILE = os.path.join(EMBEDDING_DIR, f"train_top_{TOP_K}_similar_indices.csv") # Using CSV

# --- Helper Functions ---

def split_ocr(text, limit):
    """Splits text by words up to a specified limit."""
    if not isinstance(text, str): # Handle potential non-string data
        text = str(text)
    words = text.split()
    if len(words) > limit:
        return ' '.join(words[:limit]) + '...'
    return text

def detect_formatting(triples):
    """Detects the formatting used for section titles."""
    if not isinstance(triples, str): return 'unknown'
    if '###' in triples: return 'hash'
    if '**' in triples: return 'star'
    return 'unknown'

def process_triples(triples):
    """ Process the figurative reasoning triples based on detected formatting. """
    if not isinstance(triples, str): return "Missing reasoning" # Handle non-strings

    formatting = detect_formatting(triples)
    sections_to_keep = ['Cause-Effect', 'Figurative Understanding', 'Mental State']
    sections = []

    # Use regex for more robust splitting
    if formatting == 'hash':
        # Split by '### Title' pattern, keeping delimiter
        sections = re.split(r'(###\s*[A-Za-z\s-]+)', triples)
    elif formatting == 'star':
        # Split by '**Title**' pattern, keeping delimiter
        sections = re.split(r'(\*\*[A-Za-z\s-]+\*\*)', triples)
    else:
         # If no clear formatting, return the original (or process differently)
         # Maybe just return the first N characters/words?
         # For now, let's return a placeholder or the original if short
         # return triples[:200] + "..." if len(triples) > 200 else triples
        return "Unknown or simple formatting" # Or return the original triples

    reduced_triples = []
    # Start from index 1 because split often creates an empty first element
    for i in range(1, len(sections), 2): # Step by 2 (delimiter, content)
        title_raw = sections[i].strip('#* ')
        content_raw = sections[i+1].strip() if (i + 1) < len(sections) else ""

        # Clean title
        title = ' '.join(title_raw.split()) # Normalize whitespace

        if title in sections_to_keep:
            # Clean content - remove potential leading numbers/bullets
            clean_content = re.sub(r'^\s*[\d\.\-\*]+\s*', '', content_raw, flags=re.MULTILINE).strip()
            if clean_content: # Only add if content exists
                reduced_triples.append(f"{title.lower()}: {clean_content}")

    if reduced_triples:
        return '\n'.join(reduced_triples)
    else:
        # Fallback if no specific sections found after splitting
        # return "No specific sections found. Original: " + (triples[:150] + "..." if len(triples) > 150 else triples)
         return "Relevant reasoning sections not found"

def process_triples(triples):
    if not isinstance(triples, str): return "Missing reasoning"

    sections_to_keep = ['Cause-Effect', 'Figurative Understanding', 'Mental State']
    reduced_triples = []

    # Try finding sections with **Title:** pattern first
    # Regex specifically looks for the desired titles enclosed in **
    # It captures the title itself in group 1
    pattern_star = r'\*\*(Cause-Effect|Figurative Understanding|Mental State)\*\*'
    matches_star = list(re.finditer(pattern_star, triples))

    if matches_star: # Found star formatting
        for i, match in enumerate(matches_star):
            title = match.group(1).strip() # Get the captured title name

            # Content starts after the current match ends
            start_index = match.end()
            # Content ends at the start of the next match, or end of string
            end_index = matches_star[i+1].start() if (i + 1) < len(matches_star) else len(triples)

            content_raw = triples[start_index:end_index]

            # Clean the extracted content
            # Remove leading list markers (like 1., -, *), colons, whitespace
            clean_content = re.sub(r'^[\s\:\-\*\d\.]+', '', content_raw).strip()

            # Further cleaning: remove potential internal '**...**' if they weren't meant to be titles
            clean_content = re.sub(r'\*\*[A-Za-z\s-]+\*\*:', '', clean_content).strip()


            if clean_content: # Only add if content exists
                reduced_triples.append(f"{title.lower()}: {clean_content}")

    # If star format didn't yield results, try hash format
    elif '###' in triples:
        pattern_hash = r'###\s*(Cause-Effect|Figurative Understanding|Mental State)'
        matches_hash = list(re.finditer(pattern_hash, triples))
        if matches_hash:
            for i, match in enumerate(matches_hash):
                title = match.group(1).strip() # Get the captured title name

                start_index = match.end()
                end_index = matches_hash[i+1].start() if (i + 1) < len(matches_hash) else len(triples)
                content_raw = triples[start_index:end_index]
                # Clean the extracted content (similar cleaning as above)
                clean_content = re.sub(r'^[\s\:\-\*\d\.]+', '', content_raw).strip()
                if clean_content:
                    reduced_triples.append(f"{title.lower()}: {clean_content}")

    # Return results or fallback
    if reduced_triples:
        return '\n'.join(reduced_triples)
    else:
        # Optional: Log the original triples for inspection if reduction fails
        # print(f"Debug: No relevant sections extracted from: {triples[:100]}...")
        return "Relevant reasoning sections not found" # Or return a snippet of the original

def prompt_it(current_df, original_train_df, n_shot=1):
    """Formats data into prompts including few-shot examples."""
    prompts = []
    ocr_word_limit = 75
    reasoning_word_limit = 150 # Limit reasoning text length in prompt

    # Create a lookup dictionary from the original training data for faster access
    original_train_lookup = original_train_df.set_index('sample_id').to_dict('index')

    print(f"Generating prompts with {n_shot} examples...")
    for idx, row in tqdm(current_df.iterrows(), total=len(current_df), desc="Generating Prompts"):
        current_ocr = row['ocr_text']
        current_reasoning = row['processed_reasoning'] # Use the processed version
        retrieved_indices_str = row[f'top_{TOP_K}_train_indices']

        # Safely evaluate the string representation of the list
        try:
            retrieved_indices = ast.literal_eval(retrieved_indices_str)
            if not isinstance(retrieved_indices, list):
                 raise ValueError("Parsed indices is not a list")
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Warning: Could not parse indices for sample {row['sample_id']}: '{retrieved_indices_str}'. Error: {e}. Skipping few-shot examples for this sample.")
            retrieved_indices = [] # Default to empty list if parsing fails

        # --- Build Few-Shot Examples ---
        few_shot_examples_str = ""
        indices_to_use = retrieved_indices[:n_shot] # Select the top N indices

        example_count = 0
        for i, train_index in enumerate(indices_to_use):
            try:
                # Look up the example in the original training data using its *index*
                example_row = original_train_df.iloc[train_index]
                shot_ocr = example_row['ocr_text']
                # Process reasoning for the example on the fly if not already done
                shot_reasoning = process_triples(example_row['figurative_reasoning'])
                shot_label = example_row['meme_anxiety_category']

                # Apply limits
                shot_ocr_limited = split_ocr(shot_ocr, ocr_word_limit)
                shot_reasoning_limited = split_ocr(shot_reasoning, reasoning_word_limit) # Use split_ocr for word limit

                few_shot_examples_str += f"##Example {example_count + 1}:\n"
                few_shot_examples_str += f"<|ocr_text|>{shot_ocr_limited}\n"
                few_shot_examples_str += f"<|commonsense figurative explanation|>{shot_reasoning_limited}\n"
                few_shot_examples_str += f"The mental health disorder of the person for this post is: {shot_label}\n\n"
                example_count += 1

            except IndexError:
                print(f"Warning: Training index {train_index} out of bounds for sample {row['sample_id']}. Skipping this few-shot example.")
            except Exception as e:
                print(f"Warning: Error retrieving data for training index {train_index} (sample {row['sample_id']}): {e}. Skipping this few-shot example.")

        # --- Build Final Prompt ---
        current_ocr_limited = split_ocr(current_ocr, ocr_word_limit)
        current_reasoning_limited = split_ocr(current_reasoning, reasoning_word_limit)

        # Define the available categories for the model's reference
        # (Get these from the LabelEncoder later for accuracy)
        categories_str = "['Difficulty Relaxing', 'Excessive Worry', 'Impending Doom', 'Irritability', 'Lack of Worry Control', 'Nervousness', 'Restlessness']" # Placeholder

        prompt = f"#System: You specialize in analyzing mental health behaviors through social media posts. Your task is to classify the mental health issue depicted in a person's post from the following categories: {categories_str}.\n\n"
        prompt += few_shot_examples_str # Add the few-shot examples first
        prompt += "###Your_turn:\n"
        prompt += f"<|ocr_text|>{current_ocr_limited}\n"
        prompt += f"<|commonsense figurative explanation|>{current_reasoning_limited}\n"
        prompt += "The mental health disorder of the person for this post is: " # Model needs to complete this

        prompts.append(prompt)

    # Assign prompts back to a new column
    current_df['prompt'] = prompts
    return current_df


# --- Dataset Class ---
class MemeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
                'targets': torch.tensor(label, dtype=torch.long) # Ensure label is a tensor
            }
        except Exception as e:
             print(f"Error tokenizing text at index {idx}: {e}")
             print(f"Text causing error: {text[:500]}...") # Print beginning of problematic text
             # Return a dummy item or raise an error
             # Returning dummy item to avoid crashing the whole process during dataloading
             # Might skew results if too many errors occur.
             dummy_encoding = self.tokenizer.encode_plus(
                "Error processing text.", # Dummy text
                add_special_tokens=True, truncation=True, max_length=self.max_length,
                padding='max_length', return_attention_mask=True, return_tensors='pt',
             )
             return {
                'input_ids': dummy_encoding['input_ids'].squeeze(0).to(torch.long),
                'attention_mask': dummy_encoding['attention_mask'].squeeze(0).to(torch.long),
                'targets': torch.tensor(-1, dtype=torch.long) # Use an invalid label like -1
             }


# --- Training and Evaluation Functions ---
def train_model(training_loader, model, optimizer, criterion, device):
    model.train()
    train_losses = []
    loop = tqdm(training_loader, total=len(training_loader), leave=False, colour='cyan', desc="Training")
    for batch_idx, data in enumerate(loop):
        # Filter out dummy data if any were created
        valid_indices = data['targets'] != -1
        if not valid_indices.any(): continue # Skip batch if all are dummies

        ids = data['input_ids'][valid_indices].to(device)
        mask = data['attention_mask'][valid_indices].to(device)
        targets = data['targets'][valid_indices].to(device)

        optimizer.zero_grad()
        outputs = model(ids, attention_mask=mask).logits
        loss = criterion(outputs, targets)

        if torch.isnan(loss):
            print(f"Warning: NaN loss encountered in training batch {batch_idx}. Skipping batch.")
            # Optional: investigate the inputs (`ids`, `targets`) that caused NaN
            continue # Skip optimizer step and loss recording for this batch

        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())
        loop.set_postfix(loss=loss.item())

    return np.mean(train_losses) if train_losses else 0 # Return 0 if all batches were skipped


def eval_model(validation_loader, model, criterion, device, target_names):
    model.eval()
    val_targets = []
    val_outputs = []
    val_losses = []
    loop = tqdm(validation_loader, total=len(validation_loader), leave=False, colour='magenta', desc="Evaluating")
    with torch.no_grad():
        for batch_idx, data in enumerate(loop):
            valid_indices = data['targets'] != -1
            if not valid_indices.any(): continue

            ids = data['input_ids'][valid_indices].to(device)
            mask = data['attention_mask'][valid_indices].to(device)
            targets = data['targets'][valid_indices].to(device)

            outputs = model(ids, attention_mask=mask).logits
            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in validation batch {batch_idx}. Skipping batch metrics.")
                continue

            val_losses.append(loss.item())
            val_targets.extend(targets.cpu().numpy())
            val_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    val_targets = np.array(val_targets)
    val_outputs = np.array(val_outputs)

    if len(val_targets) == 0: # Handle case where all validation batches had issues
        print("Warning: No valid validation samples processed.")
        return {'accuracy': 0, 'macro avg': {'f1-score': 0}, 'weighted avg': {'f1-score': 0}}, [], [], 0, 0

    accuracy = accuracy_score(val_targets, val_outputs)
    # Use labels=np.unique(val_targets) to only report on classes present in the batch
    report = classification_report(
        val_targets, val_outputs,
        target_names=target_names,
        output_dict=True,
        zero_division=0, # Avoid warnings for classes with no predictions
        labels=np.arange(len(target_names)) # Ensure report includes all potential classes
    )

    return report, val_targets, val_outputs, np.mean(val_losses) if val_losses else 0, accuracy


# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # 1. Load Data
    print("Loading data...")
    try:
        train_orig_df = pd.read_json(TARGET_TRAIN_FILE_PATH)
        test_orig_df = pd.read_json(TARGET_TEST_FILE_PATH)
        train_retrieval_df = pd.read_csv(RETRIEVED_TRAIN_INDICES_FILE)
        test_retrieval_df = pd.read_csv(RETRIEVED_TEST_INDICES_FILE)
        print(f"Loaded {len(train_orig_df)} original train, {len(test_orig_df)} original test samples.")
        print(f"Loaded {len(train_retrieval_df)} train, {len(test_retrieval_df)} test retrieval results.")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        print("Ensure original data JSONs and retrieval CSVs exist.")
        sys.exit(1)

    # 2. Merge DataFrames
    print("Merging dataframes...")
    # Ensure sample_id types match if necessary (e.g., both strings or both ints)
    train_orig_df['sample_id'] = train_orig_df['sample_id'].astype(str)
    test_orig_df['sample_id'] = test_orig_df['sample_id'].astype(str)
    train_retrieval_df['sample_id'] = train_retrieval_df['sample_id'].astype(str)
    test_retrieval_df['sample_id'] = test_retrieval_df['sample_id'].astype(str)

    # Use outer join to keep all samples, check for missing data later
    train_df = pd.merge(train_orig_df, train_retrieval_df, on='sample_id', how='left')
    test_df = pd.merge(test_orig_df, test_retrieval_df, on='sample_id', how='left')

    # Check for samples missing retrieval info (if using left join)
    if train_df[f'top_{TOP_K}_train_indices'].isnull().any():
        print(f"Warning: {train_df[f'top_{TOP_K}_train_indices'].isnull().sum()} training samples missing retrieval indices.")
        # Fill missing indices with empty list string representation or handle as needed
        train_df[f'top_{TOP_K}_train_indices'].fillna('[]', inplace=True)
    if test_df[f'top_{TOP_K}_train_indices'].isnull().any():
        print(f"Warning: {test_df[f'top_{TOP_K}_train_indices'].isnull().sum()} test samples missing retrieval indices.")
        test_df[f'top_{TOP_K}_train_indices'].fillna('[]', inplace=True)

    print(f"Merged train shape: {train_df.shape}, Merged test shape: {test_df.shape}")


    # 3. Process Figurative Reasoning ('triples' or 'figurative_reasoning')
    # Ensure the correct column name is used
    reasoning_col = 'figurative_reasoning' if 'figurative_reasoning' in train_df.columns else 'triples'
    if reasoning_col not in train_df.columns:
         print(f"Error: Reasoning column ('figurative_reasoning' or 'triples') not found in train_df.")
         sys.exit(1)
    print(f"Processing reasoning column: {reasoning_col}")
    train_df['processed_reasoning'] = train_df[reasoning_col].apply(process_triples)
    test_df['processed_reasoning'] = test_df[reasoning_col].apply(process_triples)


    # 4. Label Encoding
    print("Encoding labels...")
    le = LabelEncoder()
    # Use the correct column name for labels
    label_col = 'meme_anxiety_category' if 'meme_anxiety_category' in train_df.columns else 'meme_anxiety_categories'
    if label_col not in train_df.columns:
         print(f"Error: Label column ('meme_anxiety_category' or 'meme_anxiety_categories') not found.")
         sys.exit(1)

    train_df['labels'] = le.fit_transform(train_df[label_col])
    test_df['labels'] = le.transform(test_df[label_col]) # Use transform only on test
    num_labels = len(le.classes_)
    class_names = list(le.classes_)
    print(f"Found {num_labels} classes: {class_names}")

    # 5. Generate Prompts
    # Pass the original training df for lookup
    train_df = prompt_it(train_df, train_orig_df, n_shot=N_SHOT)
    test_df = prompt_it(test_df, train_orig_df, n_shot=N_SHOT) # Use same N_SHOT for test

    # Display a sample prompt
    print("\n--- Sample Prompt (Test Set) ---")
    print(test_df['prompt'].iloc[0])
    print("---------------------------------\n")

    # 6. Tokenizer and Model Loading
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # 7. Layer Freezing
    print(f"Freezing layers below {FREEZE_LAYERS_BELOW}...")
    layer_prefix = "model." # Common prefix for BART layers in HF implementation
    frozen_count = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        do_freeze = False
        # Check both encoder and decoder layers
        if name.startswith(layer_prefix + "encoder.layers.") or \
           name.startswith(layer_prefix + "decoder.layers."):
           try:
               # Extract layer number (e.g., model.encoder.layers.5. -> 5)
               layer_num = int(name.split('.')[3])
               if layer_num < FREEZE_LAYERS_BELOW:
                   param.requires_grad = False
                   do_freeze = True
                   frozen_count +=1
           except (IndexError, ValueError):
                # Handle cases where layer number isn't where expected (e.g., embeddings, final layer norm)
                pass # Keep requires_grad=True by default
        # Optionally freeze embeddings too?
        # if name.startswith(layer_prefix + "shared.") or \
        #    name.startswith(layer_prefix + "encoder.embed_") or \
        #    name.startswith(layer_prefix + "decoder.embed_"):
        #     param.requires_grad = False
        #     do_freeze = True
        #     frozen_count +=1

    print(f"Froze {frozen_count} out of {total_params} parameter groups.")

    # Move model to device(s)
    if torch.cuda.device_count() > 1 and len(DEVICE_IDS) > 1 :
        print(f"Using {len(DEVICE_IDS)} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)
    model.to(PRIMARY_DEVICE)


    # 8. Create Datasets and DataLoaders
    print("Creating datasets and dataloaders...")
    train_dataset = MemeDataset(train_df['prompt'].tolist(), train_df['labels'].tolist(), tokenizer, MAX_LENGTH)
    val_dataset = MemeDataset(test_df['prompt'].tolist(), test_df['labels'].tolist(), tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 9. Optimizer and Loss Function
    # Filter parameters that require gradients for the optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Standard for multi-class classification

    # 10. Training Loop
    print("Starting training...")
    history = defaultdict(list)
    best_f1_macro = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f'\n--- Epoch {epoch}/{EPOCHS} ---')
        train_loss = train_model(train_loader, model, optimizer, criterion, PRIMARY_DEVICE)
        report, _, _, val_loss, accuracy = eval_model(val_loader, model, criterion, PRIMARY_DEVICE, class_names)

        val_f1_macro = report['macro avg']['f1-score']
        val_f1_weighted = report['weighted avg']['f1-score']

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {accuracy:.4f}")
        print(f"  Val F1 Macro: {val_f1_macro:.4f}")
        print(f"  Val F1 Weighted: {val_f1_weighted:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)

        # Save model checkpoint based on best macro F1
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            # Handle DataParallel wrapper if used
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            save_dir = "mental_bart_anxiety_finetuned"
            os.makedirs(save_dir, exist_ok=True)
            model_to_save.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            # Save label encoder classes
            pd.Series(le.classes_).to_json(os.path.join(save_dir, "label_classes.json"))
            print(f"  -> New best model saved with Macro F1: {best_f1_macro:.4f} in '{save_dir}'")
            # Print detailed report for the best epoch
            print("\nClassification Report (Best Epoch):\n", classification_report(
                 eval_model(val_loader, model, criterion, PRIMARY_DEVICE, class_names)[1], # Rerun eval to get targets/outputs
                 eval_model(val_loader, model, criterion, PRIMARY_DEVICE, class_names)[2],
                 target_names=class_names, zero_division=0, labels=np.arange(len(class_names))
             ))


    # 11. Plotting Results
    print("\nPlotting training history...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 15)) # Adjusted size

    # Plotting losses
    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Losses')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()
    axs[0].grid(True)

    # Plotting Accuracy
    axs[1].plot(history['val_accuracy'], label='Validation Accuracy', color='green')
    axs[1].set_title('Validation Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    axs[1].grid(True)


    # Plotting F1 scores
    axs[2].plot(history['val_f1_macro'], label='Validation F1 Macro', color='orange')
    axs[2].plot(history['val_f1_weighted'], label='Validation F1 Weighted', color='red')
    axs[2].set_title('Validation F1 Scores')
    axs[2].set_ylabel('F1 Score')
    axs[2].set_xlabel('Epoch')
    axs[2].legend()
    axs[2].grid(True)


    plt.tight_layout()
    plot_filename = os.path.join(EMBEDDING_DIR, "training_history_plot.png") # Save plot in output dir
    plt.savefig(plot_filename)
    print(f"Training history plot saved to {plot_filename}")
    # plt.show() # Optionally display plot

    print("\n--- Training Complete ---")