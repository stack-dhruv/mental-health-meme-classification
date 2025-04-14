import os
import sys
import pandas as pd
import torch
import tqdm
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# --- Configuration ---

# Set device (use GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# DATASET DIRECTORY CONFIGURATION
try:
    # Construct the relative path first (assuming the script is somewhere within the project)
    # Adjust the number of ".." based on your script's location relative to the project root
    script_dir = os.path.dirname(__file__) # Get the directory of the current script
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # Adjust as needed
    DATASET_DIRECTORY = os.path.join(project_root, "dataset")
except NameError:
    # __file__ is not defined (e.g., running in an interactive environment)
    # Fallback to a hardcoded or relative path from the CWD
    print("Warning: __file__ not defined. Using current working directory structure.")
    relative_dataset_path = os.path.join("..", "..", "..", "dataset") # Adjust if CWD is different
    DATASET_DIRECTORY = os.path.abspath(relative_dataset_path)
    if not os.path.exists(DATASET_DIRECTORY):
         # Try another common structure if the first guess failed
        relative_dataset_path = os.path.join("dataset")
        DATASET_DIRECTORY = os.path.abspath(relative_dataset_path)


ANXIETY_DATASET_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Anxiety_Data")
TARGET_TRAIN_FILE_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "final", "cleaned", "anxiety_train_combined_preprocessed.json")
TARGET_TEST_FILE_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "final", "cleaned", "anxiety_test_combined_preprocessed.json")
TARGET_TRAIN_IMAGES_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "anxiety_train_image")
TARGET_TEST_IMAGES_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "anxiety_test_image")

# Embedding Configuration
TEXT_MODEL_NAME = 'BAAI/bge-m3'
EMBEDDING_DIM = 1024 # Target dimension 'd' for each modality

# Output File Names
OUTPUT_DIR = "anxiety_embeddings_output" # Create a directory to store embeddings
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_OCR_EMB_FILE = os.path.join(OUTPUT_DIR, "train_ocr_embeddings.pt")
TRAIN_FIG_EMB_FILE = os.path.join(OUTPUT_DIR, "train_figurative_reasoning_embeddings.pt")
TRAIN_IMG_EMB_FILE = os.path.join(OUTPUT_DIR, "train_image_resnet50_embeddings.pt")
TRAIN_FUSED_EMB_FILE = os.path.join(OUTPUT_DIR, "train_fused_knowledge_db.pt") # n x 3d

TEST_OCR_EMB_FILE = os.path.join(OUTPUT_DIR, "test_ocr_embeddings.pt")
TEST_FIG_EMB_FILE = os.path.join(OUTPUT_DIR, "test_figurative_reasoning_embeddings.pt")
TEST_IMG_EMB_FILE = os.path.join(OUTPUT_DIR, "test_image_resnet50_embeddings.pt")
# We typically don't create a fused *database* for test, but generate fused embeddings on the fly
# TEST_FUSED_EMB_FILE = os.path.join(OUTPUT_DIR,"test_fused_embeddings.pt")

# --- Model Initialization ---

# 1. Sentence Transformer (Text Embeddings)
print(f"Loading Sentence Transformer: {TEXT_MODEL_NAME}...")
text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
# Check if the model output dim matches our target, bge-m3 should be 1024
actual_text_dim = text_model.get_sentence_embedding_dimension()
if actual_text_dim != EMBEDDING_DIM:
     print(f"Warning: Text model {TEXT_MODEL_NAME} outputs {actual_text_dim} dimensions, but {EMBEDDING_DIM} was requested. Using {actual_text_dim}.")
     # If strictly needed, you might add a projection layer, but bge-m3 is 1024.
     # EMBEDDING_DIM = actual_text_dim # Use the model's actual dimension
else:
    print(f"Text model loaded successfully. Output dimension: {actual_text_dim}")


# 2. Image Feature Extractor (ResNet-50 modified)
print("Loading and modifying ResNet-50...")
image_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # Load pre-trained ResNet-50 V2 weights

# Get the number of input features for the original classifier
num_ftrs = image_model.fc.in_features # This is typically 2048 for ResNet-50

# Remove the final classification layer
image_model.fc = nn.Identity()

# Create a new sequential model to add a projection layer if needed
# If ResNet output (2048) is not EMBEDDING_DIM (1024), add projection
if num_ftrs != EMBEDDING_DIM:
    print(f"Adding projection layer to ResNet-50: {num_ftrs} -> {EMBEDDING_DIM}")
    feature_extractor = nn.Sequential(
        image_model,
        nn.Linear(num_ftrs, EMBEDDING_DIM),
        # Optional: Add activation/normalization if desired
        # nn.ReLU(),
        # nn.LayerNorm(EMBEDDING_DIM)
    ).to(DEVICE)
else:
    print("ResNet-50 feature dimension matches target dimension. No projection layer added.")
    feature_extractor = image_model.to(DEVICE) # Just use the modified ResNet

feature_extractor.eval() # Set to evaluation mode

# Define Image Transformations required by ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5853591561317444, 0.5546306371688843, 0.5390808582305908], std=[0.33117732405662537, 0.33410581946372986, 0.34365367889404297]),
])

# --- Data Loading ---
print("Loading JSON data...")
try:
    train_df = pd.read_json(TARGET_TRAIN_FILE_PATH)
    test_df = pd.read_json(TARGET_TEST_FILE_PATH)
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")
    # Ensure required columns exist
    required_cols = ['sample_id', 'ocr_text', 'figurative_reasoning']
    if not all(col in train_df.columns for col in required_cols):
        raise ValueError(f"Training data missing one of required columns: {required_cols}")
    if not all(col in test_df.columns for col in required_cols):
        raise ValueError(f"Test data missing one of required columns: {required_cols}")

    # Handle potential NaN/None values in text columns (replace with empty string)
    train_df['ocr_text'].fillna('', inplace=True)
    train_df['figurative_reasoning'].fillna('', inplace=True)
    test_df['ocr_text'].fillna('', inplace=True)
    test_df['figurative_reasoning'].fillna('', inplace=True)

except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    print("Please ensure the JSON files exist at the specified paths:")
    print(f"Train: {TARGET_TRAIN_FILE_PATH}")
    print(f"Test: {TARGET_TEST_FILE_PATH}")
    sys.exit(1) # Exit if data cannot be loaded
except ValueError as e:
    print(f"Error in data format: {e}")
    sys.exit(1)


# --- Embedding Generation Functions ---

def generate_text_embeddings(model, texts, batch_size=32):
    """Generates embeddings for a list of texts using the Sentence Transformer."""
    print(f"Generating text embeddings for {len(texts)} items...")
    # Use model.encode for potentially large datasets, handles batching internally
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size, device=DEVICE)
    return embeddings.cpu() # Move to CPU for saving/general use

def generate_image_embeddings(model, image_paths, transform, df):
    """Generates embeddings for images specified by paths."""
    model.eval() # Ensure model is in eval mode
    embeddings = []
    print(f"Generating image embeddings for {len(df)} items...")
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        image_id = row['sample_id']
        # Try common image extensions if not specified or path is just ID
        possible_extensions = ['.jpg', '.png', '.jpeg']
        img_path = None
        for ext in possible_extensions:
            potential_path = os.path.join(image_paths, f"{image_id}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None:
             print(f"Warning: Image not found for sample_id {image_id} in {image_paths}. Skipping.")
             # Append a zero tensor or handle as needed. Appending zeros maintains order.
             embeddings.append(torch.zeros(EMBEDDING_DIM))
             continue # Skip to the next image

        try:
            img = Image.open(img_path).convert('RGB') # Ensure image is RGB
            img_t = transform(img).unsqueeze(0).to(DEVICE) # Add batch dim and move to device
            with torch.no_grad():
                embedding = model(img_t)
            embeddings.append(embedding.squeeze(0).cpu()) # Remove batch dim and move to CPU
        except Exception as e:
            print(f"Error processing image {img_path} for sample_id {image_id}: {e}. Skipping.")
            embeddings.append(torch.zeros(EMBEDDING_DIM)) # Append zero tensor on error

    if not embeddings: # Handle case where no images were processed
        return torch.empty((0, EMBEDDING_DIM))
    return torch.stack(embeddings)


# --- Generate and Save TRAIN Embeddings ---

print("\n--- Processing Training Data ---")
# 1. OCR Text Embeddings
train_ocr_texts = train_df['ocr_text'].tolist()
train_ocr_embeddings = generate_text_embeddings(text_model, train_ocr_texts)
print(f"Saving training OCR embeddings to {TRAIN_OCR_EMB_FILE}...")
torch.save(train_ocr_embeddings, TRAIN_OCR_EMB_FILE)
print(f"Shape: {train_ocr_embeddings.shape}") # Expected: [n_train, EMBEDDING_DIM]

# 2. Figurative Reasoning Embeddings
train_fig_texts = train_df['figurative_reasoning'].tolist()
train_figurative_embeddings = generate_text_embeddings(text_model, train_fig_texts)
print(f"Saving training Figurative Reasoning embeddings to {TRAIN_FIG_EMB_FILE}...")
torch.save(train_figurative_embeddings, TRAIN_FIG_EMB_FILE)
print(f"Shape: {train_figurative_embeddings.shape}") # Expected: [n_train, EMBEDDING_DIM]

# 3. Image Embeddings
train_image_embeddings = generate_image_embeddings(feature_extractor, TARGET_TRAIN_IMAGES_PATH, preprocess, train_df)
print(f"Saving training Image embeddings to {TRAIN_IMG_EMB_FILE}...")
torch.save(train_image_embeddings, TRAIN_IMG_EMB_FILE)
print(f"Shape: {train_image_embeddings.shape}") # Expected: [n_train, EMBEDDING_DIM]

# 4. Fuse Training Embeddings (Knowledge Fusion DB)
print("Fusing training embeddings...")
# Ensure all embeddings have the same number of samples (first dimension)
if not (train_ocr_embeddings.shape[0] == train_figurative_embeddings.shape[0] == train_image_embeddings.shape[0]):
    print("Error: Mismatch in the number of samples between OCR, Figurative, and Image embeddings for training data.")
    print(f"OCR: {train_ocr_embeddings.shape[0]}, Fig: {train_figurative_embeddings.shape[0]}, Img: {train_image_embeddings.shape[0]}")
    sys.exit(1)

# Concatenate along the feature dimension (dim=1)
train_fused_embeddings = torch.cat((train_ocr_embeddings, train_figurative_embeddings, train_image_embeddings), dim=1)
print(f"Saving fused training Knowledge Fusion DB to {TRAIN_FUSED_EMB_FILE}...")
torch.save(train_fused_embeddings, TRAIN_FUSED_EMB_FILE)
print(f"Shape: {train_fused_embeddings.shape}") # Expected: [n_train, 3 * EMBEDDING_DIM]

# --- Generate and Save TEST Embeddings (Individual) ---

print("\n--- Processing Test Data ---")
# 1. OCR Text Embeddings
test_ocr_texts = test_df['ocr_text'].tolist()
test_ocr_embeddings = generate_text_embeddings(text_model, test_ocr_texts)
print(f"Saving test OCR embeddings to {TEST_OCR_EMB_FILE}...")
torch.save(test_ocr_embeddings, TEST_OCR_EMB_FILE)
print(f"Shape: {test_ocr_embeddings.shape}") # Expected: [n_test, EMBEDDING_DIM]

# 2. Figurative Reasoning Embeddings
test_fig_texts = test_df['figurative_reasoning'].tolist()
test_figurative_embeddings = generate_text_embeddings(text_model, test_fig_texts)
print(f"Saving test Figurative Reasoning embeddings to {TEST_FIG_EMB_FILE}...")
torch.save(test_figurative_embeddings, TEST_FIG_EMB_FILE)
print(f"Shape: {test_figurative_embeddings.shape}") # Expected: [n_test, EMBEDDING_DIM]

# 3. Image Embeddings
test_image_embeddings = generate_image_embeddings(feature_extractor, TARGET_TEST_IMAGES_PATH, preprocess, test_df)
print(f"Saving test Image embeddings to {TEST_IMG_EMB_FILE}...")
torch.save(test_image_embeddings, TEST_IMG_EMB_FILE)
print(f"Shape: {test_image_embeddings.shape}") # Expected: [n_test, EMBEDDING_DIM]

# Note: Test embeddings are typically fused *during inference* when generating the query `e_k`
# Example of how you would fuse a single test instance (k):
# test_k_fused = torch.cat((test_ocr_embeddings[k], test_figurative_embeddings[k], test_image_embeddings[k]), dim=0) # Shape: [3 * EMBEDDING_DIM]

print("\n--- Embedding Generation Complete ---")
print(f"All embeddings saved in directory: {OUTPUT_DIR}")
print(f"Fused training database created at: {TRAIN_FUSED_EMB_FILE} with shape {train_fused_embeddings.shape}")