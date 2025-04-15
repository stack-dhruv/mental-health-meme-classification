import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import pandas as pd
import torch
# Use standard tqdm, not from sentence_transformers
from tqdm import tqdm
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
    script_dir = os.path.dirname(__file__)
    # Adjust based on your script's location relative to the project root
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..")) # Example: if script is in project_root/scripts/something
    if not os.path.exists(os.path.join(project_root, 'dataset')): # Basic check
        # Try alternative if the above is wrong
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # Example: if script is in project_root/src/scripts/something

    DATASET_DIRECTORY = os.path.join(project_root, "dataset")
    print(f"Attempting to use DATASET_DIRECTORY: {DATASET_DIRECTORY}")
    if not os.path.exists(DATASET_DIRECTORY):
         print(f"ERROR: Cannot find dataset directory at {DATASET_DIRECTORY}. Please check paths.")
         sys.exit(1)

except NameError:
    print("Warning: __file__ not defined. Using hardcoded relative paths from CWD.")
    # Assume CWD is project root for this fallback
    project_root = os.path.abspath(".")
    DATASET_DIRECTORY = os.path.join(project_root, "dataset")
    if not os.path.exists(DATASET_DIRECTORY):
        print(f"ERROR: Cannot find dataset directory at {DATASET_DIRECTORY} relative to CWD.")
        sys.exit(1)


DEPRESSION_DATASET_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Depressive_Data")
# Verify the existence of the base directory
if not os.path.exists(DEPRESSION_DATASET_DIRECTORY):
    print(f"ERROR: Depression dataset directory not found at {DEPRESSION_DATASET_DIRECTORY}")
    sys.exit(1)


# --- Input File Paths ---
TARGET_TRAIN_FILE_PATH = os.path.join(DEPRESSION_DATASET_DIRECTORY, "final", "cleaned", "depressive_train_combined_preprocessed.json")
TARGET_TEST_FILE_PATH = os.path.join(DEPRESSION_DATASET_DIRECTORY, "final", "cleaned", "depressive_test_combined_preprocessed.json")
TARGET_VAL_FILE_PATH = os.path.join(DEPRESSION_DATASET_DIRECTORY, "final", "cleaned", "depressive_val_combined_preprocessed.json") # Corrected variable name

# --- Image Paths ---
# Adjust these paths carefully based on your EXACT directory structure
IMAGES_BASE_PATH = os.path.join(DEPRESSION_DATASET_DIRECTORY, "Images", "depressive_image") # Base path for images
if not os.path.exists(IMAGES_BASE_PATH):
     # Try the potentially nested path from the original script if the simpler one fails
     IMAGES_BASE_PATH = os.path.join(DEPRESSION_DATASET_DIRECTORY, "Depressive_Data", "Images")
     print(f"Warning: Simple image path not found, trying nested path: {IMAGES_BASE_PATH}")
     if not os.path.exists(IMAGES_BASE_PATH):
          print(f"ERROR: Image base directory not found at {IMAGES_BASE_PATH} or the simpler path.")
          sys.exit(1)

TARGET_TRAIN_IMAGES_PATH = os.path.join(IMAGES_BASE_PATH, "train")
TARGET_TEST_IMAGES_PATH = os.path.join(IMAGES_BASE_PATH, "test")
TARGET_VAL_IMAGES_PATH = os.path.join(IMAGES_BASE_PATH, "val") # Added validation image path


# --- Embedding Configuration ---
TEXT_MODEL_NAME = 'BAAI/bge-m3'
EMBEDDING_DIM = 1024 # Target dimension 'd' for each modality


# --- Output Configuration ---
OUTPUT_DIR = "depressive_embeddings_output" # Changed output directory name
# Ensure output directory exists relative to project root or CWD
OUTPUT_DIR_PATH = os.path.join(project_root, OUTPUT_DIR) # Place it in the project root
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
print(f"Output embeddings will be saved in: {OUTPUT_DIR_PATH}")

# Define output file names within the output directory
TRAIN_OCR_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "train_ocr_embeddings.pt")
TRAIN_FIG_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "train_figurative_reasoning_embeddings.pt")
TRAIN_IMG_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "train_image_resnet50_embeddings.pt")
TRAIN_FUSED_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "train_fused_knowledge_db.pt") # n x 3d

TEST_OCR_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "test_ocr_embeddings.pt")
TEST_FIG_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "test_figurative_reasoning_embeddings.pt")
TEST_IMG_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "test_image_resnet50_embeddings.pt")

VAL_OCR_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "val_ocr_embeddings.pt") # Added Val output files
VAL_FIG_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "val_figurative_reasoning_embeddings.pt")
VAL_IMG_EMB_FILE = os.path.join(OUTPUT_DIR_PATH, "val_image_resnet50_embeddings.pt")


# --- Model Initialization ---

# 1. Sentence Transformer
print(f"Loading Sentence Transformer: {TEXT_MODEL_NAME}...")
text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
actual_text_dim = text_model.get_sentence_embedding_dimension()
if actual_text_dim != EMBEDDING_DIM:
     print(f"Warning: Text model {TEXT_MODEL_NAME} outputs {actual_text_dim} dimensions, not {EMBEDDING_DIM}. Using {actual_text_dim}.")
     # EMBEDDING_DIM = actual_text_dim # Adjust if needed
else:
    print(f"Text model loaded. Output dimension: {actual_text_dim}")


# 2. Image Feature Extractor (ResNet-50 modified)
print("Loading and modifying ResNet-50...")
image_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Identity()

if num_ftrs != EMBEDDING_DIM:
    print(f"Adding projection layer to ResNet-50: {num_ftrs} -> {EMBEDDING_DIM}")
    feature_extractor = nn.Sequential(
        image_model,
        nn.Linear(num_ftrs, EMBEDDING_DIM),
    ).to(DEVICE)
else:
    print("ResNet-50 feature dimension matches target. No projection layer added.")
    feature_extractor = image_model.to(DEVICE)
feature_extractor.eval()

# Define Image Transformations (Using the stats from your provided script)
# Ensure these stats are correct for the Depressive dataset
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # Added CenterCrop, common practice
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5853591561317444, 0.5546306371688843, 0.5390808582305908],
                         std=[0.33117732405662537, 0.33410581946372986, 0.34365367889404297]),
])
print("Image preprocessing pipeline defined.")

# --- Data Loading ---
print("Loading JSON data for train, test, and validation sets...")
try:
    train_df = pd.read_json(TARGET_TRAIN_FILE_PATH)
    test_df = pd.read_json(TARGET_TEST_FILE_PATH)
    val_df = pd.read_json(TARGET_VAL_FILE_PATH) # Load validation data
    print(f"Loaded {len(train_df)} train, {len(test_df)} test, and {len(val_df)} validation samples.")

    required_cols = ['sample_id', 'ocr_text', 'figurative_reasoning']
    # Check required columns for all splits
    if not all(col in train_df.columns for col in required_cols):
        raise ValueError(f"Training data missing required columns: {required_cols}")
    if not all(col in test_df.columns for col in required_cols):
        raise ValueError(f"Test data missing required columns: {required_cols}")
    if not all(col in val_df.columns for col in required_cols):
        raise ValueError(f"Validation data missing required columns: {required_cols}")

    # Handle potential NaN/None values in text columns for all splits
    for df in [train_df, test_df, val_df]:
        df['ocr_text'].fillna('', inplace=True)
        df['figurative_reasoning'].fillna('', inplace=True)

except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    print("Please ensure the JSON files exist at the specified paths:")
    print(f"  Train: {TARGET_TRAIN_FILE_PATH}")
    print(f"  Test: {TARGET_TEST_FILE_PATH}")
    print(f"  Val: {TARGET_VAL_FILE_PATH}")
    sys.exit(1)
except ValueError as e:
    print(f"Error in data format: {e}")
    sys.exit(1)


# --- Embedding Generation Functions ---

def generate_text_embeddings(model, texts, batch_size=64, desc=""): # Increased batch size
    """Generates embeddings for a list of texts using the Sentence Transformer."""
    print(f"Generating text embeddings for {len(texts)} items ({desc})...")
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,
        device=DEVICE
    )
    return embeddings.cpu()

def generate_image_embeddings(feature_extractor_model, image_paths_base, transform_pipeline, dataframe, desc=""):
    """Generates embeddings for images specified by paths."""
    feature_extractor_model.eval()
    embeddings = []
    print(f"Generating image embeddings for {len(dataframe)} items ({desc})...")
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Processing Images ({desc})"):
        image_id = row['sample_id']
        # Construct path relative to the specific split's image base
        possible_extensions = ['.jpg', '.png', '.jpeg']
        img_path = None
        # Try to find the image file with common extensions
        for ext in possible_extensions:
            # Use the specific image path base for train/test/val
            potential_path = os.path.join(image_paths_base, f"{image_id}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
            # Try without prefix if sample_id already includes it (less common)
            potential_path_no_prefix = os.path.join(image_paths_base, f"{image_id.split('-')[-1]}{ext}")
            if image_id.count('-') > 0 and os.path.exists(potential_path_no_prefix):
                img_path = potential_path_no_prefix
                # print(f"Debug: Found image using non-prefixed ID: {img_path}") # Optional debug
                break


        if img_path is None:
             # Be more specific about the path being checked
             print(f"Warning: Image not found for sample_id {image_id} in {image_paths_base} (tried extensions: {possible_extensions}). Skipping.")
             embeddings.append(torch.zeros(EMBEDDING_DIM)) # Use consistent dimension
             continue

        try:
            img = Image.open(img_path).convert('RGB')
            img_t = transform_pipeline(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = feature_extractor_model(img_t)
            embeddings.append(embedding.squeeze(0).cpu())
        except Exception as e:
            print(f"Error processing image {img_path} (ID: {image_id}): {e}. Appending zeros.")
            embeddings.append(torch.zeros(EMBEDDING_DIM))

    if not embeddings: return torch.empty((0, EMBEDDING_DIM))
    # Check if all embeddings are zeros (indicates widespread image loading failure)
    if torch.stack(embeddings).eq(0).all():
        print(f"WARNING: All generated image embeddings for '{desc}' are zero. Please check image paths and loading process.")
        # print(f"Example paths checked: {os.path.join(image_paths_base, f'{dataframe['sample_id'].iloc[0]}.jpg')}, etc.")
    return torch.stack(embeddings)


# --- Process and Save Function ---
def process_and_save_split(df, split_name, image_path_base, ocr_out_file, fig_out_file, img_out_file):
    """Helper function to process one data split (train, test, or val)."""
    print(f"\n--- Processing {split_name.capitalize()} Data ({len(df)} samples) ---")

    # 1. OCR Text Embeddings
    if os.path.exists(ocr_out_file):
        print(f"{split_name.capitalize()} OCR embeddings already exist. Loading from {ocr_out_file}...")
        ocr_embeddings = torch.load(ocr_out_file)
    else:
        ocr_texts = df['ocr_text'].tolist()
        ocr_embeddings = generate_text_embeddings(text_model, ocr_texts, desc=f"{split_name} OCR")
        print(f"Saving {split_name} OCR embeddings to {ocr_out_file}...")
        torch.save(ocr_embeddings, ocr_out_file)
    print(f"Shape: {ocr_embeddings.shape}")

    # 2. Figurative Reasoning Embeddings
    if os.path.exists(fig_out_file):
        print(f"{split_name.capitalize()} Figurative Reasoning embeddings already exist. Loading from {fig_out_file}...")
        fig_embeddings = torch.load(fig_out_file)
    else:
        fig_texts = df['figurative_reasoning'].tolist()
        fig_embeddings = generate_text_embeddings(text_model, fig_texts, desc=f"{split_name} Figurative")
        print(f"Saving {split_name} Figurative Reasoning embeddings to {fig_out_file}...")
        torch.save(fig_embeddings, fig_out_file)
    print(f"Shape: {fig_embeddings.shape}")

    # 3. Image Embeddings
    if os.path.exists(img_out_file):
        print(f"{split_name.capitalize()} Image embeddings already exist. Loading from {img_out_file}...")
        img_embeddings = torch.load(img_out_file)
    else:
        img_embeddings = generate_image_embeddings(feature_extractor, image_path_base, preprocess, df, desc=split_name)
        print(f"Saving {split_name} Image embeddings to {img_out_file}...")
        torch.save(img_embeddings, img_out_file)
    print(f"Shape: {img_embeddings.shape}")

    return ocr_embeddings, fig_embeddings, img_embeddings


# --- Main Processing ---

# Process Training Data
train_ocr_embeddings, train_fig_embeddings, train_image_embeddings = process_and_save_split(
    train_df, "train", TARGET_TRAIN_IMAGES_PATH, TRAIN_OCR_EMB_FILE, TRAIN_FIG_EMB_FILE, TRAIN_IMG_EMB_FILE
)

# Process Test Data
process_and_save_split(
    test_df, "test", TARGET_TEST_IMAGES_PATH, TEST_OCR_EMB_FILE, TEST_FIG_EMB_FILE, TEST_IMG_EMB_FILE
)

# Process Validation Data
process_and_save_split(
    val_df, "val", TARGET_VAL_IMAGES_PATH, VAL_OCR_EMB_FILE, VAL_FIG_EMB_FILE, VAL_IMG_EMB_FILE
)

# --- Fuse Training Embeddings (Only for Training Set) ---
print("\n--- Fusing Training Embeddings for RAG DB ---")
# Ensure all training embeddings have the same number of samples
if not (train_ocr_embeddings.shape[0] == train_fig_embeddings.shape[0] == train_image_embeddings.shape[0]):
    print("Error: Mismatch in the number of samples between OCR, Figurative, and Image embeddings for training data.")
    print(f"OCR: {train_ocr_embeddings.shape[0]}, Fig: {train_fig_embeddings.shape[0]}, Img: {train_image_embeddings.shape[0]}")
    sys.exit(1)

# Concatenate along the feature dimension (dim=1)
train_fused_embeddings = torch.cat((train_ocr_embeddings, train_fig_embeddings, train_image_embeddings), dim=1)
print(f"Saving fused training Knowledge Fusion DB to {TRAIN_FUSED_EMB_FILE}...")
torch.save(train_fused_embeddings, TRAIN_FUSED_EMB_FILE)
print(f"Shape: {train_fused_embeddings.shape}") # Expected: [n_train, 3 * EMBEDDING_DIM]


# --- Final Summary ---
print("\n--- Embedding Generation Complete for Depressive Dataset ---")
print(f"All individual embeddings (train, test, val) saved in directory: {OUTPUT_DIR_PATH}")
print(f"Fused training database created at: {TRAIN_FUSED_EMB_FILE} with shape {train_fused_embeddings.shape}")