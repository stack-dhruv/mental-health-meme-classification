import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm # Use tqdm directly

# --- Configuration ---

# Set device (use GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Number of top similar examples to retrieve
# Note: For training set retrieval, we'll find K+1 and exclude self if needed.
TOP_K = 3

# DATASET DIRECTORY CONFIGURATION
try:
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # Adjust if needed
    DATASET_DIRECTORY = os.path.join(project_root, "dataset")
except NameError:
    print("Warning: __file__ not defined. Using current working directory structure.")
    relative_dataset_path = os.path.join("..", "..", "..", "dataset") # Adjust if CWD is different
    DATASET_DIRECTORY = os.path.abspath(relative_dataset_path)
    if not os.path.exists(DATASET_DIRECTORY):
        relative_dataset_path = os.path.join("dataset") # Try alternate structure
        DATASET_DIRECTORY = os.path.abspath(relative_dataset_path)

ANXIETY_DATASET_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Anxiety_Data")

# Original data files needed for sample IDs
TARGET_TRAIN_FILE_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "final", "cleaned", "anxiety_train_combined_preprocessed.json")
TARGET_TEST_FILE_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "final", "cleaned", "anxiety_test_combined_preprocessed.json")

# --- Embedding File Paths ---
# Make sure this matches the directory where the first script saved the embeddings
EMBEDDING_DIR = "anxiety_embeddings_output" # Adjust if you used a different name
os.makedirs(EMBEDDING_DIR, exist_ok=True) # Ensure directory exists

# Training Embeddings (Fused DB for searching, individuals for querying)
TRAIN_FUSED_EMB_FILE = os.path.join(EMBEDDING_DIR, "train_fused_knowledge_db.pt")
TRAIN_OCR_EMB_FILE = os.path.join(EMBEDDING_DIR, "train_ocr_embeddings.pt")
TRAIN_FIG_EMB_FILE = os.path.join(EMBEDDING_DIR, "train_figurative_reasoning_embeddings.pt")
TRAIN_IMG_EMB_FILE = os.path.join(EMBEDDING_DIR, "train_image_resnet50_embeddings.pt")

# Test Embeddings (individuals needed for querying)
TEST_OCR_EMB_FILE = os.path.join(EMBEDDING_DIR, "test_ocr_embeddings.pt")
TEST_FIG_EMB_FILE = os.path.join(EMBEDDING_DIR, "test_figurative_reasoning_embeddings.pt")
TEST_IMG_EMB_FILE = os.path.join(EMBEDDING_DIR, "test_image_resnet50_embeddings.pt")

# --- Output File Paths ---
# Save results in the embedding directory
TEST_OUTPUT_CSV_FILE = os.path.join(EMBEDDING_DIR, f"test_top_{TOP_K}_similar_indices.csv")
TEST_OUTPUT_JSON_FILE = os.path.join(EMBEDDING_DIR, f"test_top_{TOP_K}_similar_indices.json")
TRAIN_OUTPUT_CSV_FILE = os.path.join(EMBEDDING_DIR, f"train_top_{TOP_K}_similar_indices.csv")
TRAIN_OUTPUT_JSON_FILE = os.path.join(EMBEDDING_DIR, f"train_top_{TOP_K}_similar_indices.json")


# --- Similarity Finder Class ---

class SimilarityFinder:
    """
    Finds top K similar items from a database based on cosine similarity
    to a query embedding. Assumes database and queries are pre-computed.
    """
    def __init__(self, fused_database_tensor, device):
        """
        Initializes the finder with the fused training database.
        Args:
            fused_database_tensor (torch.Tensor): The N x 3D tensor of fused training embeddings.
            device: The torch device ('cuda' or 'cpu').
        """
        print(f"Loading database tensor with shape: {fused_database_tensor.shape}")
        self.database = fused_database_tensor.to(device)
        self.db_size = fused_database_tensor.shape[0]
        self.device = device
        print(f"Database loaded onto {self.device}.")

    def find_top_k(self, query_tensor, k=1, exclude_self_index=None):
        """
        Finds the indices of the top K most similar items in the database.

        Args:
            query_tensor (torch.Tensor): The 1 x 3D tensor for the query item.
            k (int): The number of top similar items to return.
            exclude_self_index (int, optional): If provided, this index will be
                                                 excluded from the top K results.
                                                 Finds k+1 and removes self if it's top.

        Returns:
            list: A list of the top K indices from the database.
        """
        if query_tensor.ndim == 1:
            query_tensor = query_tensor.unsqueeze(0) # Add batch dim

        query_tensor = query_tensor.to(self.device) # Ensure query is on the same device

        k_to_find = k
        if exclude_self_index is not None:
             # Find one extra in case the top match is the item itself
             k_to_find = min(k + 1, self.db_size) # Cannot find more than db size

        # Calculate cosine similarity
        similarities = F.cosine_similarity(query_tensor, self.database, dim=1)

        # Get the top K (or K+1) scores and their indices
        top_scores, top_indices = torch.topk(similarities, k=k_to_find)

        # Convert to list
        top_indices_list = top_indices.cpu().tolist()

        # Handle self-exclusion if necessary
        if exclude_self_index is not None:
            if exclude_self_index in top_indices_list:
                top_indices_list.remove(exclude_self_index)
            # Return the top K from the remaining list
            return top_indices_list[:k]
        else:
            # Return the original top K
            return top_indices_list

# --- Data Processing Function ---

def process_dataset(finder, ocr_embeddings, fig_embeddings, img_embeddings, metadata_df, k, is_training_set, desc):
    """
    Processes a dataset (train or test) to find top K similar indices for each sample.

    Args:
        finder (SimilarityFinder): The initialized similarity finder object.
        ocr_embeddings (torch.Tensor): OCR embeddings for the dataset.
        fig_embeddings (torch.Tensor): Figurative embeddings for the dataset.
        img_embeddings (torch.Tensor): Image embeddings for the dataset.
        metadata_df (pd.DataFrame): DataFrame with metadata (must include 'sample_id').
        k (int): Number of top similar indices to find.
        is_training_set (bool): True if processing the training set (for self-exclusion).
        desc (str): Description for the tqdm progress bar.

    Returns:
        list: A list of dictionaries, each containing 'sample_id' and 'top_k_indices'.
    """
    num_samples = len(metadata_df)
    results = []
    print(f"\nRetrieving top {k} similar training indices for {num_samples} samples in {desc}...")

    # Dimension check before loop
    expected_dim = finder.database.shape[1]
    query_dim = ocr_embeddings.shape[1] + fig_embeddings.shape[1] + img_embeddings.shape[1]
    if expected_dim != query_dim:
         raise ValueError(f"Dimension mismatch! Database: {expected_dim}, Query (concatenated): {query_dim}")

    for i in tqdm(range(num_samples), desc=desc):
        query_ocr = ocr_embeddings[i]
        query_fig = fig_embeddings[i]
        query_img = img_embeddings[i]

        # Fuse to create the query vector e_k
        query_fused = torch.cat((query_ocr.flatten(), query_fig.flatten(), query_img.flatten()), dim=0)

        # Find top K similar indices, excluding self if it's the training set
        exclude_index = i if is_training_set else None
        top_indices = finder.find_top_k(query_fused, k=k, exclude_self_index=exclude_index)

        # Store result
        sample_id = metadata_df['sample_id'].iloc[i]
        results.append({
            "sample_id": sample_id, # Use 'sample_id' consistently
            f"top_{k}_train_indices": top_indices
        })
    return results

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Data and Embeddings
    print("Loading embeddings and data...")
    try:
        # Load fused training database
        train_fused_db = torch.load(TRAIN_FUSED_EMB_FILE, map_location=DEVICE)

        # Load individual training embeddings
        train_ocr_embeddings = torch.load(TRAIN_OCR_EMB_FILE, map_location=DEVICE)
        train_figurative_embeddings = torch.load(TRAIN_FIG_EMB_FILE, map_location=DEVICE)
        train_image_embeddings = torch.load(TRAIN_IMG_EMB_FILE, map_location=DEVICE)

        # Load individual test embeddings
        test_ocr_embeddings = torch.load(TEST_OCR_EMB_FILE, map_location=DEVICE)
        test_figurative_embeddings = torch.load(TEST_FIG_EMB_FILE, map_location=DEVICE)
        test_image_embeddings = torch.load(TEST_IMG_EMB_FILE, map_location=DEVICE)

        # Load original metadata files
        train_df = pd.read_json(TARGET_TRAIN_FILE_PATH)
        test_df = pd.read_json(TARGET_TEST_FILE_PATH)
        print(f"Loaded {len(train_df)} train and {len(test_df)} test samples metadata.")

    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        print("Please ensure the embedding generation script ran successfully and saved embeddings to:")
        print(f"  Embed Dir: {EMBEDDING_DIR}")
        print(f"Required files: {os.path.basename(TRAIN_FUSED_EMB_FILE)}, {os.path.basename(TRAIN_OCR_EMB_FILE)}, ..., {os.path.basename(TARGET_TEST_FILE_PATH)}")
        sys.exit(1)

    # Basic count checks
    if not (len(train_ocr_embeddings) == len(train_figurative_embeddings) == len(train_image_embeddings) == len(train_df)):
        print("Error: Mismatch in sample counts for training data/embeddings.")
        sys.exit(1)
    if not (len(test_ocr_embeddings) == len(test_figurative_embeddings) == len(test_image_embeddings) == len(test_df)):
        print("Error: Mismatch in sample counts for test data/embeddings.")
        sys.exit(1)

    # 2. Initialize Similarity Finder
    print("Initializing similarity finder...")
    finder = SimilarityFinder(train_fused_db, DEVICE)

    # 3. Process Test Set
    test_results = process_dataset(
        finder=finder,
        ocr_embeddings=test_ocr_embeddings,
        fig_embeddings=test_figurative_embeddings,
        img_embeddings=test_image_embeddings,
        metadata_df=test_df,
        k=TOP_K,
        is_training_set=False,
        desc="Processing Test Set"
    )

    # 4. Process Training Set
    train_results = process_dataset(
        finder=finder,
        ocr_embeddings=train_ocr_embeddings,
        fig_embeddings=train_figurative_embeddings,
        img_embeddings=train_image_embeddings,
        metadata_df=train_df,
        k=TOP_K,
        is_training_set=True, # Exclude self from results
        desc="Processing Train Set"
    )

    # 5. Save Results
    print("\nSaving results...")

    # Save Test Results
    test_results_df = pd.DataFrame(test_results)
    try:
        test_results_df.to_csv(TEST_OUTPUT_CSV_FILE, index=False)
        test_results_df.to_json(TEST_OUTPUT_JSON_FILE, orient='records', indent=4)
        print(f"Test results saved successfully to:")
        print(f"  CSV: {TEST_OUTPUT_CSV_FILE}")
        print(f"  JSON: {TEST_OUTPUT_JSON_FILE}")
    except Exception as e:
        print(f"Error saving test results: {e}")

    # Save Train Results
    train_results_df = pd.DataFrame(train_results)
    try:
        train_results_df.to_csv(TRAIN_OUTPUT_CSV_FILE, index=False)
        train_results_df.to_json(TRAIN_OUTPUT_JSON_FILE, orient='records', indent=4)
        print(f"Train results saved successfully to:")
        print(f"  CSV: {TRAIN_OUTPUT_CSV_FILE}")
        print(f"  JSON: {TRAIN_OUTPUT_JSON_FILE}")
    except Exception as e:
        print(f"Error saving train results: {e}")

    print("\n--- Retrieval Complete ---")