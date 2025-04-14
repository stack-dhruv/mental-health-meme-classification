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
TOP_K = 3 # As used in the original test loop example

# DATASET DIRECTORY CONFIGURATION (Assuming structure from previous script)
try:
    script_dir = os.path.dirname(__file__)
    # Adjust path relative to *this* script's location
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # Adjust if needed
    DATASET_DIRECTORY = os.path.join(project_root, "dataset")
except NameError:
    print("Warning: __file__ not defined. Using current working directory structure.")
    # Adjust these relative paths based on your CWD if running interactively
    relative_dataset_path = os.path.join("..", "..", "..", "dataset")
    DATASET_DIRECTORY = os.path.abspath(relative_dataset_path)
    if not os.path.exists(DATASET_DIRECTORY):
        relative_dataset_path = os.path.join("dataset") # Try alternate structure
        DATASET_DIRECTORY = os.path.abspath(relative_dataset_path)

ANXIETY_DATASET_DIRECTORY = os.path.join(DATASET_DIRECTORY, "Anxiety_Data")
# Original data file needed for sample IDs
TARGET_TEST_FILE_PATH = os.path.join(ANXIETY_DATASET_DIRECTORY, "final", "cleaned", "anxiety_test_combined_preprocessed.json")

# --- Embedding File Paths (Load from the directory created previously) ---
# Make sure this matches the directory where the first script saved the embeddings
EMBEDDING_DIR = "anxiety_embeddings_output" # Or "anxiety_embeddings_output" if you used that

# Training Embeddings (only the fused DB is needed for searching)
TRAIN_FUSED_EMB_FILE = os.path.join(EMBEDDING_DIR, "train_fused_knowledge_db.pt") # n x 3d

# Test Embeddings (individual components needed to build the query)
TEST_OCR_EMB_FILE = os.path.join(EMBEDDING_DIR, "test_ocr_embeddings.pt")
TEST_FIG_EMB_FILE = os.path.join(EMBEDDING_DIR, "test_figurative_reasoning_embeddings.pt")
TEST_IMG_EMB_FILE = os.path.join(EMBEDDING_DIR, "test_image_resnet50_embeddings.pt")

# Output File Path (Save results in the embedding directory)
OUTPUT_CSV_FILE = os.path.join(EMBEDDING_DIR, f"test_top_{TOP_K}_similar_indices.csv")
OUTPUT_JSON_FILE = os.path.join(EMBEDDING_DIR, f"test_top_{TOP_K}_similar_indices.json") # Also save as JSON

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
        # Ensure database is on the correct device
        self.database = fused_database_tensor.to(device)
        self.device = device
        print(f"Database loaded onto {self.device}.")

    def find_top_k(self, query_tensor, k=1):
        """
        Finds the indices of the top K most similar items in the database.

        Args:
            query_tensor (torch.Tensor): The 1 x 3D tensor for the query item.
            k (int): The number of top similar items to return.

        Returns:
            list: A list of the top K indices from the database.
        """
        if query_tensor.ndim == 1:
            # Ensure query has a batch dimension for cosine_similarity
            query_tensor = query_tensor.unsqueeze(0)

        query_tensor = query_tensor.to(self.device) # Ensure query is on the same device

        # Calculate cosine similarity between the query and all database entries
        # query_tensor shape: [1, 3D], self.database shape: [N, 3D]
        # Resulting similarities shape: [N]
        similarities = F.cosine_similarity(query_tensor, self.database, dim=1)

        # Get the top K scores and their indices
        # `torch.topk` is efficient
        top_k_scores, top_k_indices = torch.topk(similarities, k=k)

        # Return indices as a standard Python list (moved to CPU if needed)
        return top_k_indices.cpu().tolist()

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Data and Embeddings
    print("Loading embeddings and data...")
    try:
        # Load the fused training database (E from the paper)
        train_fused_db = torch.load(TRAIN_FUSED_EMB_FILE, map_location=DEVICE)

        # Load individual test embeddings
        test_ocr_embeddings = torch.load(TEST_OCR_EMB_FILE, map_location=DEVICE)
        test_figurative_embeddings = torch.load(TEST_FIG_EMB_FILE, map_location=DEVICE)
        test_image_embeddings = torch.load(TEST_IMG_EMB_FILE, map_location=DEVICE)

        # Load the original test data file to get sample IDs
        test_df = pd.read_json(TARGET_TEST_FILE_PATH)
        print(f"Loaded {len(test_df)} test samples metadata.")

    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        print("Please ensure the previous script ran successfully and saved embeddings to:")
        print(f"  Fused DB: {TRAIN_FUSED_EMB_FILE}")
        print(f"  Test OCR: {TEST_OCR_EMB_FILE}")
        print(f"  Test Fig: {TEST_FIG_EMB_FILE}")
        print(f"  Test Img: {TEST_IMG_EMB_FILE}")
        print(f"  Test Meta: {TARGET_TEST_FILE_PATH}")
        sys.exit(1)

    # Basic dimension check
    num_test_samples = len(test_df)
    if not (len(test_ocr_embeddings) == num_test_samples and
            len(test_figurative_embeddings) == num_test_samples and
            len(test_image_embeddings) == num_test_samples):
        print("Error: Mismatch between number of test samples in JSON and loaded embedding files.")
        print(f"  JSON samples: {num_test_samples}")
        print(f"  OCR embeds: {len(test_ocr_embeddings)}")
        print(f"  Fig embeds: {len(test_figurative_embeddings)}")
        print(f"  Img embeds: {len(test_image_embeddings)}")
        sys.exit(1)

    if train_fused_db.shape[1] != (test_ocr_embeddings.shape[1] +
                                   test_figurative_embeddings.shape[1] +
                                   test_image_embeddings.shape[1]):
         print("Error: Dimension mismatch between fused training DB and concatenated test embeddings.")
         print(f"  Train DB dim: {train_fused_db.shape[1]}")
         print(f"  Test concat dim: {test_ocr_embeddings.shape[1] + test_figurative_embeddings.shape[1] + test_image_embeddings.shape[1]}")
         sys.exit(1)


    # 2. Initialize Similarity Finder
    print("Initializing similarity finder...")
    finder = SimilarityFinder(train_fused_db, DEVICE)

    # 3. Perform Retrieval for each Test Sample
    print(f"Retrieving top {TOP_K} similar training indices for {num_test_samples} test samples...")
    results = []
    for i in tqdm(range(num_test_samples), desc="Finding Similar Memes"):
        # Get individual embeddings for the i-th test sample
        query_ocr = test_ocr_embeddings[i]
        query_fig = test_figurative_embeddings[i]
        query_img = test_image_embeddings[i]

        # Fuse them to create the query vector e_k
        # Ensure they are treated as 1D vectors before cat
        query_fused = torch.cat((query_ocr.flatten(), query_fig.flatten(), query_img.flatten()), dim=0)

        # Find the top K similar indices from the training database
        top_indices = finder.find_top_k(query_fused, k=TOP_K)

        # Store the result (e.g., sample ID and the list of indices)
        sample_id = test_df['sample_id'].iloc[i] # Get corresponding sample ID
        results.append({
            "test_sample_id": sample_id,
            f"top_{TOP_K}_train_indices": top_indices
        })

    # 4. Save Results
    print("Saving results...")

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Save as CSV
    try:
        results_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Results saved successfully to CSV: {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"Error saving results to CSV {OUTPUT_CSV_FILE}: {e}")

    # Save as JSON
    try:
        results_df.to_json(OUTPUT_JSON_FILE, orient='records', indent=4)
        print(f"Results saved successfully to JSON: {OUTPUT_JSON_FILE}")
    except Exception as e:
        print(f"Error saving results to JSON {OUTPUT_JSON_FILE}: {e}")

    print("\n--- Retrieval Complete ---")