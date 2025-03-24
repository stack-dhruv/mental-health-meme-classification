import os
import torch
import json
import numpy as np
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import sys
import os

# Add the `src` directory to the Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.depression.ocr_bert import DepressionDataset, DepressionClassifier

def evaluate_model(model_path, test_data_path, categories, max_len, batch_size, device):
    """
    Evaluates the trained model on the test dataset and calculates metrics.

    Args:
        model_path (str): Path to the trained model file.
        test_data_path (str): Path to the test dataset JSON file.
        categories (list): List of depressive categories.
        max_len (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for DataLoader.
        device (torch.device): Device to run the evaluation on (CPU or GPU).

    Returns:
        dict: Dictionary containing evaluation metrics (macro_f1, weighted_f1, hamming_loss).
    """
    # Load the test dataset
    with open(test_data_path, 'r') as test_file:
        test_data = json.load(test_file)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_dataset = DepressionDataset(test_data, tokenizer, max_len, categories)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Load the trained model
    model = DepressionClassifier(num_labels=len(categories))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Initialize variables for evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Convert predictions to binary (threshold = 0.5)
    all_preds = (np.array(all_preds) > 0.5).astype(int)
    all_labels = np.array(all_labels)

    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    hamming = hamming_loss(all_labels, all_preds)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hamming_loss": hamming,
    }

if __name__ == "__main__":
    # Configuration
    MAX_LEN = 512
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CATEGORIES = [
        "Lack of Interest",
        "Self-Harm",
        "Feeling Down",
        "Concentration Problem",
        "Lack of Energy",
        "Sleeping Disorder",
        "Low Self-Esteem",
        "Eating Disorder",
    ]

    # Paths
    MODEL_PATH = "/media/nas_mount/avinash_ocr/Dhruvkumar_Patel_MT24032/mental-health-meme-classification/src/depression/output/ocr_bert/best_model.pt"
    TEST_DATA_PATH = "/media/nas_mount/avinash_ocr/Dhruvkumar_Patel_MT24032/mental-health-meme-classification/dataset/Depressive_Data/test.json"

    # Evaluate the model
    metrics = evaluate_model(MODEL_PATH, TEST_DATA_PATH, CATEGORIES, MAX_LEN, BATCH_SIZE, DEVICE)

    # Print the results
    print("Evaluation Metrics:")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")