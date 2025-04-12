import os
import json
import torch
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import logging
from tqdm import tqdm

# Import the necessary functions from the main file
from anxiety.training.ocr_bert import (
    AnxietyClassifier, AnxietyDataset, evaluate_model, load_data, set_seed, MAX_LEN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_classification_report(report_str):
    """
    Parse the classification report string into a structured format
    
    Args:
        report_str (str): Classification report string from sklearn
        
    Returns:
        list: List of dictionaries containing the parsed data
    """
    lines = report_str.strip().split('\n')
    rows = []
    
    # Skip the header row
    for line in lines[2:]:  # Skip header rows
        if line.strip() and not line.startswith('accuracy') and not line.startswith('macro') and not line.startswith('weighted'):
            # More robust parsing method
            parts = line.strip().split(maxsplit=4)
            if len(parts) >= 5:  # Ensure we have class, precision, recall, f1, support
                class_name = parts[0]
                
                # Try to convert values to proper types
                try:
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1_score = float(parts[3])
                    # Support is usually an integer wrapped in parentheses at the end
                    support_str = parts[4]
                    # Extract the number from possible formats like "123" or "(123)"
                    support = int(''.join(c for c in support_str if c.isdigit()))
                    
                    rows.append({
                        'class': class_name,
                        'precision': precision,
                        'recall': recall,
                        'f1-score': f1_score,
                        'support': support
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line '{line}': {str(e)}")
    
    return rows

def test_saved_model(model_path, test_file, batch_size=16):
    """
    Test a saved anxiety classification model on a test dataset.
    
    Args:
        model_path (str): Path to the saved model state dict
        test_file (str): Path to the test data JSON file
        batch_size (int): Batch size for evaluation
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data
    logger.info(f"Loading test data from {test_file}")
    test_data = load_data(test_file)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Extract all unique labels from the test data
    test_labels = [sample['meme_anxiety_category'] for sample in test_data]
    
    # Create label encoder and fit it to the test labels
    label_encoder = LabelEncoder()
    label_encoder.fit(test_labels)
    num_labels = len(label_encoder.classes_)
    logger.info(f"Number of labels in test data: {num_labels}")
    logger.info(f"Labels: {label_encoder.classes_}")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = AnxietyClassifier(num_labels=num_labels)
    
    # Load saved model weights
    logger.info(f"Loading model weights from {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None
    
    # Create test dataset and dataloader
    test_dataset = AnxietyDataset(test_data, tokenizer, MAX_LEN, label_encoder)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Evaluate model
    logger.info("Evaluating model on test data")
    test_metrics = evaluate_model(model, test_dataloader, device, label_encoder)
    
    # Log metrics
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
    logger.info(f"Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    logger.info(f"Test Report:\n{test_metrics['report']}")
    
    # Save the test report to a TXT file
    output_dir = os.path.dirname(model_path)
    txt_path = os.path.join(output_dir, "test_report.txt")
    logger.info(f"Saving test report to {txt_path}")
    
    try:
        # Save to TXT
        with open(txt_path, 'w') as txtfile:
            txtfile.write(f"Test Metrics for Anxiety Classification Model\n")
            txtfile.write(f"==========================================\n\n")
            txtfile.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
            txtfile.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
            txtfile.write(f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
            txtfile.write(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}\n\n")
            txtfile.write(f"Detailed Classification Report:\n")
            txtfile.write(f"-----------------------------\n\n")
            txtfile.write(test_metrics['report'])
            
        logger.info(f"Test report saved successfully to {txt_path}")
    except Exception as e:
        logger.error(f"Error saving test report to TXT: {str(e)}")
    
    return test_metrics

if __name__ == "__main__":
    # Set paths for model and test data
    base_dir = os.path.join("mental-health-meme-classification")
    model_path = os.path.join(base_dir, "src", "anxiety", "output", "best_model.pt")
    test_file = os.path.join(base_dir, "dataset", "Anxiety_Data", "anxiety_test.json")
    
    # Test the saved model
    test_metrics = test_saved_model(model_path, test_file)
    
    # Inform user about the saved report
    if test_metrics:
        output_dir = os.path.join(base_dir, "src", "anxiety", "output")
        print(f"Test report saved to {os.path.join(output_dir, 'test_report.txt')}")
