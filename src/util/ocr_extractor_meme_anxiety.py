# this notebook is used for extracting the ocr text of the images in the anxiety dataset
import easyocr
import os
import json
import csv
from pathlib import Path
from tqdm import tqdm  # Import tqdm for progress bar

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
reader = easyocr.Reader(['en'])

def parse_ocr_result(ocr_result):
    """
    Parse OCR result and return concatenated text.
    
    Args:
        ocr_result: List of tuples, each containing coordinates, text, and confidence score
    
    Returns:
        str: Combined text from all OCR detections
    """
    if not ocr_result:
        return ""
    
    # Sort by y-coordinate (vertical position) of the top-left point
    sorted_result = sorted(ocr_result, key=lambda x: x[0][0][1])
    
    # Group text elements that are on the same line (similar y-coordinates)
    lines = []
    current_line = [sorted_result[0]]
    y_threshold = 20  # Adjust based on your text spacing
    
    for item in sorted_result[1:]:
        current_y = item[0][0][1]
        prev_y = current_line[-1][0][0][1]
        
        if abs(current_y - prev_y) <= y_threshold:
            # Same line
            current_line.append(item)
        else:
            # New line
            lines.append(sorted(current_line, key=lambda x: x[0][0][0]))  # Sort by x-coordinate
            current_line = [item]
    
    # Add the last line
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x[0][0][0]))
    
    # Concatenate text in each line and join lines with newlines
    text_lines = []
    for line in lines:
        line_text = " ".join(item[1] for item in line)
        text_lines.append(line_text)
    
    return " ".join(text_lines)

# Define the dataset directory, CSV file path, and output JSON file path
dataset_dir = r"C:\Users\abhil\OneDrive\Documents\Mtech CSE\2_Sem\NLP\MHMC\mental-health-meme-classification\dataset\Anxiety_Data\anxiety_test_image"
csv_path = r"C:\Users\abhil\OneDrive\Documents\Mtech CSE\2_Sem\NLP\MHMC\mental-health-meme-classification\dataset\Anxiety_Data\anxiety_test.csv"
output_json_path = r"C:\Users\abhil\OneDrive\Documents\Mtech CSE\2_Sem\NLP\MHMC\mental-health-meme-classification\dataset\Anxiety_Data\anxiety_test.json"

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Read the CSV file to map sample_id to meme_anxiety_categories
categories = {}
with open(csv_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        categories[row['sample_id']] = row['meme_anxiety_categories']

# List to store OCR results
ocr_results = []

# Iterate over all image files in the dataset directory with a progress bar
for image_file in tqdm(os.listdir(dataset_dir), desc="Processing Images", unit="image"):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(dataset_dir, image_file)
        
        # Perform OCR on the image
        ocr_result = reader.readtext(image_path)
        
        # Parse the OCR result to get concatenated text
        ocr_text = parse_ocr_result(ocr_result)
        
        # Extract the sample_id (filename without extension)
        sample_id = os.path.splitext(image_file)[0]
        
        # Get the corresponding category from the CSV
        meme_category = categories.get(sample_id, "Unknown")
        
        # Append the result to the list
        ocr_results.append({
            "sample_id": sample_id,
            "ocr_text": ocr_text,
            "meme_anxiety_category": meme_category
        })

# Save the combined results to the JSON file
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(ocr_results, json_file, ensure_ascii=False, indent=4)

print(f"OCR results combined with categories saved to {output_json_path}")

