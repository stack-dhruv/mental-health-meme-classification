import re
import numpy as np
import pickle
import torch
import ast
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    """
    if not isinstance(triples, str) or not triples.strip():
        return "Missing or empty reasoning" # Handle None, empty strings

    text_lower = triples.lower() # Lowercase the entire input first

    sections_to_find = {
        'cause-effect': None,
        'figurative understanding': None,
        'mental state': None
    }

    # Pattern to find section markers
    pattern_section_start = r'(?:^\s*\d+\.\s*)?(?:[\#\*]*)\s*(cause-effect|figurative understanding|mental state)\b'

    matches = list(re.finditer(pattern_section_start, text_lower, re.MULTILINE))

    # If no potential section markers are found at all, fallback immediately
    if not matches:
        return text_lower # Fallback to original lowercased text

    # Store found sections and their start/end points
    found_sections_pos = {}
    for match in matches:
        title = match.group(1).strip()
        # Find the end of the marker
        marker_end_match = re.search(r'[:\*]+', text_lower[match.end():match.end()+10])
        marker_end_pos = match.end() + marker_end_match.end() if marker_end_match else match.end()
        found_sections_pos[title] = {'start': match.start(), 'marker_end': marker_end_pos}

    # Determine content boundaries and extract raw content
    sorted_titles = sorted(found_sections_pos.keys(), key=lambda k: found_sections_pos[k]['start'])

    raw_content = {}
    for i, title in enumerate(sorted_titles):
        start_content = found_sections_pos[title]['marker_end']
        # End content at the start of the next found section marker, or end of text
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

    # Check if all three required sections were found and have content
    all_sections_found = all(content is not None for content in sections_to_find.values())

    if all_sections_found:
        # Format the output string
        output_lines = [f"{title}: {content}" for title, content in sections_to_find.items()]
        return '\n'.join(output_lines)
    else:
        # If any section is missing or empty after cleaning, return the original lowercased text
        return text_lower

def create_prompt(ocr_text, figurative_reasoning, class_names):
    """Creates a prompt for inference in the same format as during training"""
    ocr_word_limit = 75
    class_names_str = str(list(class_names))
    
    # Process the figurative reasoning
    processed_reasoning = process_triples(figurative_reasoning)
    
    # Limit OCR text length
    ocr_limited = split_ocr(ocr_text, ocr_word_limit)
    
    # Build the prompt without few-shot examples (since we don't have access to training data during inference)
    prompt = f"#System: You specialize in analyzing mental health behaviors through social media posts. Your task is to identify all mental health issues depicted in a person's post from the following categories: {class_names_str}. A post may exhibit multiple mental health issues.\n\n"
    prompt += "###Your_turn:\n"
    prompt += f"<|ocr_text|>{ocr_limited}\n"
    prompt += f"<|commonsense figurative explanation|>{processed_reasoning}\n"
    prompt += "The mental health disorders of the person for this post are: "
    
    return prompt

def clean_text(text):
    """Remove non-ASCII characters and normalize spaces"""
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    text = text.strip()
    
    return text

def infer_depression_categories(ocr_text, figurative_reasoning):
    """
    Run inference on a single example to predict depression categories using proper prompt formatting
    
    Args:
        ocr_text: The OCR text from the meme
        figurative_reasoning: The figurative reasoning analysis
        
    Returns:
        dict: Dictionary containing predictions and confidence scores
    """
    # Define paths
    model_dir = "/media/nas_mount/avinash_ocr/Dhruvkumar_Patel_MT24032/others/mental-health-meme-classification/src/depression/training/results/mental_bart_depression_finetuned_multilabel"
    mlb_path = f"{model_dir}/multilabel_binarizer.pkl"
    threshold_path = f"{model_dir}/threshold.txt"
    
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()  # Set model to evaluation mode
    
    # Load the multilabel binarizer
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    
    # Load the threshold for prediction
    try:
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
    except:
        # Default threshold if file can't be read
        threshold = 0.5
    
    # Get class names from the multilabel binarizer
    class_names = mlb.classes_
    
    # Clean text inputs (in case we have non-ASCII chars)
    cleaned_ocr = clean_text(ocr_text)
    cleaned_reasoning = clean_text(figurative_reasoning)
    
    # Create the prompt using the same formatting as during training
    prompt = create_prompt(cleaned_ocr, cleaned_reasoning, class_names)
    
    print(f"Generated prompt for inference:\n{prompt}\n")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and convert to probabilities with sigmoid for multilabel
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).detach().numpy()[0]
    
    # Apply threshold for positive predictions
    predicted_indices = np.where(probabilities >= threshold)[0]
    
    # If no category exceeds threshold, take the highest one
    if len(predicted_indices) == 0:
        predicted_indices = [np.argmax(probabilities)]
    
    # Sort the indices by probability values (descending)
    sorted_idx_prob_pairs = sorted(
        [(idx, probabilities[idx]) for idx in predicted_indices],
        key=lambda x: x[1],
        reverse=True
    )
    sorted_indices = [idx for idx, _ in sorted_idx_prob_pairs]
    sorted_scores = [score for _, score in sorted_idx_prob_pairs]
    
    # Get predicted categories
    predicted_categories = [class_names[idx] for idx in sorted_indices]
    
    # Format results
    results = {
        "predicted_categories": predicted_categories,
        "confidence_scores": sorted_scores,
        "all_categories": class_names.tolist(),
        "all_probabilities": probabilities.tolist()
    }
    
    return results

# Example usage
if __name__ == "__main__":
    # Sample input
    ocr_text = "Me Doctor: Are you having homicidal thoughts? No. Me Doctor: Are you having any suicidal thoughts? @silvercrowv1"
    figurative_reasoning = "1. **Cause-effect**: The meme illustrates a common scenario where a person is asked about their mental health, specifically suicidal thoughts. The cause is the doctor's inquiry, and the effect is the person's response, which humorously reveals they are not suicidal but are instead thinking about homicide. 2. **Figurative Understanding**: The meme uses a humorous and ironic undertone to highlight the absurdity of the situation. The underlying metaphor is that the person is so distressed that they are considering violent thoughts, which is a stark contrast to the typical concern about suicidal thoughts. This humorously suggests that the person's mental state is so severe that even violent thoughts are considered a less severe concern. 3. **Mental State**: The mental state depicted in the meme includes feelings of distress, anxiety, and possibly depression. The person is shown in a state of discomfort and unease, as indicated by their facial expressions and body language. The humor in the meme comes from the person's response, which is a direct and somewhat exaggerated reaction to the doctor's question."
    
    # Get prediction
    prediction = infer_depression_categories(ocr_text, figurative_reasoning)
    
    # Print results
    print("Predicted Depression Categories:")
    for category, score in zip(prediction["predicted_categories"], prediction["confidence_scores"]):
        print(f"- {category}: {score:.4f}")
    
    # If you want to see all categories and their probabilities
    print("\nAll Category Probabilities:")
    for category, prob in zip(prediction["all_categories"], prediction["all_probabilities"]):
        print(f"- {category}: {prob:.4f}")