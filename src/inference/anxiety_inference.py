import re
import numpy as np
import json # Changed from pickle
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
    """Creates a prompt for inference in the same format as during training (without few-shot examples)"""
    ocr_word_limit = 75
    class_names_str = str(list(class_names)) # Convert list/array to string representation

    # Process the figurative reasoning
    processed_reasoning = process_triples(figurative_reasoning)

    # Limit OCR text length
    ocr_limited = split_ocr(ocr_text, ocr_word_limit)

    # Build the prompt without few-shot examples (consistent with depression inference and typical inference setup)
    # Adjusted system message for single-label task
    prompt = f"#System: You specialize in analyzing mental health behaviors through social media posts. Your task is to classify the mental health issue depicted in a person's post from the following categories: {class_names_str}.\n\n"
    prompt += "###Your_turn:\n"
    prompt += f"<|ocr_text|>{ocr_limited}\n"
    prompt += f"<|commonsense figurative explanation|>{processed_reasoning}\n"
    prompt += "The mental health disorder of the person for this post is: " # Matches training prompt structure

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

def infer_anxiety_category(ocr_text, figurative_reasoning):
    """
    Run inference on a single example to predict the anxiety category using proper prompt formatting.

    Args:
        ocr_text: The OCR text from the meme
        figurative_reasoning: The figurative reasoning analysis

    Returns:
        dict: Dictionary containing the prediction and confidence score
    """
    # Define paths
    model_dir = "/media/nas_mount/avinash_ocr/Dhruvkumar_Patel_MT24032/others/mental-health-meme-classification/src/anxiety/training/results/iter_4_good_clean_lower_hyperparams/mental_bart_anxiety_finetuned_best_hyperparams"
    class_map_path = f"{model_dir}/label_classes.json" # Path to the class names JSON file

    # Load the model and tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model or tokenizer from {model_dir}: {e}")
        return {"error": "Failed to load model/tokenizer"}

    # Load the class names
    try:
        with open(class_map_path, 'r') as f:
            class_names_dict = json.load(f)
            
            # Convert from dictionary format to list format if needed
            if isinstance(class_names_dict, dict):
                # Sort by numeric key to ensure correct order
                sorted_items = sorted([(int(k), v) for k, v in class_names_dict.items()])
                class_names = [item[1] for item in sorted_items]
            elif isinstance(class_names_dict, list):
                class_names = class_names_dict
            else:
                raise ValueError(f"Unexpected class names format: {type(class_names_dict)}")
                
            print(f"Loaded {len(class_names)} class names: {class_names}")
    except FileNotFoundError:
        print(f"Error: Class map file not found at {class_map_path}")
        return {"error": "Class map file not found"}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {class_map_path}")
        return {"error": "Failed to decode class map JSON"}
    except ValueError as e:
        print(f"Error processing class names: {e}")
        return {"error": "Invalid class names format"}
    except Exception as e:
        print(f"An unexpected error occurred loading class names: {e}")
        return {"error": "Failed to load class names"}

    # Clean text inputs
    cleaned_ocr = clean_text(ocr_text)
    cleaned_reasoning = clean_text(figurative_reasoning)

    # Create the prompt using the same formatting as during training
    prompt = create_prompt(cleaned_ocr, cleaned_reasoning, class_names)

    print(f"Generated prompt for inference:\n{prompt}\n")

    # Tokenize the prompt
    # Use padding=True, truncation=True, and max_length consistent with training
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512) # Max length from training script

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and convert to probabilities with softmax for single-label
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0] # Use softmax

    # Get the index of the highest probability (this is the prediction)
    predicted_index = np.argmax(probabilities)

    # Get the predicted category name and its confidence score
    predicted_category = class_names[predicted_index]
    confidence_score = probabilities[predicted_index]

    # Format results
    results = {
        "predicted_category": predicted_category,
        "confidence_score": float(confidence_score), # Convert numpy float to python float
        "all_categories": class_names,
        "all_probabilities": probabilities.tolist() # Convert numpy array to list
    }

    return results

# Example usage
if __name__ == "__main__":
    # Sample input (same as depression example for consistency, or use a relevant anxiety one)
    ocr_text = "Me trying to figure out if I'm actually hungry or just anxious and need to chew something" # More anxiety-related example
    figurative_reasoning = """
    1. Cause-Effect: The meme depicts a person contemplating their physical sensation (potential hunger) versus an emotional state (anxiety manifesting physically). The cause is the internal feeling of needing to chew or eat, and the effect is the confusion about its origin – genuine hunger or anxiety relief.
    2. Figurative Understanding: This uses relatable introspection. The "figuring out" is a metaphor for the common difficulty in distinguishing between physical needs and emotional responses, especially when anxiety causes physical symptoms like oral fixation or a desire for comfort eating. It highlights the mind-body connection in anxiety.
    3. Mental State: The primary mental state is anxiety, characterized by uncertainty, physical restlessness (needing to chew), and introspection bordering on overthinking. There's an element of self-awareness but also confusion about internal signals.
    """

    # Get prediction
    prediction = infer_anxiety_category(ocr_text, figurative_reasoning)

    # Print results
    if "error" not in prediction:
        print(f"Predicted Anxiety Category: {prediction['predicted_category']}")
        print(f"Confidence Score: {prediction['confidence_score']:.4f}")

        # If you want to see all categories and their probabilities
        print("\nAll Category Probabilities:")
        for category, prob in zip(prediction["all_categories"], prediction["all_probabilities"]):
            print(f"- {category}: {prob:.4f}")
    else:
        print(f"Inference failed: {prediction['error']}")