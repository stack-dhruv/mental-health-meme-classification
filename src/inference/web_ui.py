# app.py
import streamlit as st
import easyocr
import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import time
import json
import os
import re
import pickle # For loading MultiLabelBinarizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
API_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60
API_KEY = st.secrets.get("HYPERBOLIC_API_KEY", None)

# --- Fallback API Key (Use Secrets!) ---
if not API_KEY:
    API_KEY = "your_actual_api_key_here_fallback" # Replace if needed, but use secrets!

# --- Model Paths (CRITICAL: Ensure these paths are correct and accessible!) ---
# Option 1: Use absolute paths
# DEPRESSION_MODEL_DIR = "/path/to/your/depression/model/mental_bart_depression_finetuned_multilabel"
# ANXIETY_MODEL_DIR = "/path/to/your/anxiety/model/mental_bart_anxiety_finetuned_best_hyperparams"

# Option 2: Use paths relative to this app.py file (if models are nearby)
# Assuming models are in subdirectories 'models/depression' and 'models/anxiety'
base_path = os.path.dirname(__file__) # Get directory where app.py is located
# DEPRESSION_MODEL_DIR = os.path.join(base_path, "models", "depression", "mental_bart_depression_finetuned_multilabel")
# ANXIETY_MODEL_DIR = os.path.join(base_path, "models", "anxiety", "mental_bart_anxiety_finetuned_best_hyperparams")

# Option 3: Using the paths from your inference scripts (ensure they are valid)
DEPRESSION_MODEL_DIR = "/media/nas_mount/avinash_ocr/Dhruvkumar_Patel_MT24032/others/mental-health-meme-classification/src/depression/training/results/mental_bart_depression_finetuned_multilabel"
ANXIETY_MODEL_DIR = "/media/nas_mount/avinash_ocr/Dhruvkumar_Patel_MT24032/others/mental-health-meme-classification/src/anxiety/training/results/iter_4_good_clean_lower_hyperparams/mental_bart_anxiety_finetuned_best_hyperparams"


# --- Helper Functions (OCR, Reasoning, Text Processing - from previous steps and inference scripts) ---

@st.cache_resource
def get_ocr_reader():
    """Initializes and returns the EasyOCR reader."""
    print("Initializing EasyOCR Reader...")
    try:
        # Changed gpu=False based on your latest web_ui code example
        reader = easyocr.Reader(['en'], gpu=False)
        return reader
    except Exception as e:
        st.error(f"Failed to initialize EasyOCR Reader: {e}")
        return None

def parse_ocr_result(ocr_result):
    """Parse OCR result and return concatenated text."""
    if not ocr_result: return ""
    try:
        if not all(isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], (list, tuple)) for item in ocr_result):
            return " ".join(item[1] for item in ocr_result if len(item) >=2)
        sorted_result = sorted(ocr_result, key=lambda x: x[0][0][1])
        lines = []
        if not sorted_result: return ""
        current_line = [sorted_result[0]]
        y_threshold = 20
        for item in sorted_result[1:]:
            if not (isinstance(item[0], (list, tuple)) and len(item[0]) > 0 and isinstance(item[0][0], (list, tuple)) and len(item[0][0]) >= 2): continue
            current_y = item[0][0][1]
            prev_y = current_line[-1][0][0][1]
            if abs(current_y - prev_y) <= y_threshold:
                current_line.append(item)
            else:
                lines.append(sorted(current_line, key=lambda x: x[0][0][0]))
                current_line = [item]
        if current_line: lines.append(sorted(current_line, key=lambda x: x[0][0][0]))
        text_lines = [" ".join(item[1] for item in line) for line in lines]
        return " ".join(text_lines)
    except Exception as e:
        st.error(f"Error parsing OCR results: {e}")
        return " ".join(item[1] for item in ocr_result if len(item) >=2)

@st.cache_data(show_spinner=False)
def extract_ocr(image_bytes):
    """Performs OCR on the image bytes and returns the text."""
    reader = get_ocr_reader()
    if reader is None: return "Error: OCR Reader not available."
    try:
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)
        ocr_result = reader.readtext(image_np)
        ocr_text = parse_ocr_result(ocr_result)
        return ocr_text
    except Exception as e:
        return f"Error during OCR processing: {e}"

def encode_image(img, format="JPEG"):
    """Encodes a PIL Image object to a base64 string."""
    buffered = BytesIO()
    try:
        if img.mode == 'RGBA' and format.upper() == 'JPEG': img = img.convert('RGB')
        img.save(buffered, format=format)
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_string
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

@st.cache_data(show_spinner=False)
def extract_figurative_reasoning(image_bytes, analysis_type):
    """Calls the Hyperbolic API to get figurative reasoning."""
    if not API_KEY or API_KEY == "your_actual_api_key_here_fallback":
        return "Error: API Key not configured. Please set it in Streamlit secrets."
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    try:
        img = Image.open(BytesIO(image_bytes))
        image_format = img.format if img.format else "JPEG"
        if image_format.upper() not in ["JPEG", "PNG", "GIF", "WEBP"]: image_format = "PNG"
        base64_img = encode_image(img, format=image_format)
        if base64_img is None: return "Error: Could not encode image for API."
    except Exception as e: return f"Error processing image before API call: {e}"
    prompt_template = """Analyze the following {analysis_topic} meme image to extract common sense reasoning in the form of triples. These relationships should capture the following elements:
1. Cause-effect: Identify concrete causes or results of the situation depicted in the meme.
2. Figurative Understanding: Capture underlying metaphors, analogies, or symbolic meanings that convey the meme’s deeper message, including any ironic or humorous undertones.
3. Mental State: Capture specific mental or emotional states depicted in the meme.""" # Use the full prompt from your example
    prompt = prompt_template.format(analysis_topic=analysis_type.upper())
    payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/{image_format.lower()};base64,{base64_img}"}}]}], "model": "Qwen/Qwen2.5-VL-7B-Instruct", "max_tokens": 512, "temperature": 0.1, "top_p": 0.001}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        api_result = response.json()
        if 'choices' in api_result and len(api_result['choices']) > 0:
            message = api_result['choices'][0].get('message', {})
            content = message.get('content', "Error: 'content' not found in API response message.")
            return content
        else: return f"Error: Unexpected API response structure: {api_result}"
    except requests.exceptions.Timeout: return f"Error: API request timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
    except requests.exceptions.HTTPError as e:
        error_message = f"Error: API request failed with status {e.response.status_code}."
        try: error_details = e.response.json(); error_message += f"\nDetails: {json.dumps(error_details, indent=2)}"
        except json.JSONDecodeError: error_message += f"\nResponse Body: {e.response.text[:500]}..."
        return error_message
    except requests.exceptions.RequestException as e: return f"Error: API request failed (network issue): {e}"
    except Exception as e: return f"Error: An unexpected error occurred during API call: {e}"


# --- Text Processing Helpers (from inference scripts) ---

def split_ocr(text, limit):
    """Splits text by words up to a specified limit."""
    if not isinstance(text, str): text = str(text)
    words = text.split()
    if len(words) > limit: return ' '.join(words[:limit]) + '...'
    return text

def process_triples(triples):
    """Robustly processes the figurative reasoning string."""
    # This function is identical in both inference scripts, using one copy.
    if not isinstance(triples, str) or not triples.strip():
        return "Missing or empty reasoning"
    text_lower = triples.lower()
    sections_to_find = {'cause-effect': None, 'figurative understanding': None, 'mental state': None}
    pattern_section_start = r'(?:^\s*\d+\.\s*)?(?:[\#\*]*)\s*(cause-effect|figurative understanding|mental state)\b'
    matches = list(re.finditer(pattern_section_start, text_lower, re.MULTILINE))
    if not matches: return text_lower
    found_sections_pos = {}
    for match in matches:
        title = match.group(1).strip()
        marker_end_match = re.search(r'[:\*]+', text_lower[match.end():match.end()+10])
        marker_end_pos = match.end() + marker_end_match.end() if marker_end_match else match.end()
        found_sections_pos[title] = {'start': match.start(), 'marker_end': marker_end_pos}
    sorted_titles = sorted(found_sections_pos.keys(), key=lambda k: found_sections_pos[k]['start'])
    raw_content = {}
    for i, title in enumerate(sorted_titles):
        start_content = found_sections_pos[title]['marker_end']
        next_section_start = len(text_lower)
        if (i + 1) < len(sorted_titles):
            next_title = sorted_titles[i+1]
            next_section_start = found_sections_pos[next_title]['start']
        content = text_lower[start_content:next_section_start].strip()
        raw_content[title] = content
    for title in sections_to_find.keys():
        if title in raw_content:
            content_to_clean = raw_content[title]
            lines = content_to_clean.splitlines()
            cleaned_lines = []
            for line in lines:
                cleaned_line = re.sub(r'^\s*[-\*\u2022\d\.\:]+\s*', '', line).strip()
                cleaned_line = re.sub(r'^\s*\w+\s*:', '', cleaned_line).strip()
                if cleaned_line: cleaned_lines.append(cleaned_line)
            cleaned_content = ' '.join(cleaned_lines)
            if cleaned_content: sections_to_find[title] = cleaned_content
    all_sections_found = all(content is not None for content in sections_to_find.values())
    if all_sections_found:
        output_lines = [f"{title}: {content}" for title, content in sections_to_find.items()]
        return '\n'.join(output_lines)
    else:
        return text_lower

def clean_text(text):
    """Remove non-ASCII characters and normalize spaces"""
    if not isinstance(text, str): text = str(text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Using the more general create_prompt from depressive_inference.py
def create_prompt(ocr_text, figurative_reasoning, class_names):
    """Creates a prompt for inference."""
    ocr_word_limit = 75
    # Ensure class_names is list-like before converting
    class_names_str = str(list(class_names))

    processed_reasoning = process_triples(figurative_reasoning)
    ocr_limited = split_ocr(ocr_text, ocr_word_limit)

    prompt = f"#System: You specialize in analyzing mental health behaviors through social media posts. Your task is to identify all mental health issues depicted in a person's post from the following categories: {class_names_str}. A post may exhibit multiple mental health issues.\n\n"
    prompt += "###Your_turn:\n"
    prompt += f"<|ocr_text|>{ocr_limited}\n"
    prompt += f"<|commonsense figurative explanation|>{processed_reasoning}\n"
    prompt += "The mental health disorders of the person for this post are: "
    return prompt


# --- Model and Artifact Loading (Cached) ---

@st.cache_resource
def load_depression_resources(model_dir):
    """Loads the depression model, tokenizer, MLB, and threshold."""
    print(f"Loading depression resources from: {model_dir}")
    mlb_path = os.path.join(model_dir, "multilabel_binarizer.pkl")
    threshold_path = os.path.join(model_dir, "threshold.txt")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.eval() # Set to evaluation mode

        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)

        try:
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
        except FileNotFoundError:
            st.warning(f"Threshold file not found at {threshold_path}. Using default 0.5.")
            threshold = 0.5
        except ValueError:
            st.warning(f"Invalid value in threshold file {threshold_path}. Using default 0.5.")
            threshold = 0.5

        print("Depression resources loaded successfully.")
        return model, tokenizer, mlb, threshold
    except FileNotFoundError as e:
        st.error(f"Error loading depression resources: File not found - {e}. Please check paths in app.py.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading depression resources from {model_dir}: {e}")
        return None, None, None, None

@st.cache_resource
def load_anxiety_resources(model_dir):
    """Loads the anxiety model, tokenizer, and class names."""
    print(f"Loading anxiety resources from: {model_dir}")
    class_map_path = os.path.join(model_dir, "label_classes.json")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.eval() # Set to evaluation mode

        with open(class_map_path, 'r') as f:
            class_names_dict = json.load(f)
            if isinstance(class_names_dict, dict):
                sorted_items = sorted([(int(k), v) for k, v in class_names_dict.items()])
                class_names = [item[1] for item in sorted_items]
            elif isinstance(class_names_dict, list):
                class_names = class_names_dict
            else: raise ValueError("Unexpected class names format")

        print("Anxiety resources loaded successfully.")
        return model, tokenizer, class_names
    except FileNotFoundError as e:
        st.error(f"Error loading anxiety resources: File not found - {e}. Please check paths in app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading anxiety resources from {model_dir}: {e}")
        return None, None, None


# --- Inference Functions (Adapted from inference scripts) ---

# Note: We pass the loaded resources to these functions now

def infer_depression_categories(model, tokenizer, mlb, threshold, ocr_text, figurative_reasoning):
    """Run inference for depression categories."""
    if not all([model, tokenizer, mlb, threshold]):
        return {"error": "Depression model/resources not loaded properly."}

    class_names = mlb.classes_
    cleaned_ocr = clean_text(ocr_text)
    cleaned_reasoning = clean_text(figurative_reasoning)
    prompt = create_prompt(cleaned_ocr, cleaned_reasoning, class_names)
    # print(f"Generated depression prompt for inference:\n{prompt}\n") # Optional debug

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0] # Use CPU after detaching
        predicted_indices = np.where(probabilities >= threshold)[0]

        if len(predicted_indices) == 0 and len(probabilities) > 0:
            predicted_indices = [np.argmax(probabilities)]

        sorted_idx_prob_pairs = sorted([(idx, probabilities[idx]) for idx in predicted_indices], key=lambda x: x[1], reverse=True)
        predicted_categories = [class_names[idx] for idx, _ in sorted_idx_prob_pairs]
        sorted_scores = [float(score) for _, score in sorted_idx_prob_pairs] # Convert to float

        results = {
            "predicted_categories": predicted_categories,
            "confidence_scores": sorted_scores,
            "all_categories": class_names.tolist(),
            "all_probabilities": probabilities.tolist()
        }
        return results
    except Exception as e:
        st.error(f"Error during depression inference: {e}")
        return {"error": f"Inference failed: {e}"}


def infer_anxiety_category(model, tokenizer, class_names, ocr_text, figurative_reasoning):
    """Run inference for anxiety category."""
    if not all([model, tokenizer, class_names]):
         return {"error": "Anxiety model/resources not loaded properly."}

    cleaned_ocr = clean_text(ocr_text)
    cleaned_reasoning = clean_text(figurative_reasoning)
    prompt = create_prompt(cleaned_ocr, cleaned_reasoning, class_names)
    # print(f"Generated anxiety prompt for inference:\n{prompt}\n") # Optional debug

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0] # Use softmax, move to CPU
        predicted_index = np.argmax(probabilities)
        predicted_category = class_names[predicted_index]
        confidence_score = probabilities[predicted_index]

        results = {
            "predicted_category": predicted_category,
            "confidence_score": float(confidence_score),
            "all_categories": class_names, # Already a list
            "all_probabilities": probabilities.tolist()
        }
        return results
    except Exception as e:
        st.error(f"Error during anxiety inference: {e}")
        return {"error": f"Inference failed: {e}"}


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🧠 Meme Mental Health Symptom Analysis")
st.markdown("Upload a meme image and select the analysis type to extract OCR text, figurative reasoning, and classify symptoms.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("1. Upload Meme Image", type=["png", "jpg", "jpeg"])

    analysis_option = st.radio(
        "2. Select Analysis Type:",
        ('Depressive Symptom Analysis', 'Anxiety Symptom Analysis'),
        key='analysis_type',
        # Clear cache for reasoning/inference if option changes
        # This isn't perfect cache invalidation but helps
        on_change=lambda: st.session_state.pop('ocr_text', None) or st.session_state.pop('reasoning_text', None) or st.session_state.pop('inference_results', None)
    )
    analysis_type_short = "Depression" if "Depressive" in analysis_option else "Anxiety"

# --- Initialize Session State Variables ---
if 'ocr_text' not in st.session_state: st.session_state['ocr_text'] = None
if 'reasoning_text' not in st.session_state: st.session_state['reasoning_text'] = None
if 'inference_results' not in st.session_state: st.session_state['inference_results'] = None


# --- Main Area ---
if uploaded_file is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_file, width=400) # Display the uploaded image

    image_bytes = uploaded_file.getvalue() # Read image data once

    # --- Column Layout ---
    col1, col2 = st.columns(2)

    # --- Step 1: OCR ---
    with col1:
        st.markdown("#### 📄 OCR Text")
        # Only run OCR if not already done for this upload
        if st.session_state.ocr_text is None:
            with st.spinner("Extracting text using EasyOCR..."):
                st.session_state.ocr_text = extract_ocr(image_bytes)
        # Display OCR text (or error)
        if isinstance(st.session_state.ocr_text, str) and st.session_state.ocr_text.startswith("Error"):
             st.error(st.session_state.ocr_text)
        else:
             st.text_area("Extracted Text:", value=st.session_state.ocr_text or "", height=200, key="ocr_display")

    # --- Step 2: Figurative Reasoning ---
    with col2:
        st.markdown(f"#### 🤔 Figurative Reasoning ({analysis_type_short})")
        # Only run Reasoning if OCR was successful and reasoning not yet done
        if st.session_state.ocr_text and not isinstance(st.session_state.ocr_text, str) or not st.session_state.ocr_text.startswith("Error"):
             if st.session_state.reasoning_text is None:
                  with st.spinner(f"Requesting {analysis_type_short} reasoning from API..."):
                       st.session_state.reasoning_text = extract_figurative_reasoning(image_bytes, analysis_type_short)
        # Display Reasoning text (or error)
        if isinstance(st.session_state.reasoning_text, str) and st.session_state.reasoning_text.startswith("Error"):
             st.error(st.session_state.reasoning_text)
        elif st.session_state.reasoning_text:
             st.text_area("API Response:", value=st.session_state.reasoning_text, height=200, key="reasoning_display")
        elif isinstance(st.session_state.ocr_text, str) and st.session_state.ocr_text.startswith("Error"):
             st.info("Reasoning step skipped due to OCR error.")
        else:
             st.info("Waiting for OCR text...")


    # --- Step 3: Inference (only if OCR and Reasoning are successful) ---
    st.markdown("---") # Separator
    st.subheader("🔍 Symptom Classification")

    ocr_successful = st.session_state.ocr_text and not (isinstance(st.session_state.ocr_text, str) and st.session_state.ocr_text.startswith("Error"))
    reasoning_successful = st.session_state.reasoning_text and not (isinstance(st.session_state.reasoning_text, str) and st.session_state.reasoning_text.startswith("Error"))

    if ocr_successful and reasoning_successful:
        if st.session_state.inference_results is None: # Run inference only once
            with st.spinner(f"Running {analysis_type_short} symptom classification model..."):
                if analysis_type_short == "Depression":
                    # Load resources
                    dep_model, dep_tokenizer, dep_mlb, dep_threshold = load_depression_resources(DEPRESSION_MODEL_DIR)
                    if dep_model: # Check if loading was successful
                        st.session_state.inference_results = infer_depression_categories(
                            dep_model, dep_tokenizer, dep_mlb, dep_threshold,
                            st.session_state.ocr_text, st.session_state.reasoning_text
                        )
                    else:
                        st.session_state.inference_results = {"error": "Could not load depression model resources."}
                else: # Anxiety
                    # Load resources
                    anx_model, anx_tokenizer, anx_class_names = load_anxiety_resources(ANXIETY_MODEL_DIR)
                    if anx_model: # Check if loading was successful
                        st.session_state.inference_results = infer_anxiety_category(
                            anx_model, anx_tokenizer, anx_class_names,
                            st.session_state.ocr_text, st.session_state.reasoning_text
                        )
                    else:
                        st.session_state.inference_results = {"error": "Could not load anxiety model resources."}

        # --- Display Inference Results ---
        results = st.session_state.inference_results
        if results:
            if "error" in results:
                st.error(f"Classification Failed: {results['error']}")
            elif analysis_type_short == "Depression":
                st.markdown("**Predicted Depressive Categories:**")
                if results.get("predicted_categories"):
                    for category, score in zip(results["predicted_categories"], results["confidence_scores"]):
                        st.markdown(f"- **{category}** (Confidence: {score:.4f})")
                else:
                    st.markdown("*No categories met the threshold.*")

                with st.expander("Show All Category Probabilities (Depression)"):
                    if results.get("all_categories"):
                        prob_data = sorted(zip(results["all_categories"], results["all_probabilities"]), key=lambda x: x[1], reverse=True)
                        for category, prob in prob_data:
                            st.markdown(f"- {category}: {prob:.4f}")
                    else:
                        st.write("Probability data not available.")

            else: # Anxiety
                st.markdown(f"**Predicted Anxiety Category:**")
                st.markdown(f"- **{results.get('predicted_category', 'N/A')}** (Confidence: {results.get('confidence_score', 0.0):.4f})")

                with st.expander("Show All Category Probabilities (Anxiety)"):
                     if results.get("all_categories"):
                        prob_data = sorted(zip(results["all_categories"], results["all_probabilities"]), key=lambda x: x[1], reverse=True)
                        for category, prob in prob_data:
                             st.markdown(f"- {category}: {prob:.4f}")
                     else:
                        st.write("Probability data not available.")
        else:
            st.info("Classification results are pending or failed.") # Should not happen if logic is correct

    elif not ocr_successful:
        st.warning("Symptom classification requires successful OCR extraction.")
    elif not reasoning_successful:
        st.warning("Symptom classification requires successful Figurative Reasoning extraction.")

else:
    st.info("Please upload an image using the sidebar to begin analysis.")

# --- Optional: Add footer ---
st.markdown("---")
st.markdown("Built with Streamlit, EasyOCR, Transformers, and Hyperbolic API.")