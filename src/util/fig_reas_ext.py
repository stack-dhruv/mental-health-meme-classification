import base64
import requests
from io import BytesIO
from PIL import Image
import os
import json
import time
from tqdm import tqdm
from collections import deque
import random # Import random for jitter
import threading # Import threading
import queue     # Import queue
import math      # For ceiling division (though not strictly needed here)

# --- Configuration ---
IMAGE_DIR = "/Users/dhruv/Work/IITD/winter_semester/NLP/Project/mental-health-meme-classification/dataset/Depressive_Data/Images/depressive_image/val"
OUTPUT_JSONL_FILE = "fig_extractor_results.jsonl" # JSON Lines format for incremental saving
ERROR_LOG_FILE = "failed_images.log"
API_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
# Consider loading API key from environment variables for security
# API_KEY = os.getenv("HYPERBOLIC_API_KEY")
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJpYW1kaHJ1djE1NjNAZ21haWwuY29tIiwiaWF0IjoxNzM0MTU3NzQ4fQ.Ygj-dLfNiZn0AWjHFifMI6HAOmF6Pcqa0NLSuPV_VAw" # Replace with your actual key or load securely
REQUEST_TIMEOUT_SECONDS = 60 # Timeout for API requests
REQUEST_LIMIT_PER_MINUTE = 59 # API Rate limit
MAX_RETRIES = 5 # Maximum number of retries for 5xx server errors
INITIAL_BACKOFF = 1 # Initial wait time in seconds for retry
NUM_WORKER_THREADS = 10 # Number of concurrent threads (adjust based on performance/errors)

# --- Helper Functions ---
def encode_image(img, format="JPEG"):
    """Encodes a PIL Image object to a base64 string."""
    buffered = BytesIO()
    try:
        # Convert RGBA to RGB if necessary for JPEG saving
        if img.mode == 'RGBA' and format.upper() == 'JPEG':
            img = img.convert('RGB')
        img.save(buffered, format=format)
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_string
    except Exception as e:
        # Handle potential errors during image saving/encoding
        return None

def log_error(image_filename, error_message):
    """Logs an error message to the error file."""
    try:
        with open(ERROR_LOG_FILE, 'a') as f_err:
            f_err.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {image_filename}: {error_message}\n")
    except IOError as e:
        print(f"CRITICAL: Could not write to error log file {ERROR_LOG_FILE}: {e}")

def save_result(image_filename, result_data):
    """Saves the result for a single image to the JSON Lines output file."""
    output_record = {"image_filename": image_filename, "result": result_data}
    try:
        with open(OUTPUT_JSONL_FILE, 'a') as f_out:
            json.dump(output_record, f_out)
            f_out.write('\n') # Add newline for JSON Lines format
    except IOError as e:
        log_error(image_filename, f"Failed to write result to {OUTPUT_JSONL_FILE}: {e}")
    except Exception as e:
        log_error(image_filename, f"Unexpected error saving result: {e}")

def load_processed_images(filename):
    """Loads the set of successfully processed image filenames from a JSONL file."""
    processed = set()
    if not os.path.exists(filename):
        return processed
    try:
        with open(filename, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'image_filename' in record:
                        processed.add(record['image_filename'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {filename}: {line.strip()}")
    except IOError as e:
        print(f"Warning: Could not read existing results file {filename}: {e}")
    return processed

# --- Worker Function ---
def worker(task_queue, result_queue, request_timestamps, rate_limit_lock, headers, prompt):
    """Processes images from the task queue and puts results in the result queue."""
    # Retry and backoff parameters are local to the worker's attempts
    local_max_retries = MAX_RETRIES
    local_initial_backoff = INITIAL_BACKOFF

    while True:
        try:
            filename = task_queue.get_nowait() # Non-blocking get
        except queue.Empty:
            break # No more tasks

        image_path = os.path.join(IMAGE_DIR, filename)
        base64_img = None
        image_format = "JPEG"

        # 1. Load and Encode Image
        try:
            img = Image.open(image_path)
            image_format = img.format if img.format else "JPEG"
            if image_format.upper() not in ["JPEG", "PNG", "GIF", "BMP"] or (img.mode == 'RGBA' and image_format.upper() == 'JPEG'):
                image_format = "PNG"
            base64_img = encode_image(img, format=image_format)
            if base64_img is None:
                raise ValueError("Image encoding returned None.")
        except FileNotFoundError:
            log_error(filename, f"File not found at {image_path}")
            result_queue.put(("error", filename, "File not found"))
            task_queue.task_done()
            continue
        except Exception as e:
            log_error(filename, f"Error processing image: {e}")
            result_queue.put(("error", filename, f"Image processing error: {e}"))
            task_queue.task_done()
            continue

        # 2. Prepare API Payload
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_format.lower()};base64,{base64_img}"},
                    },
                ],
            }],
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.001,
        }

        # --- API Call with Rate Limiting and Retries ---
        retries = 0
        backoff_time = local_initial_backoff
        api_call_succeeded = False
        last_error_msg = "No API call attempted" # Default error if loop isn't entered

        while retries <= local_max_retries:
            # 3. Rate Limiting Check (Thread-safe) - Check before each attempt
            while True: # Loop until a request slot is available
                with rate_limit_lock:
                    now = time.monotonic()
                    # Remove timestamps older than 60 seconds
                    while request_timestamps and now - request_timestamps[0] > 60:
                        request_timestamps.popleft()

                    # Check if limit reached
                    if len(request_timestamps) < REQUEST_LIMIT_PER_MINUTE:
                        request_timestamps.append(now) # Reserve slot
                        should_wait = 0
                        break # Proceed with API call
                    else:
                        # Calculate wait time based on the oldest timestamp
                        time_to_wait = (request_timestamps[0] + 60) - now
                        should_wait = max(0.05, time_to_wait) # Wait at least a tiny bit

                if should_wait > 0:
                    time.sleep(should_wait)

            # 4. Call API and Handle Response
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status() # Raises HTTPError for 4xx/5xx
                api_result = response.json()
                result_queue.put(("success", filename, api_result)) # Put success result
                api_call_succeeded = True
                break # Exit retry loop on success

            except requests.exceptions.Timeout:
                last_error_msg = f"API request timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
                log_error(filename, last_error_msg + f" (Attempt {retries+1}/{local_max_retries+1})")

            except requests.exceptions.HTTPError as e:
                last_error_msg = f"API request failed: {str(e)}"
                # Retry only on 5xx server errors
                if 500 <= e.response.status_code < 600 and retries < local_max_retries:
                    error_detail = str(e)
                    log_error(filename, f"API request failed: {error_detail}. Retrying ({retries+1}/{local_max_retries})...")
                    # Exponential backoff with jitter
                    wait_time = backoff_time + random.uniform(0, 0.1 * backoff_time)
                    time.sleep(wait_time)
                    backoff_time *= 2 # Double the backoff time for next potential retry
                    retries += 1
                    continue # Go to next retry iteration
                else:
                    # Handle non-retryable HTTP errors (4xx) or max retries reached for 5xx
                    error_detail = str(e)
                    if e.response is not None:
                         error_detail += f" | Status Code: {e.response.status_code} | Response: {e.response.text[:200]}..."
                    last_error_msg = f"API request failed permanently: {error_detail}"
                    log_error(filename, last_error_msg)
                    break # Exit retry loop

            except requests.exceptions.RequestException as e:
                # Handle other request errors (DNS, Connection, etc.) - usually not retryable
                last_error_msg = f"API request failed (network/other): {e}"
                log_error(filename, last_error_msg)
                break # Exit retry loop

            except json.JSONDecodeError as e:
                 # If API returns non-JSON on success status (unlikely but possible)
                 last_error_msg = f"Failed to decode API JSON response: {e}"
                 log_error(filename, last_error_msg)
                 break # Exit retry loop
            except Exception as e:
                # Catch-all for unexpected errors during API call/handling
                last_error_msg = f"An unexpected error occurred during API call: {e}"
                log_error(filename, last_error_msg)
                break # Exit retry loop

            # Increment retry count if a retryable error occurred and we didn't 'continue'
            retries += 1
            # Add a small delay before the next retry attempt even if not backing off exponentially
            time.sleep(0.1)

        # If the loop finished without success
        if not api_call_succeeded:
             result_queue.put(("error", filename, last_error_msg)) # Put the final error result

        task_queue.task_done() # Signal task completion for this filename

# --- Main Processing Logic ---

if not API_KEY:
    print("Error: API key not found.")
    exit()

if not os.path.isdir(IMAGE_DIR):
    print(f"Error: Image directory not found at {IMAGE_DIR}")
    exit()

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

prompt = """
Analyze the following depression meme image to extract common sense reasoning in the form of triples. These relationships should capture the following elements:
1. Cause-effect: Identify concrete causes or results of the situation depicted in the meme.
2. Figurative Understanding: Capture underlying metaphors, analogies, or symbolic meanings that convey the meme’s deeper message, including any ironic or humorous undertones.
3. Mental State: Capture specific mental or emotional states depicted in the meme.
"""

# Get list of all image files
try:
    all_files = os.listdir(IMAGE_DIR)
    all_image_files = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
except OSError as e:
    print(f"Error reading image directory {IMAGE_DIR}: {e}")
    exit()

total_images_in_dir = len(all_image_files)
print(f"Found {total_images_in_dir} total images in {IMAGE_DIR}.")

# Load already processed images
processed_images_set = load_processed_images(OUTPUT_JSONL_FILE)
print(f"Found {len(processed_images_set)} already processed images in {OUTPUT_JSONL_FILE}.")

# Determine images to process
images_to_process = [f for f in all_image_files if f not in processed_images_set]
num_to_process = len(images_to_process)

if num_to_process == 0:
    print("All images have already been processed. Exiting.")
    exit()

print(f"Starting processing for {num_to_process} remaining images using up to {NUM_WORKER_THREADS} threads.")
print(f"Results will be appended to: {OUTPUT_JSONL_FILE}")
print(f"Errors will be logged to: {ERROR_LOG_FILE}")

# --- Setup Queues, Lock, and Threads ---
task_queue = queue.Queue()
result_queue = queue.Queue()
request_timestamps = deque() # Shared deque for rate limiting
rate_limit_lock = threading.Lock() # Lock for thread-safe access to deque

# Fill the task queue
for filename in images_to_process:
    task_queue.put(filename)

# Start worker threads
threads = []
for _ in range(NUM_WORKER_THREADS):
    # Pass shared resources (queues, deque, lock) and static data (headers, prompt)
    t = threading.Thread(target=worker, args=(task_queue, result_queue, request_timestamps, rate_limit_lock, headers, prompt), daemon=True)
    t.start()
    threads.append(t)

# --- Process Results and Track Progress (Main Thread) ---
current_run_processed_count = 0
current_run_error_count = 0
processed_in_loop = 0

# Use tqdm for overall progress based on results processed
with tqdm(total=num_to_process, desc="Processing Images") as pbar:
    while processed_in_loop < num_to_process:
        try:
            # Get results from the queue, block for a short time to avoid busy-waiting
            status, filename, data = result_queue.get(timeout=0.5)

            if status == "success":
                save_result(filename, data) # Save the successful result
                current_run_processed_count += 1
            else: # status == "error"
                # Error was already logged by the worker thread
                current_run_error_count += 1

            processed_in_loop += 1
            pbar.update(1) # Update progress bar for each processed item

        except queue.Empty:
            # Check if all worker threads have finished if the queue is empty
            if not any(t.is_alive() for t in threads):
                if task_queue.empty(): # Double check if tasks really finished
                     break # Exit loop if workers are done and queue is empty
                else:
                     # Should not happen if task_done() is called correctly, but handle defensively
                     print("Warning: Workers finished but task queue not empty. Waiting...")
                     time.sleep(1)


# --- Final Summary ---
print("\n--- Processing Summary (This Run) ---")
print(f"Images attempted in this run: {num_to_process}")
print(f"Successfully processed in this run: {current_run_processed_count}")
print(f"Failed in this run: {current_run_error_count}")
final_processed_count = len(load_processed_images(OUTPUT_JSONL_FILE))
print("\n--- Overall Status ---")
print(f"Total images in directory: {total_images_in_dir}")
print(f"Total successfully processed (cumulative): {final_processed_count}")
print(f"Results saved to: {OUTPUT_JSONL_FILE}")
print(f"Error details logged in: {ERROR_LOG_FILE}")
print("Processing complete.")
