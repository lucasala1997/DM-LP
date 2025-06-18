import os
import time
import logging
import base64
from io import BytesIO
from PIL import Image

import torch
from tqdm import tqdm

# vLLM imports
from vllm import LLM, SamplingParams
import PIL
# -------------------------- CONFIGURATION -------------------------- #
PIL.Image.MAX_IMAGE_PIXELS = None
# Folders & paths
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
data_folder = "/leonardo_scratch/large/userexternal/gsullutr/chunk_3"
result_folder = "/leonardo_scratch/large/userexternal/gsullutr/chunk_3_isbn"
checkpoint_file = os.path.join(result_folder, "checkpoint.txt")
isbn_found_file = os.path.join(result_folder, "isbn_found.txt")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# Logging
log_file = os.path.join(result_folder, "isbn_detection.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting the processing script with vLLM.")

# Set random seed for reproducibility
seed_value = 7
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Load checkpoint and ISBN paths
processed_documents = set()
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        processed_documents = set(line.strip() for line in f)

isbn_found = []
if os.path.exists(isbn_found_file):
    with open(isbn_found_file, "r") as f:
        isbn_found = [line.strip() for line in f]

# -------------------------- HELPER FUNCTIONS -------------------------- #

def resize_image_to_base64(image_path, max_size=512):
    """
    Resize an image so that its maximum dimension is max_size,
    then encode in base64 (PNG).
    """
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img = img.convert("RGB")
        return img

# The instruction prompt
isbn_detection = (
    "Analyze the provided page to determine if it contains an ISBN code. An ISBN (International Standard Book Number) is a unique international identification system for monographic publications, assigned to each product form or edition produced by a specific publisher. This code is used throughout the supply and distribution chain as a key element for ordering, inventory, rights management, and sales data monitoring.  It typically consists of 13 digits, often displayed in a format such as '978-3-16-148410-0' or similar variations (e.g., without '-' between digits' blocks). Use this information to examine the content of the PDF, including visible text, metadata or images containing textual information, and indicate whether an ISBN is present. In Arabic books, ISBNs are typically found on the copyright page or back cover. Look for:\n"
    "- A 10 or 13-digit number sequence\n"
    "- The prefix 'ISBN' or 'ISSN' or 'ردمك' or 'الترقيم الدولي' or 'تدمك' (Arabic equivalents) or 'شابک' (Persian equivalent)\n"
    "- Numbers in either Arabic numerals '0; 1; 2; 3; 4; 5; 6; 7; 8; 9' or Arabic-Indic numerals '٠ ;١ ; ٢ ; ٣; ٤; ٥; ٦; ٧; ٨; ٩' or Eastern Arabic-Indic numerals '۰; ۱; ۲; ۳; ۴; ۵;    ۶; ۷; ۸; ۹'\n"
    "- Common ISBN locations like copyright page, back cover, or publishing details page\n\n"
    "- Numbers in either Arabic or Eastern Arabic numerals\n"
    "- Common ISBN locations like copyright page, back cover, or publishing details page\n"
    "- Presence of hyphens or spaces between number groups\n\n"
    "Respond with 'ISBN Present' or 'ISBN Absent' followed by your confidence level (High/Medium/Low) and the specific location if found."
)

# -------------------------- INITIALIZE vLLM MODEL -------------------------- #
llm = LLM(
    model="models/Qwen/Qwen2-VL-72B-Instruct",  # or local checkpoint path
    tensor_parallel_size=4,
    dtype=torch.bfloat16,
    enable_prefix_caching=True,
    max_model_len=8192,
    enforce_eager=True,
    # disable_mm_preprocessor_cache=True,
    # You can add other LLM() parameters as needed
)

# Generation parameters
sampling_params = SamplingParams(
    max_tokens=128,
    temperature=0.0,  # deterministic
    top_p=1.0,
    top_k=50,
    repetition_penalty=1.0,
    # etc.
)

# -------------------------- MAIN LOGIC -------------------------- #

start_time = time.time()

# Define batch (chunk) size
CHUNK_SIZE = 1000

# This set keeps track of document folders for which an ISBN has been found.
docs_with_isbn_found = set()

# Lists for holding batch data
batch_prompts = []
batch_metadata = []  # each element is a tuple (document_folder, file_path)

# Iterate over each document folder in the data_folder.
for root, _, files in tqdm(os.walk(data_folder)):
    document_folder = os.path.abspath(root)
    if document_folder in processed_documents:
        continue

    # Filter valid image files (ignoring those starting with 'empty_page_')
    image_files = sorted(
        f
        for f in files
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
        and not f.startswith("empty_page_")
    )

    # If no valid images, mark as processed and continue
    if not image_files:
        processed_documents.add(document_folder)
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(document_folder + "\n")
        continue

    # Process each image in the document folder
    for file_name in image_files:
        # If ISBN has already been found in this document, skip further images.
        if document_folder in docs_with_isbn_found:
            continue

        file_path = os.path.join(document_folder, file_name)
        base64_image = resize_image_to_base64(file_path, max_size=1024)

        # Construct the prompt.
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{isbn_detection}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        batch_prompts.append(
            {"prompt": prompt, "multi_modal_data": {"image": base64_image}}
        )
        batch_metadata.append((document_folder, file_path))

        # When we reach the batch size, process this batch.
        if len(batch_prompts) >= CHUNK_SIZE:
            try:
                outputs = llm.generate(batch_prompts, sampling_params)
                for (doc_folder, file_path), result in zip(batch_metadata, outputs):
                    # Skip if ISBN already found for this document.
                    if doc_folder in docs_with_isbn_found:
                        continue
                    output_text = result.outputs[0].text
                    if "ISBN Present" in output_text:
                        logging.info(f"ISBN found in: {file_path}")
                        print(f"ISBN found in: {file_path}")
                        isbn_found.append(file_path)
                        with open(isbn_found_file, "a", encoding="utf-8") as f:
                            f.write(file_path + "\n")
                        docs_with_isbn_found.add(doc_folder)
            except Exception as e:
                logging.error(f"Error processing batch: {e}")

            # Clear the current batch lists.
            batch_prompts = []
            batch_metadata = []

    # Mark the document as processed after going through all its images.
    processed_documents.add(document_folder)
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(document_folder + "\n")

# Process any remaining prompts in the final batch.
if batch_prompts:
    try:
        outputs = llm.generate(batch_prompts, sampling_params)
        for (doc_folder, file_path), result in zip(batch_metadata, outputs):
            if doc_folder in docs_with_isbn_found:
                continue
            output_text = result.outputs[0].text
            if "ISBN Present" in output_text:
                logging.info(f"ISBN found in: {file_path}")
                print(f"ISBN found in: {file_path}")
                isbn_found.append(file_path)
                with open(isbn_found_file, "a", encoding="utf-8") as f:
                    f.write(file_path + "\n")
                docs_with_isbn_found.add(doc_folder)
    except Exception as e:
        logging.error(f"Error processing final batch: {e}")

end_time = time.time()
total_time = end_time - start_time

logging.info(f"Total processing time: {total_time:.2f} seconds.")
print(f"Total processing time: {total_time:.2f} seconds. Logs saved to {log_file}.")
