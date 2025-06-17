import os
import time
import logging
import base64
import json
from io import BytesIO
from PIL import Image

# Imposta le GPU da utilizzare (,2,5,6,7)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,5,7"

import torch
from tqdm import tqdm

# vLLM imports
from vllm import LLM, SamplingParams
import PIL

# -------------------------- CONFIGURATION -------------------------- #
PIL.Image.MAX_IMAGE_PIXELS = None

# Folders & paths (MODIFICATI)
data_folder = "/raid/homes/luca.sala/chunk_4"
result_folder = "/raid/homes/luca.sala/chunk_4_index_multipage"

checkpoint_file = os.path.join(result_folder, "checkpoint.txt")
index_found_file = os.path.join(result_folder, "index_found.txt")
responses_file = os.path.join(result_folder, "responses.json")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# Logging
log_file = os.path.join(result_folder, "index_detection.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting the processing script with vLLM for index detection.")

# Set random seed for reproducibility
seed_value = 7
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Load checkpoint and index paths
processed_documents = set()
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        processed_documents = set(line.strip() for line in f)

index_found = []
if os.path.exists(index_found_file):
    with open(index_found_file, "r") as f:
        index_found = [line.strip() for line in f]

# Load previous responses if exist
all_responses = {}
if os.path.exists(responses_file):
    with open(responses_file, "r", encoding="utf-8") as f:
        all_responses = json.load(f)

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

# #v7.2
# index_detection = (

#     "Task: Search the image of an Arabic book page for any of the following keywords:"
#     "فهرس", "دليل الكتاب", "المحتويات", "موضوع", "الفهارس", "فهرس المحتويات","فهرست", "فهرست مطالب", "فهرس الموضوعات", "محتويات الكتاب", "مواضیع الكتاب", "المواضيع" 
#     "\n"
    
#     "Start strictly from the last page. Analyze pages sequentially backward (one page at a time). Prioritize checking the final 10% of the book.\n\n"

#     "Focus intensively on pages showing strong index-like visual layouts (multi-column format, dotted leader lines, topic-title structure). Minimize processing on normal prose pages.\n\n"
    
#     "DETECTION RULES:\n"
#     "- **Do NOT** consider pages listing symbols, abbreviations, or glossary unless clear page numbers and topic titles are present.\n"
#     "- **Mandatory:** Detected lines must include a topic or heading (noun phrase) followed by dotted leaders (........) and an Arabic numeral (e.g., ١٢٣) representing a page number.\n"
#     "- **Mandatory:** The page must reference multiple topics, not just symbols or codes.\n"
#     "- **If the page title contains words like 'رموز', 'اختصارات', 'مصطلحات'**, assume it is NOT an index unless strong contrary evidence appears.\n\n"
    
#     "Terminate search immediately when a correct index page is identified.\n\n"

#     # === DESIRED OUTPUT ===
    
#     "If a valid index page is detected, respond with exactly one line in the following format:\n"
    
#     "- index|<Confidence Level>|<Strongest Visual Cue>\n"
#     "- not index|<Confidence Level>|<Strongest Visual Cue>\n\n"
    
#     "Confidence Level must be: High, Medium, or Low.\n"
#     "Visual Cue should briefly describe the key indicator (e.g., 'dotted leaders with page numbers', 'symbol list', etc.).\n\n"
    
#     # === FORMAT EXAMPLES ===
#     "index|High|dotted leaders with page numbers\n"
#     "not index|Low|symbol glossary without page numbers\n"
# )

# v7.1
index_detection= (

    "Task: Search in the image of an Arabic book page for any of the following keywords:"
    "فهرس", "دليل الكتاب", "المحتويات", "موضوع", "الفهارس", "فهرس المحتويات","فهرست", "فهرست مطالب", "فهرس الموضوعات", "محتويات الكتاب", "مواضیع الكتاب", "المواضيع" 
    "\n"
    
    "Start the search strictly from the last page of the book. Analyze pages sequentially backward (moving toward the front). Prioritize inspection within the final 10% of the book's pages first.\n\n"

    "Focus attention more intensively on pages with strong visual layout cues suggesting an index (e.g., multi-column layout, dotted leader lines, structured headings). Minimize token and visual processing for pages with typical prose formatting (single text block without columns or lists).\n\n"

    "Terminate the search immediately upon confident identification of an index page.\n\n"

    # === DESIRED OUTPUT ===
    
    "If a keyword appears clearly near the top of the page, respond with exactly **one line** using the following format:\n"
    
    "- For an index page:\n"
    "    index|<Confidence Level>|<Strongest Visual Cue>\n"
    
    "- For a non-index page:\n"
    "    not index|<Confidence Level>|<Strongest Visual Cue>\n\n"
    
    "Confidence Level must be one of: High, Medium, or Low.\n"
    "The Visual Cue should be a very short phrase describing the most decisive visual feature observed.\n"
    
    # === FORMAT EXAMPLES (do NOT include extra text) ===
    "index|High|two columns with dotted leaders\n"
    "not index|Low|single prose column\n"
)

# -------------------------- INITIALIZE vLLM MODEL -------------------------- #
llm = LLM(
    model="/raid/homes/luca.sala/models/Qwen/Qwen2-VL-72B-Instruct",
    tensor_parallel_size=4,
    dtype=torch.bfloat16,
    enable_prefix_caching=True,
    max_model_len=8192,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    max_tokens=128,
    temperature=0.0,
    top_p=1.0,
    top_k=50,
    repetition_penalty=1.0,
)

# -------------------------- MAIN LOGIC -------------------------- #

start_time = time.time()
CHUNK_SIZE = 1000

docs_with_index_found = set()   # mantiene i riferimenti ma non interrompe più il flusso
batch_prompts = []
batch_metadata = []

for root, _, files in tqdm(os.walk(data_folder)):
    document_folder = os.path.abspath(root)
    if document_folder in processed_documents:
        continue

    image_files = sorted(
        f
        for f in files
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    )

    if not image_files:
        processed_documents.add(document_folder)
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(document_folder + "\n")
        continue

    for file_name in image_files:
        # --- rimosso lo skip basato su docs_with_index_found ---

        file_path = os.path.join(document_folder, file_name)
        base64_image = resize_image_to_base64(file_path, max_size=1024)

        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{index_detection}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        batch_prompts.append(
            {"prompt": prompt, "multi_modal_data": {"image": base64_image}}
        )
        batch_metadata.append((document_folder, file_path))

        if len(batch_prompts) >= CHUNK_SIZE:
            try:
                outputs = llm.generate(batch_prompts, sampling_params)
                for (doc_folder, f_path), result in zip(batch_metadata, outputs):
                    output_text = result.outputs[0].text
                    all_responses.setdefault(doc_folder, {})[f_path] = output_text
                    # --- rimosso lo skip qui ---
                    if output_text.strip().lower().startswith("index"):
                        logging.info(f"index found in: {f_path}")
                        print(f"index found in: {f_path}")
                        index_found.append(f_path)
                        with open(index_found_file, "a", encoding="utf-8") as ff:
                            ff.write(f_path + "\n")
                        docs_with_index_found.add(doc_folder)
            except Exception as e:
                logging.error(f"Error processing batch: {e}")

            batch_prompts = []
            batch_metadata = []

    processed_documents.add(document_folder)
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(document_folder + "\n")

if batch_prompts:
    try:
        outputs = llm.generate(batch_prompts, sampling_params)
        for (doc_folder, f_path), result in zip(batch_metadata, outputs):
            output_text = result.outputs[0].text
            all_responses.setdefault(doc_folder, {})[f_path] = output_text
            # --- rimosso lo skip qui ---
            if output_text.strip().lower().startswith("index"):
                logging.info(f"index found in: {f_path}")
                print(f"index found in: {f_path}")
                index_found.append(f_path)
                with open(index_found_file, "a", encoding="utf-8") as ff:
                    ff.write(f_path + "\n")
                docs_with_index_found.add(doc_folder)
    except Exception as e:
        logging.error(f"Error processing final batch: {e}")

# Save all responses to JSON
with open(responses_file, "w", encoding="utf-8") as f:
    json.dump(all_responses, f, ensure_ascii=False, indent=2)

end_time = time.time()
total_time = end_time - start_time

logging.info(f"Total processing time: {total_time:.2f} seconds.")
print(f"Total processing time: {total_time:.2f} seconds. Logs saved to {log_file}. Responses saved to {responses_file}.")
