import os
import time
import logging
import base64
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
data_folder = "/raid/homes/luca.sala/DONE/chunk_1_images"
result_folder = "/raid/homes/luca.sala/chunk_1_frontispiece_multipage"

checkpoint_file = os.path.join(result_folder, "checkpoint.txt")
frontispiece_found_file = os.path.join(result_folder, "frontispiece_found.txt")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# Logging
log_file = os.path.join(result_folder, "frontispiece_detection.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting the processing script with vLLM for frontispiece detection.")

# Set random seed for reproducibility
seed_value = 7
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Load checkpoint and frontispiece paths
processed_documents = set()
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        processed_documents = set(line.strip() for line in f)

frontispiece_found = []
if os.path.exists(frontispiece_found_file):
    with open(frontispiece_found_file, "r") as f:
        frontispiece_found = [line.strip() for line in f]

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

# Prompt personalizzato (MODIFICATO)
frontispiece_detection = (
    "The preferred source of information is the title page (Frontispiece), or, for resources lacking a title page, the title-page substitute. If information traditionally given on the title page is given on facing pages,with or without repetition, the two pages are treated as the preferred source of information. The title page (Frontispiece) is a page normally placed at the beginning of a printed resource (usually black and white), presenting the most complete information about the resource and the works it contains. It contains a title and usually, though not necessarily, the most complete information on it, a formulation of responsibility as well as, in whole or in part, the publication formulation. To this is added a page with the title of a book, either abbreviated or in full, as printed on the front of a paper preceding the title page, usually in smaller type than the characters that make up the title itself on the title page. Based on what said, analyze this page and determine if it's a frontispiece of an Arabic book. A frontispiece is typically the decorative or illustrated page at the beginning of the book, often containing the title, author's name, and possibly publisher information or decorative elements. Other pages close to the frontispiece have usually useful informations and must be checked carefully. Information could deal with publishing rights, ISBN, publishing house contatcts as telephone numbers (preceded with the terms 'هاتف' or 'فاكس' or other). In some cases close to the title page you will find a page containing a CIP record (could be in a highlighted textbox or in a two column form). In this CIP there are the book's title ('عنوان الكتاب'), subtitle, and author(s) ('المؤلف' or 'تأليف'); ISBN code; Subject classifications (such as Dewey Decimal or Library of Congress Classification numbers etc.); Descriptive information like pagination, dimensions, and illustrations; Publication details (e.g., publisher and date).\n"
    "If any of those elemnts are present in any page close to the title page then consider it as 'Forntispiece' otherwise you should classify those pages as 'Not Frontispiece'. Only if the page is empty consider it as 'Not Frontispiece' Focus on:\n"
        "- Presence of distinctive Arabic typography or calligraphy for the title\n"
        "- Decorative borders, Islamic geometric patterns, or ornamental designs\n"
        "- Publisher logos or imprints\n"
        "- Author information placement\n"
        "- Barcodes \n"
        "- Body of text, which presents more lines of text and more characters, must not be considered as 'Frontispiece'\n\n"
        "Respond with either 'Frontispiece' or 'Not Frontispiece' followed by your confidence level (High/Medium/Low) and key visual elements that informed your decision."
)

# -------------------------- INITIALIZE vLLM MODEL -------------------------- #
llm = LLM(
    model="/raid/homes/luca.sala/models/Qwen/Qwen2-VL-72B-Instruct",  # o percorso locale del checkpoint
    tensor_parallel_size=4,  # Aggiornato per utilizzare le 5 GPU specificate
    dtype=torch.bfloat16,
    enable_prefix_caching=True,
    max_model_len=8192,
    enforce_eager=True,
)

# Generation parameters
sampling_params = SamplingParams(
    max_tokens=128,
    temperature=0.0,  # deterministico
    top_p=1.0,
    top_k=50,
    repetition_penalty=1.0,
)

# -------------------------- MAIN LOGIC -------------------------- #

start_time = time.time()

# Define batch (chunk) size
CHUNK_SIZE = 1000

# (la variabile rimane ma non serve più a saltare pagine)
docs_with_frontispiece_found = set()

# Lists for holding batch data
batch_prompts = []
batch_metadata = []  # each element is a tuple (document_folder, file_path)

# Iterate over each document folder in the data_folder.
for root, _, files in tqdm(os.walk(data_folder)):
    document_folder = os.path.abspath(root)
    if document_folder in processed_documents:
        continue

    # Filtra i file di immagini validi.
    image_files = sorted(
        f
        for f in files
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    )

    # Se non ci sono immagini valide, salta.
    if not image_files:
        processed_documents.add(document_folder)
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(document_folder + "\n")
        continue

    # Processa ciascuna immagine nella cartella.
    for file_name in image_files:
        # ---  RIMOSSA la logica di skip qui  ---

        file_path = os.path.join(document_folder, file_name)
        base64_image = resize_image_to_base64(file_path, max_size=1024)

        # Costruisce il prompt
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{frontispiece_detection}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        batch_prompts.append(
            {"prompt": prompt, "multi_modal_data": {"image": base64_image}}
        )
        batch_metadata.append((document_folder, file_path))

        # Se raggiungiamo il CHUNK_SIZE, processiamo il batch.
        if len(batch_prompts) >= CHUNK_SIZE:
            try:
                outputs = llm.generate(batch_prompts, sampling_params)
                for (doc_folder, f_path), result in zip(batch_metadata, outputs):
                    # --- RIMOSSA la logica di skip qui  ---
                    output_text = result.outputs[0].text
                    if output_text.strip().startswith("Frontispiece"):
                        logging.info(f"Frontispiece found in: {f_path}")
                        print(f"Frontispiece found in: {f_path}")
                        frontispiece_found.append(f_path)
                        with open(frontispiece_found_file, "a", encoding="utf-8") as ff:
                            ff.write(f_path + "\n")
                        docs_with_frontispiece_found.add(doc_folder)  # non influisce più
            except Exception as e:
                logging.error(f"Error processing batch: {e}")

            # Svuota le liste dopo l'elaborazione del batch
            batch_prompts = []
            batch_metadata = []

    # Conclusa la cartella, la segniamo come processata
    processed_documents.add(document_folder)
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(document_folder + "\n")

# Se rimangono prompt nell'ultimo batch, elaborali.
if batch_prompts:
    try:
        outputs = llm.generate(batch_prompts, sampling_params)
        for (doc_folder, f_path), result in zip(batch_metadata, outputs):
            # --- RIMOSSA la logica di skip qui  ---
            output_text = result.outputs[0].text
            if output_text.strip().startswith("Frontispiece"):
                logging.info(f"Frontispiece found in: {f_path}")
                print(f"Frontispiece found in: {f_path}")
                frontispiece_found.append(f_path)
                with open(frontispiece_found_file, "a", encoding="utf-8") as ff:
                    ff.write(f_path + "\n")
                docs_with_frontispiece_found.add(doc_folder)  # non influisce più
    except Exception as e:
        logging.error(f"Error processing final batch: {e}")

end_time = time.time()
total_time = end_time - start_time

logging.info(f"Total processing time: {total_time:.2f} seconds.")
print(f"Total processing time: {total_time:.2f} seconds. Logs saved to {log_file}.")
