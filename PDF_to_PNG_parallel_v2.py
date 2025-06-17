import os
import fitz  # PyMuPDF: pip install pymupdf
import json
import logging
import time
from tqdm import tqdm  # pip install tqdm
import concurrent.futures
import multiprocessing

# Paths for checkpoint and log files
CHECKPOINT_PATH = "checkpoints/checkpoint_pdf_to_images.json"
LOG_PATH = "logs/pdf_to_images.log"

# ---------------------------------------------
# Configure the logger
# ---------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler for logging
file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler to display logs in the terminal
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_checkpoint():
    """
    Loads the processing checkpoint from a JSON file.
    """
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Checkpoint file {CHECKPOINT_PATH} is empty or corrupted. Resetting.")
            return {}
    return {}

def save_checkpoint(checkpoint_data):
    """
    Saves the processing checkpoint to a JSON file.
    """
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

def get_pdf_page_count(pdf_path):
    """
    Retrieves the total number of pages in a PDF using PyMuPDF (fitz).
    """
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = pdf_document.page_count
        pdf_document.close()
        return total_pages
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return None

def save_image(image_pix, output_folder, page_type, page_number):
    """
    Saves an image (PyMuPDF pixmap) to the specified output folder as a PNG.
    """
    filename = os.path.join(output_folder, f"{page_type}_page_{page_number}.png")
    try:
        image_pix.save(filename, "PNG")
        logger.info(f"Saved image: {filename}")
    except Exception as e:
        logger.error(f"Error saving {page_type} page {page_number}: {e}")

def convert_pdf_to_images_with_fitz(pdf_path, output_folder, head_pages=10, tail_pages=10, dpi=600):
    """
    Converts the first `head_pages` and last `tail_pages` of a PDF to png images using PyMuPDF.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_folder = os.path.join(output_folder, pdf_name)
    os.makedirs(pdf_output_folder, exist_ok=True)

    total_pages = get_pdf_page_count(pdf_path)
    if total_pages is None:
        logger.error(f"Skipping PDF due to error in reading page count: {pdf_path}")
        return False  # Mark as failed in the checkpoint

    if head_pages + tail_pages > total_pages:
        tail_pages = max(0, total_pages - head_pages)

    try:
        pdf_document = fitz.open(pdf_path)

        # Convert head pages
        for page_num in range(min(head_pages, total_pages)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            save_image(pix, pdf_output_folder, "head", page_num + 1)

        # Convert tail pages if applicable
        if tail_pages > 0:
            for page_num in range(total_pages - tail_pages, total_pages):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(dpi=dpi)
                save_image(pix, pdf_output_folder, "tail", page_num - (total_pages - tail_pages) + 1)

        pdf_document.close()
        logger.info(f"Successfully processed PDF: {pdf_path}")
        return True  # Successfully processed
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return False  # Mark as failed in the checkpoint

def process_pdf_file(root, filename, relative_path, output_folder, head_pages, tail_pages, checkpoint_data):
    """
    Processes a single PDF file and updates the checkpoint.
    """
    pdf_path = os.path.join(root, filename)
    pdf_output_folder = os.path.join(output_folder, os.path.dirname(relative_path))
    os.makedirs(pdf_output_folder, exist_ok=True)

    logger.info(f"Processing PDF: {relative_path}")
    success = convert_pdf_to_images_with_fitz(
        pdf_path, pdf_output_folder, head_pages, tail_pages
    )

    # Update checkpoint based on success or failure
    checkpoint_data[relative_path] = "processed" if success else "failed"
    save_checkpoint(checkpoint_data)

def convert_folder(input_folder, output_folder, head_pages=5, tail_pages=5):
    """
    Processes all PDFs in the input folder and its subdirectories and saves the output images in the specified output folder.
    Uses a checkpoint to process only missing or failed files.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load the checkpoint data
    checkpoint_data = load_checkpoint()

    # Create a set of already processed PDFs from the checkpoint
    already_processed = {filename for filename, status in checkpoint_data.items() if status == "processed"}

    # Gather all PDFs to process
    pdf_files = []
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                relative_path = os.path.relpath(os.path.join(root, filename), input_folder)
                if relative_path not in already_processed:
                    pdf_files.append((root, filename, relative_path))

    logger.info(f"Found {len(pdf_files)} PDFs to process.")

    # Use half of the available CPU cores
    #num_workers = max(1, multiprocessing.cpu_count() // 2)
    num_workers = max(1, multiprocessing.cpu_count())

    # Track progress with tqdm
    start_time = time.time()
    with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file") as progress_bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for root, filename, relative_path in pdf_files:
                future = executor.submit(
                    process_pdf_file,
                    root, filename, relative_path, output_folder, head_pages, tail_pages, checkpoint_data
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Ensure any exceptions are raised
                except Exception as e:
                    logger.error(f"Error processing PDF: {e}")
                progress_bar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Processing complete. Total execution time: {total_time:.2f} seconds.")
    print(f"Total execution time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    input_folder = "/raid/homes/luca.sala/data/small_VLM_test/documents_raw/"
    output_folder = "/raid/homes/luca.sala/data/small_VLM_test/documents_with_images"

    convert_folder(input_folder, output_folder, head_pages=10, tail_pages=10)