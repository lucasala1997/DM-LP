import os
import logging
import time
import multiprocessing  # Import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm  # Progress bar

# Constants
LOG_PATH = "logs/tag_white_images.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check if an image is a single color using OpenCV
def is_single_color_image_cv2(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False

        # Convert to RGB if needed
        if img.shape[-1] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Resize for faster processing
        h, w = img.shape[:2]
        if max(h, w) > 1000:
            img = cv2.resize(img, (w // 8, h // 8), interpolation=cv2.INTER_AREA)

        # Check unique colors
        unique_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        return len(unique_colors) == 1
    except Exception as e:
        logging.error(f"Error reading {image_path}: {e}")
        return False

# Process a single image and rename if it is a single color
def process_image(image_path):
    try:
        filename = os.path.basename(image_path)
        if "white_page_" in filename:
            return False  # Skip already renamed files

        if is_single_color_image_cv2(image_path):
            new_path = os.path.join(os.path.dirname(image_path),
                                    f"white_page_{filename}")
            os.rename(image_path, new_path)
            return True
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
    return False

# Generator to find PNG files in a directory
def find_png_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                yield os.path.join(root, file)

# Main function to process files
def main(directory):
    start_time = time.time()

    # Find all PNG files
    png_files = list(find_png_files(directory))
    total_files = len(png_files)
    print(f"Found {total_files} PNG files.")

    # Determine the number of workers
    max_workers = multiprocessing.cpu_count()
    print(f"Using {max_workers} workers.")
    logging.info(f"Number of CPUs used: {max_workers}. Processing...")
    renamed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary of futures and files
        futures = {executor.submit(process_image, path): path for path in png_files}

        # Use tqdm for a progress bar
        for future in tqdm(as_completed(futures), total=total_files, desc="Processing images"):
            try:
                if future.result():
                    renamed_count += 1
            except Exception as e:
                logging.error(f"Error in future: {e}")

    elapsed_time = time.time() - start_time
    print(f"Renamed {renamed_count} files.")
    print(f"Time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    input_folder = "/raid/homes/luca.sala/data/small_VLM_test/documents_with_images"
    main(input_folder)
