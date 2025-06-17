import os
import re
import shutil
import logging
import json

# ---------------------------------------------
# CONFIGURAZIONE DEL LOGGER
# ---------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Creiamo un file handler per salvare i log
file_handler = logging.FileHandler("logs/log_find_ISBN.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Percorso per il file di checkpoint
CHECKPOINT_PATH = "checkpoints/checkpoint_find_ISBN.json"

def load_checkpoint():
    """Carica i progressi dal file di checkpoint."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Il file di checkpoint '{CHECKPOINT_PATH}' è vuoto o corrotto. Ripristino.")
            return {}
    return {}

def save_checkpoint(checkpoint_data):
    """Salva i progressi nel file di checkpoint."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

def find_isbn_codes_in_text(text):
    """
    Trova tutti gli ISBN (o codici simili) presenti nel testo.

    Args:
        text (str): Contenuto del file di testo.

    Returns:
        list: Lista dei codici ISBN trovati (stringhe).
    """
    isbn_pattern = (
        r"\b(?:ISBN(?:-13)?|ISSN|ردمك|الترقيم الدولي)[-\s]?:?[-\s]?"
        r"((978[-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d{1})\b"
        r"|\b((978)?\d{9,12}[\dX])\b"
    )
    matches = re.findall(isbn_pattern, text)
    isbn_codes = [m[0] if m[0] else m[2] for m in matches if m[0] or m[2]]
    return isbn_codes

def process_directories(base_dir, target_dir):
    """
    Processa ciascuna cartella di documenti in base_dir, verificando
    se contengono file .txt con ISBN e, in caso positivo, copia la
    cartella in target_dir e crea un file isbn_codes.txt.

    Usa un checkpoint per tracciare i progressi.
    """
    if not os.path.exists(base_dir):
        logger.error(f"La directory di base '{base_dir}' non esiste.")
        print(f"[ERRORE] La directory di base '{base_dir}' non esiste.")
        return

    os.makedirs(target_dir, exist_ok=True)
    checkpoint_data = load_checkpoint()  # Carica i progressi
    cartelle_spostate = 0

    for document_folder in os.listdir(base_dir):
        document_path = os.path.join(base_dir, document_folder)

        if os.path.isdir(document_path):
            # Salta la cartella se già processata
            if checkpoint_data.get(document_folder) == "processed":
                logger.info(f"Cartella già processata: {document_folder}.")
                continue

            print(f"Analizzo la cartella: {document_folder} ...")
            logger.info(f"Analizzo la cartella '{document_folder}'.")

            contains_isbn = False
            isbn_codes_collected = []

            txt_files_found = False
            for root, _, files in os.walk(document_path):
                for file in files:
                    if file.endswith(".txt"):
                        txt_files_found = True
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                isbn_codes = find_isbn_codes_in_text(content)
                                if isbn_codes:
                                    contains_isbn = True
                                    isbn_codes_collected.extend(isbn_codes)
                        except FileNotFoundError:
                            logger.error(f"File non trovato: {file_path}")
                        except Exception as e:
                            logger.error(f"Impossibile leggere il file {file_path}: {e}")

            if not txt_files_found:
                logger.info(f"Nessun file .txt presente nella cartella '{document_folder}'.")
            elif not contains_isbn:
                logger.info(f"Nessun ISBN trovato nella cartella '{document_folder}'.")

            if contains_isbn:
                target_document_path = os.path.join(target_dir, document_folder)
                try:
                    shutil.copytree(document_path, target_document_path, dirs_exist_ok=True)
                    logger.info(f"Copiata la cartella '{document_folder}' in '{target_document_path}'.")

                    isbn_codes_collected = list(set(isbn_codes_collected))
                    isbn_file = os.path.join(target_document_path, "isbn_codes.txt")
                    with open(isbn_file, "w", encoding="utf-8") as isbnf:
                        for code in isbn_codes_collected:
                            isbnf.write(f"{code}\n")

                    cartelle_spostate += 1
                    checkpoint_data[document_folder] = "processed"  # Aggiorna il checkpoint
                    save_checkpoint(checkpoint_data)

                except Exception as e:
                    logger.error(f"Impossibile copiare '{document_path}' in '{target_document_path}': {e}")
                    checkpoint_data[document_folder] = "failed"
                    save_checkpoint(checkpoint_data)

    if cartelle_spostate == 0:
        logger.info("Non è stata spostata alcuna cartella.")
        print("[INFO] Non è stata spostata alcuna cartella.")

if __name__ == "__main__":
    base_directory = "/raid/homes/luca.sala/data/test2_HD1/documents_visionAI"
    target_directory = "/raid/homes/luca.sala/data/test2_HD1/documents_with_ISBN"
    process_directories(base_directory, target_directory)
