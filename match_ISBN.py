import os
import csv
import pandas as pd
import logging
from tqdm import tqdm  # Import tqdm for progress bar
import json

# Path configurazioni
documents_with_isbn_path = "/raid/homes/luca.sala/data/test2_HD1/documents_with_ISBN"
export_csv_path = "/raid/homes/luca.sala/data/LaPira_export_19112024-2.csv"
output_csv_path = "/raid/homes/luca.sala/data/matching_isbn_results.csv"
log_path = "/raid/homes/luca.sala/scripts/create_dataset/logs/log_match_ISBN.log"
checkpoint_path = "/raid/homes/luca.sala/scripts/create_dataset/checkpoints/checkpoint_match_ISBN.json"

# Configurazione logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",  # Overwrite the log file each run
)

def read_exported_data(export_csv_path):
    """
    Legge i dati dal file CSV esportato.
    """
    try:
        # Legge il file CSV con delimitatore punto e virgola (;)
        export_df = pd.read_csv(
            export_csv_path,
            delimiter=";",
            dtype=str,
            on_bad_lines="skip",  # Skip malformed rows
            quoting=csv.QUOTE_ALL,  # Handle quoted fields
        )
        
        # Normalizza i nomi delle colonne
        export_df.columns = export_df.columns.str.strip().str.lower()

        # Assicura che esista la colonna ISBN
        if "isbn" not in export_df.columns:
            raise ValueError(f"Colonna 'ISBN' non trovata in {export_csv_path}.")
        
        logging.info(f"File CSV caricato correttamente: {export_csv_path}")
        return export_df
    except Exception as e:
        logging.error(f"Errore durante la lettura di {export_csv_path}: {e}")
        print(f"Errore durante la lettura di {export_csv_path}: {e}")
        return pd.DataFrame()  # Restituisce un DataFrame vuoto in caso di errore

def load_checkpoint(checkpoint_path):
    """
    Carica il checkpoint dal file JSON.
    """
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"File di checkpoint corrotto o vuoto: {checkpoint_path}.")
            return {}
    return {}

def save_checkpoint(checkpoint_data, checkpoint_path):
    """
    Salva il checkpoint nel file JSON.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

def check_and_append_results(documents_path, export_df, output_csv_path, checkpoint_path):
    """
    Controlla i codici ISBN in ciascuna sottocartella e li confronta con i dati esportati.
    Scrive i risultati (inclusa l'intera riga corrispondente) in un file CSV.
    Utilizza un checkpoint per riprendere il lavoro in caso di interruzioni.
    """
    # Crea il file di output se non esiste
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Scrive l'intestazione: Folder Name + tutte le colonne del CSV
            writer.writerow(["Folder Name"] + export_df.columns.tolist())
            logging.info(f"Creato file di output: {output_csv_path}")

    # Carica lo stato del checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)

    # Lista delle cartelle da processare
    folders = [folder_name for folder_name in os.listdir(documents_path) if os.path.isdir(os.path.join(documents_path, folder_name))]
    
    # Aggiungi la barra di avanzamento
    for folder_name in tqdm(folders, desc="Processing Folders", unit="folder"):
        folder_path = os.path.join(documents_path, folder_name)

        if folder_name in checkpoint_data:
            logging.info(f"Cartella già processata: {folder_name}. Skipping...")
            continue

        isbn_file_path = os.path.join(folder_path, "isbn_codes.txt")
        
        # Controlla se il file isbn_codes.txt esiste
        if os.path.exists(isbn_file_path):
            try:
                with open(isbn_file_path, "r", encoding="utf-8") as f:
                    isbn_codes = [line.strip() for line in f.readlines() if line.strip()]
                    logging.info(f"Trovati {len(isbn_codes)} ISBN nella cartella {folder_name}.")
                
                # Confronta ogni ISBN con quelli nel DataFrame esportato
                for isbn in isbn_codes:
                    matching_rows = export_df[export_df["isbn"] == isbn]
                    if not matching_rows.empty:
                        # Aggiunge al file CSV di output se non già presente
                        with open(output_csv_path, "r", encoding="utf-8") as f:
                            existing_data = f.read()
                        if isbn not in existing_data:
                            with open(output_csv_path, "a", encoding="utf-8", newline="") as f:
                                writer = csv.writer(f)
                                for _, row in matching_rows.iterrows():
                                    writer.writerow([folder_name] + row.tolist())
                                logging.info(f"Trovato e aggiunto: ISBN {isbn} nella cartella {folder_name}.")

                # Aggiorna il checkpoint
                checkpoint_data[folder_name] = "processed"
                save_checkpoint(checkpoint_data, checkpoint_path)

            except Exception as e:
                logging.error(f"Errore durante la lettura di {isbn_file_path}: {e}")
                print(f"Errore durante la lettura di {isbn_file_path}: {e}")
        else:
            logging.warning(f"File isbn_codes.txt non trovato nella cartella {folder_name}.")

if __name__ == "__main__":
    # Legge i dati dal file esportato
    export_df = read_exported_data(export_csv_path)

    if export_df.empty:
        logging.error("Nessun dato trovato nel file esportato. Verifica il file e riprova.")
        print("Nessun dato trovato nel file esportato. Verifica il file e riprova.")
    else:
        # Controlla e aggiorna i risultati
        check_and_append_results(documents_with_isbn_path, export_df, output_csv_path, checkpoint_path)
        logging.info(f"Controllo completato. Risultati salvati in {output_csv_path}.")
        print(f"Controllo completato. Risultati salvati in {output_csv_path}.")
