#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, cv2, json, time, logging, argparse
import pandas as pd
from tqdm import tqdm
from google.cloud import vision_v1 as vision
from google.cloud.vision_v1 import AnnotateImageResponse

# ---------------- Batch config ------------------------------------------------
BATCH_SIZE             = 16            # n. max immagini “di partenza”
MAX_PAYLOAD_BYTES      = 41_943_040    # 40 MB (limite Vision API)
MAX_BATCHES_PER_MINUTE = 112           # 112×16 = 1792 img/min < 1800
SLEEP_BETWEEN_BATCHES  = 60 / MAX_BATCHES_PER_MINUTE   # ≈ 0.536 s

# ---------------- CLI ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="OCR batch (payload ≤ 40 MB) con Google Vision AI"
)
parser.add_argument("--input_folder",  required=True, help="Dir con le immagini")
parser.add_argument("--output_folder", required=True, help="Dir output risultati")
parser.add_argument("--credentials",   default="credentials/vision.json",
                    help="JSON credenziali GCP")
args = parser.parse_args()

# --- log / checkpoint personalizzati per cartella ----------------------------
input_tag      = os.path.basename(os.path.normpath(args.input_folder))
CHECKPOINT_PATH = f"checkpoints/checkpoint_{input_tag}.json"
LOG_PATH        = f"logs/log_{input_tag}.log"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.credentials
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(filename=LOG_PATH, filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')

# ---------------- Struttura risultati ----------------------------------------
results = {k: [] for k in
           ("sample_id", "page_confidence", "block_index",
            "block_coordinates", "text", "confidence")}

# ---------------- Utility -----------------------------------------------------
def numerical_sort_key(fn): m = re.search(r'\d+', fn); return int(m.group()) if m else 0
def load_ckpt(p):  return json.load(open(p)) if os.path.exists(p) else {}
def save_ckpt(d,p): os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(d, open(p,'w'), indent=2)
def ok_img(fp):    return os.path.exists(fp) and os.path.getsize(fp) > 0 and cv2.imread(fp) is not None

# ---------- batch builder: rispetta 40 MB ------------------------------------
def build_safe_batch(paths):
    """Ritorna (requests, selected_paths) che non superano MAX_PAYLOAD_BYTES."""
    requests, sel, total = [], [], 0
    for fp in paths:
        with open(fp, "rb") as f:
            img_bytes = f.read()
        img_size = len(img_bytes)
        if total + img_size > MAX_PAYLOAD_BYTES:
            break
        requests.append(
            vision.AnnotateImageRequest(
                image=vision.Image(content=img_bytes),
                features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
            )
        )
        sel.append(fp)
        total += img_size
    return requests, sel

# ---------------- Salvataggi helper ------------------------------------------
def save_individual(image_name, img, blocks, txt, jresp, parent, out_root):
    out_dir = os.path.join(out_root, parent, image_name)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{image_name}_visionai.png"), img)
    pd.DataFrame(blocks).to_excel(os.path.join(out_dir, f"{image_name}_ocr_output.xlsx"), index=False)
    open(os.path.join(out_dir, f"{image_name}_extracted_text.txt"), "w", encoding="utf-8").write(txt)
    json.dump(jresp, open(os.path.join(out_dir, f"{image_name}_response.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

def write_summary(out_root):
    pd.DataFrame(results).to_csv(os.path.join(out_root, "results_summary.csv"),
                                 index=False, encoding="utf-8")

# ---------------- Parsing risposta singola ------------------------------------
def handle_response(img_path, response, out_root):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    parent   = os.path.basename(os.path.dirname(img_path))
    if not ok_img(img_path):
        raise ValueError("Immagine non valida")
    original = cv2.imread(img_path)

    jresp     = json.loads(AnnotateImageResponse.to_json(response))
    full_text = jresp.get("fullTextAnnotation", {})
    extracted = full_text.get("text", "")
    blocks_out = []

    for page in full_text.get("pages", []):
        page_conf = round(page.get("confidence", 0.0), 2)
        for i, block in enumerate(page.get("blocks", [])):
            v = block["boundingBox"]["vertices"]
            x, y = min(pt.get("x",0) for pt in v), min(pt.get("y",0) for pt in v)
            w, h = max(pt.get("x",0) for pt in v)-x, max(pt.get("y",0) for pt in v)-y
            cv2.rectangle(original, (x,y), (x+w,y+h), (255,0,0), 2)

            blk_text = " ".join(
                "".join(sym["text"] for sym in word["symbols"])
                for par in block["paragraphs"] for word in par["words"]
            ).strip()
            blk_conf = round(block.get("confidence", 0.0), 2)

            blocks_out.append({
                "sample_id": img_name, "block_index": i,
                "page_confidence": page_conf, "block_confidence": blk_conf,
                "block_coordinates": (x,y,w,h), "text": blk_text
            })

            # summary
            results["sample_id"].append(img_name)
            results["page_confidence"].append(page_conf)
            results["block_index"].append(i)
            results["block_coordinates"].append((x,y,w,h))
            results["text"].append(blk_text)
            results["confidence"].append(blk_conf)

    save_individual(img_name, original, blocks_out, extracted, jresp, parent, out_root)

# ---------------- Batch processing -------------------------------------------
def process_batches(in_root, out_root, retries=3, retry_delay=5):
    client            = vision.ImageAnnotatorClient()
    checkpoint        = load_ckpt(CHECKPOINT_PATH)
    curr_batch_size   = BATCH_SIZE   # variabile locale adattiva

    # ---- raccolta immagini ---------------------------------------------------
    all_imgs = []
    for r, _, files in os.walk(in_root):
        files.sort(key=numerical_sort_key)
        all_imgs += [os.path.join(r, f)
                     for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    todo = [fp for fp in all_imgs if checkpoint.get(fp) not in ("processed", "failed")]
    logging.info(f"Totale immagini: {len(all_imgs)} | Da processare: {len(todo)}")

    pbar, idx = tqdm(total=len(todo), desc=f"Vision AI • {input_tag}", unit="img"), 0

    while idx < len(todo):
        responses = None                              # reset a ogni ciclo
        slice_paths = todo[idx: idx + curr_batch_size]
        requests, batch_paths = build_safe_batch(slice_paths)

        # singola immagine >40 MB
        if not requests:
            big = slice_paths[0]
            logging.error(f"File >40 MB, saltato: {big}")
            checkpoint[big] = "failed"
            save_ckpt(checkpoint, CHECKPOINT_PATH)
            idx += 1
            pbar.update(1)
            continue

        # ---------- invio con retry ------------------------------------------
        for attempt in range(1, retries + 1):
            try:
                responses = client.batch_annotate_images(requests=requests).responses
                break
            except Exception as e:
                msg = str(e)
                logging.error(f"Batch failed (tentativo {attempt}): {msg}")

                if "Request payload size exceeds the limit" in msg:
                    curr_batch_size = max(1, curr_batch_size // 2)
                    logging.warning(f"Ridimensiono batch a {curr_batch_size} immagini")
                    # ricostruisci slice e requests con batch più piccolo
                    slice_paths = slice_paths[:curr_batch_size]
                    requests, batch_paths = build_safe_batch(slice_paths)
                    time.sleep(retry_delay * attempt)
                    continue
                elif attempt < retries:
                    time.sleep(retry_delay * attempt)
                    continue
                else:
                    for fp in batch_paths:
                        checkpoint[fp] = "failed"
                    save_ckpt(checkpoint, CHECKPOINT_PATH)
                    pbar.update(len(batch_paths))
                    break   # esce dal retry-loop

        # batch fallito definitivamente
        if responses is None:
            idx += len(batch_paths)
            continue

        # ---------- gestione risposte individuali ----------------------------
        for fp, resp in zip(batch_paths, responses):
            try:
                if resp.error.message:
                    raise RuntimeError(resp.error.message)
                handle_response(fp, resp, out_root)
                checkpoint[fp] = "processed"
            except Exception as e:
                checkpoint[fp] = "failed"
                logging.error(f"Errore {fp}: {e}")

        save_ckpt(checkpoint, CHECKPOINT_PATH)
        pbar.update(len(batch_paths))
        idx += len(batch_paths)
        time.sleep(SLEEP_BETWEEN_BATCHES)

    pbar.close()
    write_summary(out_root)
    logging.info("Elaborazione completata")

# ---------------- Main -------------------------------------------------------
if __name__ == "__main__":
    process_batches(args.input_folder, args.output_folder)
