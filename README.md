# UANM

# Arabic-Book Toolkit 📚🚀

A modular Python pipeline for large-scale digitisation of Arabic books.

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `PDF_to_PNG_parallel_v2.py` | Convert PDF pages to PNGs in parallel |
| 2 | `frontispiece_tagging_multipage.py` | Detect & tag front-matter pages |
| 3 | `index_tagging_multipage.py` | Detect & tag index pages |
| 4 | `ISBN_tagging.py` | Detect & tag ISBN pages |
| 5 | `tag_empty_pages_v6.py` | Identify & label blank pages |
| 6 | `visionAPI_v4.py` | Perform OCR with Google Cloud Vision |
| 7 | `find_and_copy_ISBN.py` | Copy only folders that contain an ISBN |
| 8 | `match_ISBN.py` | Match extracted ISBNs to the catalogue |

> Each script writes **check-points** and **logs** so you can resume work without wasting compute time.

---

## Requirements

| Category | Key packages |
|----------|--------------|
| Core     | `python>=3.10`, `pandas`, `tqdm`, `opencv-python` |
| Imaging  | `pymupdf`, `Pillow`, `google-cloud-vision` |
| LLM/Vision | `torch`, `vllm`, **Qwen2-VL-72B-Instruct** (or compatible) |
| System   | Linux, CUDA-capable GPU recommended |

Install everything with:

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
