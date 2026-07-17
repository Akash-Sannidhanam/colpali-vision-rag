"""Central configuration: env vars, model names, paths, and Qdrant settings."""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent
PDFS_DIR = ROOT_DIR / "pdfs"
PAGE_IMAGES_DIR = ROOT_DIR / "page_images"
# Cropped/annotated chart slices produced at answer time. Under page_images/ but
# never re-ingested (ingest only globs pdfs/).
CROPS_DIR = PAGE_IMAGES_DIR / "crops"
QDRANT_PATH = ROOT_DIR / "qdrant_data"

# Directory containing poppler's CLI tools (pdftoppm/pdftocairo), used by pdf2image.
# Override via POPPLER_PATH; otherwise derive it from pdftoppm on PATH.
# None lets pdf2image fall back to PATH (works on Linux where poppler-utils is installed system-wide).
_poppler_bin = shutil.which("pdftoppm")
POPPLER_PATH = os.getenv("POPPLER_PATH") or (str(Path(_poppler_bin).parent) if _poppler_bin else None)

# colqwen2-v1.0 (~2B) fits comfortably in 8 GB VRAM.
# For higher chart/table accuracy on a bigger GPU, use "vidore/colqwen2.5-v0.2".
COLPALI_MODEL = "vidore/colqwen2-v1.0"
RENDER_DPI = 150

COLLECTION_NAME = "pdf_pages"
VECTOR_DIM = 128 # ColQwen emits one 128-d vector per patch
TOP_K = 3
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")