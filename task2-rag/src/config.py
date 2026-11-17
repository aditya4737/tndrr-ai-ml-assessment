from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PDF_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
PDF_PATH = DOCUMENTS_DIR / "nist_ai_rmf.pdf"
PAGES_PATH = DOCUMENTS_DIR / "extracted_pages.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL = "google/flan-t5-large"
