import json
import requests
from pypdf import PdfReader
from .config import DOCUMENTS_DIR, PDF_URL, PDF_PATH, PAGES_PATH

def ingest_document():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PDF_PATH.exists():
        print("Downloading official NIST AI RMF 1.0...")
        response = requests.get(PDF_URL, timeout=120)
        response.raise_for_status()
        PDF_PATH.write_bytes(response.content)

    reader = PdfReader(str(PDF_PATH))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_number, "text": text})

    if len(reader.pages) != 48 or len(pages) != 48:
        raise ValueError(
            f"Expected 48 pages with text; got {len(reader.pages)} PDF pages "
            f"and {len(pages)} extracted pages."
        )

    PAGES_PATH.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"PDF saved: {PDF_PATH}")
    print(f"Extracted {len(pages)} pages: {PAGES_PATH}")
    return pages

if __name__ == "__main__":
    ingest_document()
