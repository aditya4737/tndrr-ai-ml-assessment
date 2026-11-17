import json, re
from .config import ARTIFACTS_DIR, PAGES_PATH

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def create_chunks(pages, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks, chunk_id = [], 0
    step = chunk_size - overlap
    for page_item in pages:
        words = clean_text(page_item["text"]).split()
        for start in range(0, len(words), step):
            segment = words[start:start + chunk_size]
            if not segment:
                continue
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_item["page"],
                "text": " ".join(segment)
            })
            chunk_id += 1
            if start + chunk_size >= len(words):
                break
    return chunks

def build_chunk_sets():
    pages = json.loads(PAGES_PATH.read_text(encoding="utf-8"))
    small = create_chunks(pages, 150, 30)
    large = create_chunks(pages, 300, 50)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("small_chunks", small), ("large_chunks", large)]:
        (ARTIFACTS_DIR / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return small, large

if __name__ == "__main__":
    small, large = build_chunk_sets()
    print(f"Created {len(small)} small and {len(large)} large chunks.")
