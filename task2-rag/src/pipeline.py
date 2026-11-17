from .ingestion import ingest_document
from .chunking import build_chunk_sets
from .embeddings import build_embeddings

def build_index():
    ingest_document()
    build_chunk_sets()
    build_embeddings()
    print("Task 2 index built.")

if __name__ == "__main__":
    build_index()
