import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import ARTIFACTS_DIR

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_embedding_model():
    return SentenceTransformer(MODEL_NAME, device="cpu")

def encode_chunks(chunks, model):
    return model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def build_embeddings():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    small = json.loads((ARTIFACTS_DIR / "small_chunks.json").read_text(encoding="utf-8"))
    large = json.loads((ARTIFACTS_DIR / "large_chunks.json").read_text(encoding="utf-8"))
    model = load_embedding_model()
    small_vectors = encode_chunks(small, model)
    large_vectors = encode_chunks(large, model)
    np.save(ARTIFACTS_DIR / "small_embeddings.npy", small_vectors)
    np.save(ARTIFACTS_DIR / "large_embeddings.npy", large_vectors)
    return model, small_vectors, large_vectors

if __name__ == "__main__":
    build_embeddings()
    print("Saved embedding matrices.")
