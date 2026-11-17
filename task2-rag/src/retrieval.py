import json
import re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import ARTIFACTS_DIR, RERANKER_MODEL
from .embeddings import load_embedding_model


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


QUERY_EXPANSIONS = {
    "govern": (
        "GOVERN governance function cultivates and implements a culture "
        "of AI risk management organizational policies processes procedures "
        "roles responsibilities accountability strategic priorities"
    ),
    "manage": (
        "MANAGE function entails allocating risk resources to mapped and "
        "measured AI risks risk treatment prioritize respond recover "
        "communicate incidents mitigate transfer avoid accept"
    )
}


class RAGRetriever:
    """
    Local reproduction of the Colab retrieval pipeline:
    expanded dense + BM25 retrieval (RRF), followed by cross-encoder reranking.
    """

    def __init__(self):
        self.chunks = json.loads(
            (ARTIFACTS_DIR / "small_chunks.json").read_text(
                encoding="utf-8"
            )
        )
        self.vectors = np.load(
            ARTIFACTS_DIR / "small_embeddings.npy"
        )

        self.embedding_model = load_embedding_model()
        self.bm25 = BM25Okapi(
            [tokenize(chunk["text"]) for chunk in self.chunks]
        )

        self.cross_encoder = CrossEncoder(
            RERANKER_MODEL,
            device="cpu"
        )
        # Keep the alias used by older local code.
        self.reranker = self.cross_encoder

    def expanded_hybrid_retrieve(
        self,
        question,
        top_k=10,
        candidate_k=50
    ):
        q_lower = question.lower()

        if "govern" in q_lower:
            expanded_query = QUERY_EXPANSIONS["govern"]
        elif "manage" in q_lower:
            expanded_query = QUERY_EXPANSIONS["manage"]
        else:
            expanded_query = question

        # Match the notebook: original and expanded semantic rankings.
        query_embeddings = self.embedding_model.encode(
            [question, expanded_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        semantic_rankings = []

        for query_embedding in query_embeddings:
            scores = np.dot(self.vectors, query_embedding)
            ranking = np.argsort(scores)[::-1][:candidate_k]
            semantic_rankings.append(ranking)

        # Match the notebook: original and expanded BM25 rankings.
        bm25_rankings = []

        for query in [question, expanded_query]:
            scores = self.bm25.get_scores(tokenize(query))
            ranking = np.argsort(scores)[::-1][:candidate_k]
            bm25_rankings.append(ranking)

        # Reciprocal Rank Fusion across all four rankings.
        rrf_scores = {}

        for ranking in semantic_rankings + bm25_rankings:
            for rank, idx in enumerate(ranking):
                idx = int(idx)
                rrf_scores[idx] = (
                    rrf_scores.get(idx, 0.0)
                    + 1.0 / (60 + rank + 1)
                )

        final_ranking = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )[:top_k]

        return [self.chunks[int(idx)] for idx in final_ranking]

    def retrieve(
        self,
        question,
        candidate_k=10,
        top_k=6
    ):
        # Colab: expanded_hybrid_retrieve(top_k=candidate_k,
        #                                 candidate_k=50)
        candidates = self.expanded_hybrid_retrieve(
            question,
            top_k=candidate_k,
            candidate_k=50
        )

        if not candidates:
            return []

        pairs = [
            (question, chunk["text"])
            for chunk in candidates
        ]

        scores = self.cross_encoder.predict(
            pairs,
            show_progress_bar=False
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        return [
            {
                "chunk": chunk,
                "reranker_score": float(score)
            }
            for chunk, score in ranked[:top_k]
        ]
