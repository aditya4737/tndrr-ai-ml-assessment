import json
import re
import numpy as np
from pathlib import Path

from src.config import ARTIFACTS_DIR
from src.embeddings import load_embedding_model
from src.retrieval import RAGRetriever
from src.generation import AnswerGenerator


BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "evaluation_questions.json"


def normalize(text):
    text = re.sub(r"-\s+", "", str(text))
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def load_questions():
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def evaluate_dense_retrieval(
    questions, chunks, embeddings, embedding_model, top_k=5
):
    results = []
    for item in questions:
        query_embedding = embedding_model.encode(
            [item["question"]],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        scores = embeddings @ query_embedding
        indices = np.argsort(scores)[::-1][:top_k]
        retrieved = [chunks[int(i)] for i in indices]
        anchor = normalize(item["gold_anchor"])
        matching_ranks = [
            rank for rank, chunk in enumerate(retrieved, 1)
            if anchor in normalize(chunk["text"])
        ]
        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold_page": item["gold_page"],
            "retrieved_pages": [c["page"] for c in retrieved],
            "gold_passage_found": bool(matching_ranks),
            "matching_ranks": matching_ranks
        })

    hits = sum(x["gold_passage_found"] for x in results)
    return results, hits, hits / len(results) if results else 0.0


def evaluate_hybrid_retrieval(retriever, questions):
    results = []
    for item in questions:
        retrieved = retriever.retrieve(
            item["question"], candidate_k=10, top_k=6
        )
        anchor = normalize(item["gold_anchor"])
        matching_ranks = [
            rank for rank, result in enumerate(retrieved, 1)
            if anchor in normalize(result["chunk"]["text"])
        ]
        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold_page": item["gold_page"],
            "retrieved_pages": [
                x["chunk"]["page"] for x in retrieved
            ],
            "gold_passage_found": bool(matching_ranks),
            "matching_ranks": matching_ranks
        })

    hits = sum(x["gold_passage_found"] for x in results)
    return results, hits, hits / len(results) if results else 0.0


def save_json(filename, data):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved: {path}")


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()

    print("\nLoading chunk datasets...")
    small_chunks = json.loads(
        (ARTIFACTS_DIR / "small_chunks.json").read_text(encoding="utf-8")
    )
    large_chunks = json.loads(
        (ARTIFACTS_DIR / "large_chunks.json").read_text(encoding="utf-8")
    )
    small_embeddings = np.load(ARTIFACTS_DIR / "small_embeddings.npy")
    large_embeddings = np.load(ARTIFACTS_DIR / "large_embeddings.npy")
    embedding_model = load_embedding_model()

    print("\n" + "=" * 70)
    print("SMALL CHUNKS: 150 WORDS / 30 OVERLAP")
    print("=" * 70)
    small_results, small_hits, small_recall = evaluate_dense_retrieval(
        questions, small_chunks, small_embeddings, embedding_model, 5
    )
    print(f"Small chunks Recall@5: {small_hits}/{len(questions)} = {small_recall:.2%}")
    save_json("small_chunk_evaluation.json", small_results)

    print("\n" + "=" * 70)
    print("LARGE CHUNKS: 300 WORDS / 50 OVERLAP")
    print("=" * 70)
    large_results, large_hits, large_recall = evaluate_dense_retrieval(
        questions, large_chunks, large_embeddings, embedding_model, 5
    )
    print(f"Large chunks Recall@5: {large_hits}/{len(questions)} = {large_recall:.2%}")
    save_json("large_chunk_evaluation.json", large_results)

    print("\n" + "=" * 70)
    print("COLAB-ALIGNED EXPANDED HYBRID + CROSS-ENCODER")
    print("=" * 70)
    retriever = RAGRetriever()
    hybrid_results, hybrid_hits, hybrid_recall = evaluate_hybrid_retrieval(
        retriever, questions
    )
    print(f"Hybrid Recall@6: {hybrid_hits}/{len(questions)} = {hybrid_recall:.2%}")
    save_json("final_retrieval_evaluation.json", hybrid_results)

    failures = [x for x in hybrid_results if not x["gold_passage_found"]]
    print("\nRETRIEVAL FAILURES")
    for failure in failures:
        print(
            failure["question"],
            "| gold page:", failure["gold_page"],
            "| retrieved:", failure["retrieved_pages"]
        )
    save_json("retrieval_failures.json", failures)

    print("\n" + "=" * 70)
    print("GENERATING ANSWERS FOR 15 QUESTIONS")
    print("=" * 70)
    generator = AnswerGenerator(retriever, load_llm=True)
    answer_results = []

    for item in questions:
        result = generator.generate(item["question"])
        # .get() protects the evaluation from future optional metadata changes.
        record = {
            "id": item["id"],
            "question": item["question"],
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "evidence": result.get("evidence", []),
            "evidence_pages": result.get("evidence_pages", []),
            "method": result.get("method", ""),
            "llm_output": result.get("llm_output", ""),
            "answer_overlap": result.get("answer_overlap"),
            "retrieved_pages": result.get("retrieved_pages", [])
        }
        answer_results.append(record)

        print("\n" + "-" * 70)
        print("QUESTION:", item["question"])
        print("ANSWER:", record["answer"])
        print("SOURCE PAGES:", record["sources"])
        print("METHOD:", record["method"])

    save_json("final_answers.json", answer_results)

    summary = {
        "total_questions": len(questions),
        "small_chunks": {
            "chunk_size": 150,
            "overlap": 30,
            "recall_at_5": small_recall,
            "hits": small_hits
        },
        "large_chunks": {
            "chunk_size": 300,
            "overlap": 50,
            "recall_at_5": large_recall,
            "hits": large_hits
        },
        "hybrid_reranking": {
            "candidate_count": 10,
            "dense_bm25_pool_per_query": 50,
            "final_top_k": 6,
            "recall_at_6": hybrid_recall,
            "hits": hybrid_hits
        },
        "retrieval_failures": len(failures)
    }
    save_json("evaluation_summary.json", summary)

    print("\n" + "=" * 70)
    print("TASK 2 EVALUATION COMPLETED")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
