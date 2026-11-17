# Task 2 — Retrieval-Augmented Question Answering

## 1. Objective

Build a Retrieval-Augmented Generation (RAG) system from scratch
over the NIST Artificial Intelligence Risk Management Framework
(AI RMF 1.0).

The system retrieves relevant passages from the document and
uses a language model to generate source-grounded answers.

No LangChain, LlamaIndex, or similar RAG frameworks are used.

## 2. Dataset

Source: NIST AI Risk Management Framework 1.0

Official PDF:
https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

- Total pages: 48
- Text extraction: pypdf
- Page numbers are preserved during ingestion and chunking.

## 3. Architecture

PDF Ingestion
    |
    v
Text Extraction
    |
    v
Text Chunking
    |
    v
Embedding Generation
    |
    v
Hybrid Retrieval
    |
    v
Reciprocal Rank Fusion
    |
    v
Cross-Encoder Reranking
    |
    v
Evidence Selection
    |
    v
FLAN-T5 Answer Generation
    |
    v
Source-Grounded Answer

## 4. Chunking Experiments

Two chunk configurations are compared:

| Configuration | Chunk Size | Overlap |
|---|---:|---:|
| Small | 150 words | 30 words |
| Large | 300 words | 50 words |

The comparison uses Recall@5.

Recall@5 measures whether the designated gold passage
appears among the five highest-ranked retrieved chunks.

## 5. Embedding Model

Model: sentence-transformers/all-MiniLM-L6-v2

The model converts document chunks and user questions
into dense vector representations.

Cosine similarity is used to rank semantically
similar document passages.

## 6. Hybrid Retrieval

The retrieval system combines two approaches.

### Dense Retrieval

Uses embedding vectors and cosine similarity to
retrieve semantically relevant passages.

### BM25

BM25 is a lexical retrieval algorithm based on
term frequency and inverse document frequency.

It ranks documents according to the relevance
of query terms appearing in each document.

BM25 helps retrieve passages containing important
technical terms that semantic retrieval may miss.

### Reciprocal Rank Fusion

Dense retrieval and BM25 rankings are combined
using Reciprocal Rank Fusion (RRF).

RRF assigns higher combined scores to passages
that appear near the top of multiple rankings.

## 7. Cross-Encoder Reranking

Model: cross-encoder/ms-marco-MiniLM-L-6-v2

The cross-encoder evaluates each question-passage
pair jointly and produces a relevance score.

Candidate passages are sorted by this score,
and the highest-ranked passages are passed
to the answer-generation stage.

## 8. Answer Generation

Model: google/flan-t5-large

The generator receives the question and retrieved
evidence passages.

It produces a concise answer using the retrieved
document context.

A grounding check compares generated answer terms
against the evidence. If the generated answer
does not satisfy the validation threshold,
the system falls back to extractive evidence.

## 9. Evaluation

The evaluation dataset contains 15 questions
with designated gold passages.

Metrics:

- Recall@5 for small chunks.
- Recall@5 for large chunks.
- Recall@6 for hybrid retrieval and reranking.

Recall@6 checks whether the gold passage appears
among the six final reranked chunks.

Retrieval accuracy and answer correctness are
evaluated separately.

## 10. Retrieval Failure Analysis

The initial hybrid retrieval experiment identified
two questions where the gold passage was not
retrieved within the top six results:

1. Purpose of the GOVERN function.
2. Main purpose of the MANAGE function.

Potential improvements include domain-specific
query expansion, broader candidate retrieval,
and reranking.

The updated implementation incorporates
NIST-specific query expansion.

Final local evaluation results are recorded
in artifacts/evaluation_summary.json.

## 11. Limitations

- Retrieval quality depends on the embedding model
  and the selected chunk configuration.
- Cross-encoder reranking increases inference time.
- FLAN-T5-large requires substantial memory.
- Generated answers can be incomplete or incorrect
  even when the correct passage is retrieved.
- The evaluation dataset contains 15 questions
  and does not represent every possible query.

## 12. Reproducibility

Install dependencies:

    pip install -r requirements.txt

Build the document index:

    python -m src.pipeline

Run evaluation:

    python evaluate.py

Evaluation outputs are stored in the artifacts directory.