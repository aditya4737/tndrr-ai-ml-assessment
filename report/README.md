
# tndrr AI/ML Internship – Technical Assignment Report

**Candidate:** Aditya Khavate  
**Assignment:** AI/ML Internship – Assignment Round  
**Date:** November 2026

---

# TASK 1 – Fine-Tuning with LoRA/QLoRA

## 1. Objective

The objective was to fine-tune a small open-weight language model using QLoRA for a structured information extraction task.

The selected task was converting natural-language requests into structured JSON output. The objective was to make the model consistently follow a predefined JSON format rather than generate unrestricted natural-language responses.

## 2. Model and Approach

**Base model:** Qwen2.5-0.5B-Instruct  
**Fine-tuning approach:** QLoRA  
**Task:** Natural-language to structured JSON conversion.

QLoRA was selected to reduce GPU memory requirements while adapting the model to a narrow task. Instead of updating all model parameters, the approach uses low-rank trainable adapters while keeping the quantized base model frozen.

The training and evaluation process involved:
1. Preparing instruction and expected JSON output pairs.
2. Tokenizing the training examples.
3. Loading the base model in quantized form.
4. Attaching LoRA adapters to selected model layers.
5. Fine-tuning the adapters.
6. Evaluating the base and fine-tuned models on held-out examples.

### LoRA configuration

The exact configuration should be copied from the training notebook before submission:

| Parameter | Configuration |
|---|---|
| LoRA rank (r) | [INSERT ACTUAL VALUE] |
| LoRA alpha | [INSERT ACTUAL VALUE] |
| Target modules | [INSERT ACTUAL MODULES] |
| Learning rate | [INSERT ACTUAL VALUE] |
| Epochs | [INSERT ACTUAL VALUE] |

The rank controls the capacity of the low-rank adapter. Alpha controls the scaling of the adapter updates. Target modules determine which model layers receive adapters. Learning rate and epochs control the optimization process.

These values must match the actual training run.

## 3. Evaluation Results

The model was evaluated on held-out examples and additional challenge sets.

| Evaluation set | Number of examples | Base model exact match | Fine-tuned exact match |
|---|---:|---:|---:|
| Held-out test set | 100 | 0% | 100% |
| Challenge set | 20 | — | 100% |
| Hard test set | 10 | — | 60% |

Exact match measures whether the generated structured output exactly matches the expected output.

The fine-tuned model achieved 100% exact match on the held-out test set and challenge set. Performance decreased to 60% on the harder examples, indicating that the model still struggled with more difficult or less familiar inputs.

The results show that fine-tuning improved performance on the selected structured-output task. However, the hard-set results indicate that the model's generalization is not perfect.

## 4. Failure Analysis

The hard evaluation set contained four unsuccessful examples out of ten.

The following three examples must be inserted from the actual evaluation logs:

**Failure 1**
- Input: [INSERT ACTUAL FAILED PROMPT]
- Expected output: [INSERT EXPECTED JSON]
- Actual output: [INSERT MODEL OUTPUT]
- Possible reason: [EXPLAIN THE ACTUAL FAILURE]

**Failure 2**
- Input: [INSERT ACTUAL FAILED PROMPT]
- Expected output: [INSERT EXPECTED JSON]
- Actual output: [INSERT MODEL OUTPUT]
- Possible reason: [EXPLAIN THE ACTUAL FAILURE]

**Failure 3**
- Input: [INSERT ACTUAL FAILED PROMPT]
- Expected output: [INSERT EXPECTED JSON]
- Actual output: [INSERT MODEL OUTPUT]
- Possible reason: [EXPLAIN THE ACTUAL FAILURE]

Possible causes to investigate include ambiguous instructions, unfamiliar combinations of fields, incorrect value extraction, and failure to follow the required JSON schema. These should only be attributed to a particular example when supported by its actual output.

## 5. Estimated GPU Memory for Fine-Tuning a 7B Model

The following are the estimates obtained in the experiment's memory calculation:

| Fine-tuning approach | Estimated GPU memory |
|---|---:|
| Full fine-tuning | 104.31 GiB |
| LoRA | 15.04 GiB |
| QLoRA | 6.26 GiB |

Full fine-tuning requires memory for model weights, gradients, optimizer states and activations. Since all model parameters are trainable, the memory requirement is high.

LoRA freezes the base model and trains only the low-rank adapter parameters. This reduces the memory required for gradients and optimizer states.

QLoRA additionally quantizes the frozen base model, reducing the memory required to store model weights.

These are approximate estimates, not universal requirements. Actual memory consumption depends on sequence length, batch size, optimizer, precision, gradient checkpointing and implementation.

---

# TASK 2 – Retrieval-Augmented Generation (RAG) from Scratch

## 1. Objective

The objective was to build a question-answering system over a public document without using high-level RAG frameworks such as LangChain or LlamaIndex.

The system retrieves relevant passages from a document and uses a language model to generate answers based on the retrieved information.

**Source document:** NIST AI Risk Management Framework (AI RMF 1.0)  
**Document length:** 48 pages

## 2. Technology Stack

- Python
- PyPDF
- sentence-transformers/all-MiniLM-L6-v2
- NumPy
- BM25 using rank_bm25
- CrossEncoder: cross-encoder/ms-marco-MiniLM-L-6-v2
- Reciprocal Rank Fusion (RRF)
- FLAN-T5-large
- PyTorch
- Hugging Face Transformers

The implementation was built without LangChain or LlamaIndex.

## 3. System Architecture

The pipeline consists of the following stages:

1. **Document ingestion:** Extract text and page information from the NIST PDF using PyPDF.
2. **Text chunking:** Split extracted text into smaller overlapping passages.
3. **Embedding generation:** Generate vector embeddings using all-MiniLM-L6-v2.
4. **Dense retrieval:** Compare query embeddings against document embeddings using NumPy similarity calculations.
5. **Keyword retrieval:** Use BM25 to retrieve passages based on lexical relevance.
6. **Rank fusion:** Combine dense and keyword rankings using Reciprocal Rank Fusion.
7. **Re-ranking:** Use a CrossEncoder to score query-passage pairs and improve passage ordering.
8. **Answer generation:** Use FLAN-T5-large to generate an answer from retrieved passages.

## 4. Chunking Experiments

Two chunk configurations were evaluated.

| Parameter | Small chunks | Large chunks |
|---|---:|---:|
| Chunk size | 150 words | 300 words |
| Overlap | 30 words | 50 words |
| Number of chunks | 154 | 87 |
| Embedding dimensions | 384 | 384 |

The small-chunk configuration produced more granular passages, while the large-chunk configuration retained more surrounding context.

## 5. Retrieval Evaluation

Fifteen questions were created, with expected source passages identified from the NIST document.

Dense retrieval results:

| Configuration | Correct retrievals | Recall@5 |
|---|---:|---:|
| 150-word chunks | 12/15 | 80% |
| 300-word chunks | 9/15 | 60% |

The smaller chunks performed better in this experiment. More granular passages helped retrieve specific information without including as much unrelated content.

### Hybrid retrieval and re-ranking

The hybrid retrieval pipeline combined dense retrieval, BM25, rank fusion and CrossEncoder re-ranking.

An earlier evaluation with 30 candidates and a final top-6 result achieved:

- Correct retrievals: 14/15
- Recall@6: 93.33%

This was an improvement over dense-only retrieval.

The retrieval pipeline was subsequently adjusted to retrieve a broader candidate set to address the remaining MAP-function retrieval miss. The final expanded retrieval run should be reported separately if its saved evaluation output confirms the 15/15 result. Recall@6 and Recall@5 are different metrics and should not be compared as if they were identical.

## 6. Test Questions and Expected Source Pages

The following questions were used for retrieval evaluation.

| No. | Question | Expected source page |
|---:|---|---:|
| 1 | How does NIST define an AI system? | 6 |
| 2 | Why can AI systems create risks that differ from traditional software systems? | 6 |
| 3 | What are the four core functions of the NIST AI RMF? | 8 |
| 4 | What does validity and reliability mean for an AI system? | 18 |
| 5 | What is the purpose of the GOVERN function? | 26 |
| 6 | What does the MAP function establish about an AI system? | 29 |
| 7 | What is the role of the MEASURE function? | 33 |
| 8 | What is the main purpose of the MANAGE function? | 36 |
| 9 | What should organizations consider when prioritizing AI risks? | 13 |
| 10 | What are AI RMF Profiles used for? | 38 |
| 11 | What characteristics does NIST identify as necessary for trustworthy AI systems? | 17 |
| 12 | Is the NIST AI RMF intended to be voluntary or mandatory? | 7 |
| 13 | Who are AI actors in the context of the AI RMF? | 7 |
| 14 | How can changes in training data affect an AI system? | 6 |
| 15 | Why should organizations consider risks associated with third-party software and data? | 10 |

The source page numbers refer to the document's printed page numbering.

## 7. Retrieval Failure Analysis

**Failure 1 – Dense retrieval with larger chunks**

The 300-word chunk configuration retrieved the expected source passage for only 9 of the 15 questions.

Larger chunks contained more surrounding text, which sometimes reduced the relevance of the retrieved passage to a specific question.

A smaller chunk size with overlap improved retrieval granularity and achieved 12/15 correct retrievals.

**Failure 2 – MAP function retrieval**

An earlier hybrid retrieval evaluation failed to retrieve the expected passage on page 29 for the question about the MAP function.

The retrieved passages included related content from other pages, but did not initially rank the exact definition highly enough.

The pipeline was adjusted to expand the retrieval candidate set before re-ranking. This allowed the relevant passage to be included in the candidate pool.

These examples demonstrate that retrieval quality depends on chunking, candidate selection and ranking, not only embedding similarity.

## 8. BM25 and Re-ranking

### What BM25 adds to embedding search

Embedding-based retrieval finds passages based on semantic similarity. However, it may fail to prioritize exact technical terms, acronyms or uncommon keywords.

BM25 is a lexical retrieval algorithm that uses term frequency and inverse document frequency to score documents based on query terms.

Combining BM25 with embeddings provides two complementary signals:
- Semantic similarity from embeddings.
- Exact keyword relevance from BM25.

This can improve retrieval when a question contains specific terminology.

### What does a re-ranker do?

A re-ranker evaluates the relevance of retrieved passages more carefully.

A CrossEncoder receives the query and a candidate passage together and predicts a relevance score. Unlike bi-encoder retrieval, it jointly processes the query and passage.

Because this process is computationally more expensive, it is usually applied to a smaller candidate set rather than the entire document collection.

---

# TASK 3 – Running a Model Locally and Optimisation

## 1. Objective

The objective was to run an open-weight language model locally using Ollama and compare its performance under two quantization levels.

The experiment used Qwen2.5-0.5B with Q4_K_M and Q8_0 quantization.

The comparison focused on generation speed, GPU memory usage and output quality.

## 2. Experimental Setup

**Hardware**
- GPU: NVIDIA GeForce RTX 3050
- GPU VRAM: 4 GB
- System RAM: 16 GB
- Operating system: Windows

**Software**
- Ollama 0.40.16
- Python
- Requests
- psutil
- NVIDIA SMI

**Model configurations**

| Configuration | Model | Quantization | Download size |
|---|---|---|---:|
| 1 | Qwen2.5-0.5B | Q4_K_M | 397 MB |
| 2 | Qwen2.5-0.5B-Instruct | Q8_0 | 531 MB |

Both models have approximately 494 million parameters and use the Qwen2 architecture.

Both configurations were verified to run with 100% GPU offloading using `ollama ps`.

## 3. Benchmark Methodology

Ten identical prompts were evaluated for each configuration.

The benchmark used:
- Temperature: 0
- Seed: 42
- Context length: 4096 tokens
- Maximum generation tokens: 150
- Streaming generation through the Ollama API

The benchmark recorded generation token counts, generation duration and tokens per second.

## 4. Performance Results

| Metric | Q4_K_M | Q8_0 |
|---|---:|---:|
| Number of prompts | 10 | 10 |
| Average speed | 60.77 tokens/sec | 55.58 tokens/sec |
| Median speed | 60.94 tokens/sec | 55.71 tokens/sec |
| Minimum speed | 59.34 tokens/sec | 53.78 tokens/sec |
| Maximum speed | 63.13 tokens/sec | 57.82 tokens/sec |
| Total generated tokens | 1,429 | 1,402 |
| Model download size | 397 MB | 531 MB |

Q4_K_M achieved approximately 9.35% higher average generation speed than Q8_0.

## 5. GPU Memory Usage

GPU memory was measured using NVIDIA SMI before and after loading each model.

The idle GPU memory consumption was 1,533 MiB.

| Measurement | Q4_K_M | Q8_0 |
|---|---:|---:|
| Idle GPU memory | 1,533 MiB | 1,533 MiB |
| GPU memory after loading | 2,072 MiB | 2,200 MiB |
| Additional GPU memory | 539 MiB | 667 MiB |
| GPU offloading | 100% | 100% |
| Ollama runtime size | 483 MB | 616 MB |

Q8_0 used approximately 23.7% more additional GPU memory than Q4_K_M.

Additional GPU memory was calculated by subtracting the idle baseline from the loaded measurement.

These measurements include runtime overhead and other GPU allocations. They should not be interpreted as exact model-weight memory.

The benchmark's system RAM measurements represent total Windows memory usage, not isolated Ollama process memory, so they are not presented as model-specific RAM requirements.

## 6. Output Quality Comparison

The same ten prompts were used to compare the responses from both models.

Both configurations generated generally understandable answers to basic machine learning and artificial intelligence questions.

However, the responses showed differences in factual accuracy and completeness.

For example, when asked to explain transformers, the Q8_0 model incorrectly described transformers as a type of recurrent neural network (RNN). Transformers use self-attention and are not recurrent neural networks.

The Q4_K_M response was also incomplete, but did not make that particular RNN classification error.

This small qualitative evaluation did not establish a consistent output-quality advantage for Q8_0. A larger evaluation dataset and a formal scoring rubric would be required to establish a reliable difference.

## 7. What is KV Cache?

KV cache stands for Key-Value cache.

Transformer models use attention mechanisms that calculate key and value representations for input tokens.

When generating a new token, the model needs information from previously processed tokens. Without caching, it would repeatedly calculate the keys and values for earlier tokens.

KV cache stores these previously calculated representations so they can be reused during subsequent generation steps.

This reduces repeated computation and improves inference efficiency.

However, KV cache consumes GPU memory. Its memory requirement increases with:
- Number of transformer layers.
- Number of key-value heads.
- Head dimension.
- Context length.
- Number of concurrent sequences.

This is why long-context requests and high concurrency can significantly increase GPU memory requirements.

## 8. Why Does Quantization Save Memory?

Quantization reduces the number of bits used to represent model weights.

For example:

| Data type | Bits per weight |
|---|---:|
| FP16 | 16 bits |
| INT8 | 8 bits |
| 4-bit | 4 bits |

A model with fewer bits per weight requires less memory to store and load its parameters.

For the same number of parameters, 4-bit quantization requires substantially less weight storage than FP16.

The trade-off is that quantization introduces numerical approximation, which can affect model quality.

In this experiment, Q4_K_M required less additional GPU memory and achieved higher generation speed than Q8_0. Neither configuration demonstrated a consistent quality advantage in the limited qualitative evaluation.

## 9. Serving 20 Concurrent Users on a 24 GB GPU

For a hypothetical deployment, I would use a 7B parameter model with 4-bit quantization.

### Assumptions

- GPU: 24 GB VRAM.
- Model: 7B parameters, 4-bit quantized.
- Estimated model weight memory: 4.5 GiB.
- Concurrent sequences: 20.
- Maximum context length: 2,048 tokens per sequence.
- KV cache precision: FP16.
- Serving engine: vLLM or another engine supporting continuous batching and efficient KV-cache management.

### KV cache calculation

For an illustrative grouped-query-attention architecture with:
- 28 transformer layers.
- 4 KV heads.
- Head dimension of 128.
- FP16 key and value representations (2 bytes each).

KV cache memory per token:

2 × number of layers × KV heads × head dimension × bytes per value

= 2 × 28 × 4 × 128 × 2

= 57,344 bytes

= 56 KiB per token.

For one sequence of 2,048 tokens:

56 KiB × 2,048 = 112 MiB.

For 20 concurrent sequences:

112 MiB × 20 = 2,240 MiB.

Therefore, the estimated KV cache requirement is approximately 2.19 GiB.

### Estimated GPU memory budget

| Component | Estimated memory |
|---|---:|
| 4-bit model weights | 4.5 GiB |
| KV cache for 20 sequences | 2.19 GiB |
| Runtime, activations and temporary buffers | 3–5 GiB |
| Safety reserve | 2 GiB |
| Estimated total | 11.69–13.69 GiB |

The estimate fits within a 24 GB GPU under the stated assumptions.

### Deployment approach

I would use the following approach:

1. Load a single quantized model instance on the GPU.
2. Use continuous batching to process multiple requests together.
3. Use paged KV-cache allocation to reduce memory fragmentation.
4. Limit each request to a maximum context of 2,048 tokens.
5. Allow up to 20 active sequences.
6. Queue excess requests rather than allowing uncontrolled GPU memory allocation.
7. Monitor GPU memory, token throughput and request latency.

Although the estimated memory budget fits, this does not guarantee that all 20 users will receive low latency. Actual throughput depends on GPU compute capacity, prompt length, generated output length and batching efficiency.

This is a theoretical capacity estimate, not a deployment benchmark performed on my 4 GB laptop GPU.

## 10. Conclusion

The experiment demonstrated the practical impact of quantization on local language model inference.

Q4_K_M achieved an average generation speed of 60.77 tokens/sec, compared with 55.58 tokens/sec for Q8_0. It also required 539 MiB of additional GPU memory compared with 667 MiB for Q8_0.

Both models ran entirely on the GPU. The limited qualitative comparison did not establish a consistent output-quality advantage for Q8_0.

For this particular model and hardware configuration, Q4_K_M provided the better observed speed and memory trade-off.

---

# Overall Conclusion

Across the three tasks, I explored parameter-efficient fine-tuning, retrieval-augmented generation and local model inference optimisation.

In Task 1, I used QLoRA to adapt Qwen2.5-0.5B-Instruct for structured JSON generation and evaluated its performance on held-out and challenge examples.

In Task 2, I implemented a RAG pipeline from scratch using embeddings, NumPy, BM25, Reciprocal Rank Fusion and CrossEncoder re-ranking. The smaller chunk configuration achieved 80% Recall@5 compared with 60% for the larger chunk configuration.

In Task 3, I compared Q4_K_M and Q8_0 quantization on Qwen2.5-0.5B. Q4_K_M achieved higher generation speed and lower additional GPU memory usage in the measured experiment.

These tasks provided practical experience with fine-tuning, retrieval pipelines, model evaluation, quantization and inference memory management.
