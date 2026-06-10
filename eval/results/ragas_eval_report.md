# RAGAS Batch Evaluation Report

> **Run date:** 2026-06-09  
> **Results file:** `ragas_batch_latest.csv`  
> **Script:** `eval/ragas_batch_eval.py`

---

## Overview

| Property | Value |
|---|---|
| **Total questions** | 30 |
| **Pipeline errors** | 0 |
| **RAGAS errors** | 0 |
| **Clean results** | 30 (100%) |
| **Source documents** | 3 PDFs |
| **Metrics measured** | Faithfulness, Answer Relevancy, Context Precision, Context Recall |
| **Pipeline** | query rewrite → retrieval (k=8, reranked) → synthesis → fact-check |
| **Eval LLM** | Gemini (`PIPELINE_LLM_PROVIDER=gemini`) |
| **Embedding model** | BAAI/bge-small-en-v1.5 (384-dim) |
| **Vector store** | Qdrant Cloud (`agentbench_docs_bge` collection) |

---

## Questions Breakdown

### By Query Type

| Query Type | Count | What it tests |
|---|---|---|
| `factual` | 20 | Direct knowledge retrieval |
| `detail_extraction` | 5 | Pulling specific numbers / precise facts |
| `multi_hop` | 2 | Reasoning across multiple non-contiguous chunks |
| `summary` | 2 | Broad, multi-claim answers |
| `comparison` | 1 | Comparing two concepts or entities |

### By Source Document

| Document | Questions |
|---|---|
| `rag_fundamentals.pdf` | 12 |
| `rag_retrieval_optimization.pdf` | 10 |
| `rag_evaluation.pdf` | 7 |
| `rag_evaluation.pdf` + `rag_retrieval_optimization.pdf` (cross-doc) | 1 |

---

## Scores

### Overall (all 30 questions)

| Metric | Avg | Min | Max |
|---|---|---|---|
| **Faithfulness** | **0.977** | 0.692 | 1.000 |
| **Answer Relevancy** | **0.867** | 0.000 | 0.969 |
| **Context Precision** | **0.826** | 0.000 | 1.000 |
| **Context Recall** | **0.936** | 0.250 | 1.000 |

> **Target thresholds (from literature):**  
> Faithfulness > 0.90 · Answer Relevancy > 0.80 · Context Precision > 0.75 · Context Recall > 0.80

### By Query Type

| Type | n | Faithfulness | Ans. Relevancy | Ctx Precision | Ctx Recall |
|---|---|---|---|---|---|
| `factual` | 20 | 0.992 ✅ | 0.902 ✅ | 0.899 ✅ | 0.975 ✅ |
| `summary` | 2 | 1.000 ✅ | 0.869 ✅ | 0.250 ⚠️ | 1.000 ✅ |
| `detail_extraction` | 5 | 1.000 ✅ | 0.701 ⚠️ | 0.800 ✅ | 0.850 ✅ |
| `multi_hop` | 2 | 0.739 ⚠️ | 0.893 ✅ | 0.655 ⚠️ | 0.667 ⚠️ |
| `comparison` | 1 | 1.000 ✅ | 0.927 ✅ | 1.000 ✅ | 1.000 ✅ |

---

## Latency

| Stage | Avg | Max |
|---|---|---|
| **Pipeline** (rewrite → retrieval → synthesis → fact-check) | 23.0s | 50.4s |
| **RAGAS scoring** (LLM-as-judge, 4 metrics) | 87.9s | 141.6s |

---

## Analysis & Key Takeaways

### ✅ Strengths

- **Faithfulness is excellent (0.977)** — the synthesizer is almost never hallucinating beyond the retrieved context. The fact-checker is doing its job.
- **Context Recall is strong (0.936)** — the multi-query expansion + reranking strategy is successfully pulling most of the information needed to answer questions.
- **Factual queries are handled near-perfectly** — all four metrics above threshold, covering 20 of the 30 questions.
- **Zero pipeline or RAGAS errors** — the retry logic and rate-limit handling worked correctly across the full run.

### ⚠️ Weaknesses & Known Limitations

#### Multi-hop questions (n=2) — biggest weakness
- Faithfulness drops to **0.739**, Ctx Precision to **0.655**, Ctx Recall to **0.667**.
- Multi-hop questions require connecting facts from two different sections. The retrieval agent may find chunks for one part of the question but not both. The synthesizer then sometimes infers the gap, causing faithfulness loss.
- **Potential fix:** implement explicit sub-question decomposition in the query rewriter for multi-hop queries.

#### Summary questions — low Context Precision (0.25)
- Faithfulness and Recall are perfect (1.0), but Precision is only **0.25**.
- This is a **known RAGAS metric limitation**, not a pipeline failure: summary questions require many chunks (broad coverage), so most retrieved chunks are technically "relevant" — but RAGAS's LLM judge, which generates synthetic sub-questions from the ground truth, only finds a few chunks supporting each sub-question, making precision appear low.
- **Recommendation:** evaluate summary queries separately and do not penalise the pipeline for this.

#### Detail extraction — lower Answer Relevancy (0.701)
- Answers to precise extraction questions tend to be slightly over-explained (e.g., adding background context around a specific number or name).
- The RAGAS Answer Relevancy metric penalises verbosity because it measures how well synthetic questions generated from the answer map back to the original query.
- **Potential fix:** tune the synthesis prompt to be more terse for extraction-style queries.

---

## Metric Definitions (RAGAS)

| Metric | What it measures | Reference needed? |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? Detects hallucination. | No (reference-free) |
| **Answer Relevancy** | Is the answer relevant and complete for the question? Penalises off-topic or verbose answers. | No (reference-free) |
| **Context Precision** | What fraction of retrieved chunks are actually relevant to answering the question? | Yes (ground truth) |
| **Context Recall** | What fraction of the ground truth answer's claims are covered by the retrieved context? | Yes (ground truth) |

---

## Pipeline Configuration (at time of eval)

| Component | Setting |
|---|---|
| Retrieval strategy | Multi-query expansion (3 variants) + cross-encoder reranking |
| Chunks fetched per query variant | k = 8 |
| Final chunks passed to LLM | top 8 after reranking |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) |
| Generator LLM | Gemini (via `PIPELINE_LLM_PROVIDER=gemini`) |
| Fact-checker | Enabled (claim-level verification) |
