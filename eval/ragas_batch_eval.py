"""
ragas_batch_eval.py
===================
Standalone batch RAGAS evaluator for the RAG pipeline.

Strategy
--------
Reuses the SAME pipeline code that the WebSocket uses:
  retrieval_agent → synthesis_agent → factcheck_agent

Only the LAST step differs: instead of the existing background scorer
(which only runs faithfulness + answer_relevancy without ground truth),
this script runs ALL 4 RAGAS metrics against the ground truth CSV:
  • faithfulness
  • answer_relevancy
  • context_precision
  • context_recall

Rate-limit guard
----------------
A 2-minute sleep is applied after EVERY LLM call (query rewrite,
synthesis, factcheck, RAGAS scoring) to stay within Groq's free-tier
limits. Each question therefore takes ~10-12 minutes to process, so
the full 23-question suite runs in ~4-5 hours.

Usage (from backend/ directory)
--------------------------------
    python -m eval.ragas_batch_eval
    # or
    python eval/ragas_batch_eval.py

Output
------
  eval/results/ragas_batch_<timestamp>.csv
  Console progress + final summary table
"""

import asyncio
import csv
import math
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force UTF-8 encoding for stdout/stderr to avoid Windows charmap encoding errors
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── ensure the backend root is on sys.path when run directly ──────────────────
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ── load .env before any module that needs env vars ──────────────────────────
from dotenv import load_dotenv
load_dotenv(_BACKEND_ROOT / ".env")

# ── suppress LangSmith tracing during eval (same as existing scorer) ─────────
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"]     = "false"

# Force evaluation pipeline to use Gemini to avoid Groq's low daily token limits (TPD)
os.environ["PIPELINE_LLM_PROVIDER"] = "gemini"

# ── project imports ───────────────────────────────────────────────────────────
from agents.graph.graph   import build_graph
from agents.models.pipeline import PipelineState
from utils.eval           import get_state_val
from rag_pipeline.chunking import get_embedding_model, chunk_semantic
from rag_pipeline.llm     import get_eval_llm
from qdrant_client        import QdrantClient
from langchain_qdrant     import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

# Mirror the constants defined in rag_pipeline/qdrant.py
COLLECTION_NAME = "agentbench_docs_bge"
VECTOR_SIZE     = 384  # BAAI/bge-small-en-v1.5

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from typing import cast
from ragas.evaluation import EvaluationResult

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH_CSV = _BACKEND_ROOT / "files" / "ragas_ground_truth.csv"
PDF_DIR          = _BACKEND_ROOT / "files"
PDF_FILES        = [
    "rag_fundamentals.pdf",
    "rag_retrieval_optimization.pdf",
    "rag_evaluation.pdf",
]

# Eval session ID — a proper UUID4, consistent with how real sessions are created.
# Only used as a Qdrant payload filter key (metadata.session_id).
# Supabase is NOT called by this script — scoring goes straight to CSV.
EVAL_SESSION_ID = str(uuid.uuid4())
EVAL_USER_ID    = "eval-script"

# Rate-limit: sleep this many seconds after EVERY LLM query.
# The pipeline itself makes 3 LLM calls per question (rewrite, synthesis,
# factcheck). RAGAS adds ~4-5 more calls. With a 120s sleep only between
# full question cycles we hit the rate limit. So we sleep BETWEEN questions.
SLEEP_BETWEEN_QUESTIONS_SEC = 120  # 2 minutes

OUTPUT_DIR = _BACKEND_ROOT / "eval" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth() -> list[dict]:
    """Load the ground truth CSV. Returns list of row dicts."""
    rows = []
    with open(GROUND_TRUTH_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            if q:                           # skip blank trailing rows
                rows.append(row)
    print(f"[loader] Loaded {len(rows)} ground truth rows from {GROUND_TRUTH_CSV.name}")
    return rows


def build_vectorstore() -> QdrantVectorStore:
    """Connect to the same Qdrant collection the live server uses."""
    client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        api_key=os.getenv("QDRANT_API"),
        timeout=60,
    )

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    # Ensure payload indexes exist (idempotent — silently skips if already created)
    for field in ("metadata.session_id", "metadata.file_id"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    embeddings = get_embedding_model()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def index_pdfs(vectorstore: QdrantVectorStore) -> None:
    """
    Parse + chunk + index the 3 evaluation PDFs into the shared Qdrant
    collection under the EVAL_SESSION_ID so the retrieval agent can find them.
    """
    from utils.file_parser import extract_text

    for pdf_name in PDF_FILES:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"[index] WARNING: {pdf_path} not found — skipping.")
            continue

        raw_bytes = pdf_path.read_bytes()
        text = extract_text(pdf_name, raw_bytes)
        chunks = chunk_semantic(text)

        file_id = str(uuid.uuid4())
        for chunk in chunks:
            chunk.metadata.update({
                "chunk_id":   str(uuid.uuid4()),
                "file_id":    file_id,
                "user_id":    EVAL_USER_ID,
                "session_id": EVAL_SESSION_ID,
                "filename":   pdf_name,
            })

        if chunks:
            vectorstore.add_documents(chunks)
            print(f"[index] Indexed {len(chunks)} chunks from {pdf_name}")
        else:
            print(f"[index] No chunks produced from {pdf_name}")


def cleanup_eval_vectors(vectorstore: QdrantVectorStore) -> None:
    """Remove all vectors tagged with our eval session_id after the run."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        vectorstore.client.delete(
            collection_name=vectorstore.collection_name,
            points_selector=Filter(
                must=[FieldCondition(
                    key="metadata.session_id",
                    match=MatchValue(value=EVAL_SESSION_ID),
                )]
            ),
        )
        print(f"[cleanup] Removed eval vectors for session {EVAL_SESSION_ID}")
    except Exception as e:
        print(f"[cleanup] Warning — could not delete eval vectors: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    graph,
    question: str,
) -> tuple[str, list[str]]:
    """
    Run the full RAG graph (retrieval → synthesis → factcheck) for one question.
    Returns (final_answer, context_texts).

    This is exactly what ChatSessionService does internally — only without the
    WebSocket streaming layer. We call ainvoke() instead of astream_events()
    because we don't need token-by-token streaming here.
    """
    initial_state = PipelineState(
        original_query=question,
        chat_history="",          # no chat history in batch eval
        session_id=EVAL_SESSION_ID,
        rewritten_queries=[],
        retrieved_chunks=[],
        synthesized_answer="",
        citations=[],
        verified_claims=[],
        confidence_score=0.0,
        final_answer="",
        retry_count=0,
    )

    result = await graph.ainvoke(initial_state)

    # Extract final answer
    final_answer: str = get_state_val(result, "final_answer") or get_state_val(result, "synthesized_answer") or ""

    # Extract retrieved context texts (same extraction as _trigger_background_eval)
    raw_chunks = get_state_val(result, "retrieved_chunks", [])
    contexts: list[str] = []
    for chunk in raw_chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        else:
            text = getattr(chunk, "text", "")
        if text:
            contexts.append(text)

    return final_answer, contexts


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS scoring
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(val) -> Optional[float]:
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
    except (TypeError, ValueError):
        pass
    return float(val)


def score_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    eval_llm,
    embeddings,
) -> dict:
    """
    Run all 4 RAGAS metrics for a single Q&A pair.
    Runs in a thread (blocking) — wraps inside asyncio.to_thread at call site.

    Metrics:
      faithfulness      — are answer claims supported by retrieved context?
      answer_relevancy  — is the answer relevant to the question?
      context_precision — what fraction of retrieved chunks are relevant?
      context_recall    — what fraction of ground truth is covered by context?
    """
    dataset = Dataset.from_dict({
        "question":     [question],
        "answer":       [answer],
        "contexts":     [contexts],
        "ground_truth": [ground_truth],
    })

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=1, timeout=600)

    result = cast(EvaluationResult, evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=embeddings,
        run_config=run_config,
    ))

    scores = {
        "faithfulness":       _sanitize(result["faithfulness"][0]),
        "answer_relevancy":   _sanitize(result["answer_relevancy"][0]),
        "context_precision":  _sanitize(result["context_precision"][0]),
        "context_recall":     _sanitize(result["context_recall"][0]),
    }

    if any(v is None for v in scores.values()):
        print("  [ragas] Warning: One or more scores are None. Testing LLM for rate limits...")
        try:
            # dummy call to trigger rate limit error if any
            underlying = getattr(eval_llm, "langchain_llm", eval_llm)
            underlying.invoke("Hello")
        except Exception as e:
            # Raise the actual rate limit error so the caller can parse it
            print(f"  [ragas] LLM test call failed: {e}")
            raise e
        # If no rate limit error was raised, raise a generic error to retry
        raise ValueError("Ragas evaluation returned incomplete scores (None) without raising LLM error.")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], timestamp: str) -> Path:
    output_path = OUTPUT_DIR / f"ragas_batch_{timestamp}.csv"
    latest_path = OUTPUT_DIR / "ragas_batch_latest.csv"
    if not results:
        print("[output] No results to save.")
        return output_path

    fieldnames = list(results[0].keys())
    for path in [output_path, latest_path]:
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        except Exception as e:
            print(f"[output] Error writing to {path.name}: {e}")

    print(f"\n[output] Results saved → {output_path} (and latest.csv)")
    return output_path


def find_latest_results_file() -> Optional[Path]:
    latest_path = OUTPUT_DIR / "ragas_batch_latest.csv"
    if latest_path.exists():
        return latest_path
        
    csv_files = list(OUTPUT_DIR.glob("ragas_batch_*.csv"))
    if not csv_files:
        return None
    csv_files.sort(key=lambda p: p.name)
    return csv_files[-1]


def load_existing_results(file_path: Path) -> dict[str, dict]:
    """
    Load previously completed rows from CSV.
    Only rows with no errors and valid answers/scores are considered complete.
    """
    completed = {}
    try:
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("question", "").strip()
                if not q:
                    continue
                p_err = row.get("pipeline_error")
                r_err = row.get("ragas_error")
                ans = row.get("answer", "").strip()
                
                has_pipeline_error = p_err and p_err.strip() and p_err.strip().lower() != "none"
                has_ragas_error = r_err and r_err.strip() and r_err.strip().lower() != "none"
                
                # Check if we have at least one score
                has_score = any(
                    row.get(m) is not None and row.get(m).strip() != "" and row.get(m).strip().lower() != "none"
                    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                )
                
                if not has_pipeline_error and not has_ragas_error and ans and has_score:
                    row_data = dict(row)
                    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                        val = row_data.get(m)
                        if val is not None and val.strip() != "" and val.strip().lower() != "none":
                            try:
                                row_data[m] = float(val)
                            except ValueError:
                                row_data[m] = None
                        else:
                            row_data[m] = None
                    for col in ["context_count", "latency_pipeline_s", "latency_ragas_s"]:
                        val = row_data.get(col)
                        if val is not None and val.strip() != "":
                            try:
                                if "." in val:
                                    row_data[col] = float(val)
                                else:
                                    row_data[col] = int(val)
                            except ValueError:
                                pass
                    completed[q] = row_data
    except Exception as e:
        print(f"[resume] Warning — failed to read existing results file: {e}")
    return completed


def parse_rate_limit_time(error_str: str) -> float:
    """
    Parse rate limit wait time from error message.
    Returns 0 if it doesn't look like a rate limit error.
    Otherwise returns the wait time in seconds (minimum 300 if no number found,
    since user said 'pause for 5 mins').
    """
    import re
    err_lower = error_str.lower()
    is_rate_limit = any(x in err_lower for x in [
        "rate limit", "rate_limit", "429", "too many requests", 
        "limit reached", "retry-after", "retry after", "try again in"
    ])
    if not is_rate_limit:
        return 0.0

    match_hms = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", err_lower)
    if match_hms and (match_hms.group(1) or match_hms.group(2) or match_hms.group(3)):
        h = float(match_hms.group(1)) if match_hms.group(1) else 0.0
        m = float(match_hms.group(2)) if match_hms.group(2) else 0.0
        s = float(match_hms.group(3)) if match_hms.group(3) else 0.0
        total = h * 3600 + m * 60 + s
        if total > 0:
            return total

    match_sec = re.search(r"(?:retry after|retry-after|try again in)\s*(\d+(?:\.\d+)?)", err_lower)
    if match_sec:
        return float(match_sec.group(1))

    return 300.0


def print_summary(results: list[dict], elapsed_seconds: float) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 60)
    print("       RAGAS BATCH EVALUATION SUMMARY")
    print("=" * 60)

    for metric in metrics:
        vals = [r[metric] for r in results if r.get(metric) is not None]
        if vals:
            avg = sum(vals) / len(vals)
            lo  = min(vals)
            hi  = max(vals)
            print(f"  {metric:<22}  avg={avg:.4f}  min={lo:.4f}  max={hi:.4f}  (n={len(vals)})")
        else:
            print(f"  {metric:<22}  no valid scores")

    query_types = set(r.get("query_type", "") for r in results if r.get("query_type"))
    if query_types:
        print("\n  -- By query type --")
        for qt in sorted(query_types):
            subset = [r for r in results if r.get("query_type") == qt]
            for metric in ["faithfulness", "answer_relevancy"]:
                vals = [r[metric] for r in subset if r.get(metric) is not None]
                if vals:
                    avg = sum(vals) / len(vals)
                    print(f"  [{qt:>8}] {metric:<22} avg={avg:.4f}  (n={len(vals)})")

    h, rem = divmod(int(elapsed_seconds), 3600)
    m, s   = divmod(rem, 60)
    print(f"\n  Total questions : {len(results)}")
    print(f"  Total duration  : {h}h {m}m {s}s")
    print("=" * 60)


async def main() -> None:
    global SLEEP_BETWEEN_QUESTIONS_SEC
    wall_start = time.monotonic()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  RAGAS BATCH EVALUATOR")
    print(f"  Session : {EVAL_SESSION_ID}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Build Qdrant vectorstore ──────────────────────────────────────────
    print("\n[setup] Connecting to Qdrant…")
    vectorstore = build_vectorstore()

    # ── 2. Index the 3 evaluation PDFs ──────────────────────────────────────
    print("[setup] Indexing evaluation PDFs…")
    index_pdfs(vectorstore)

    # ── 3. Build the pipeline graph (same factory the live server uses) ──────
    graph = build_graph(vectorstore)

    # ── 4. Load eval LLM + embeddings for RAGAS scoring ─────────────────────
    eval_llm   = get_eval_llm()
    embeddings = get_embedding_model()

    # ── 5. Load ground truth ─────────────────────────────────────────────────
    rows = load_ground_truth()

    # ── 6. Check for Resume ──────────────────────────────────────────────────
    resumed_by_question = {}
    latest_file = find_latest_results_file()
    if latest_file:
        print(f"[resume] Found existing results file: {latest_file.name}. Checking for completed questions...")
        resumed_by_question = load_existing_results(latest_file)
        if resumed_by_question:
            print(f"[resume] Loaded {len(resumed_by_question)} already completed questions.")

    # ── 7. Main evaluation loop ──────────────────────────────────────────────
    results: list[dict] = []

    for idx, row in enumerate(rows):
        question     = row["question"]
        ground_truth = row.get("ground_truth", "")
        query_type   = row.get("query_type", "")
        source_doc   = row.get("source_doc", "")

        print(f"\n{'-'*60}")
        print(f"[{idx+1}/{len(rows)}] {query_type.upper() or 'QUERY'}")
        print(f"  Q: {question[:100]}{'…' if len(question) > 100 else ''}")

        # If already completed in resumed results, copy and skip
        if question in resumed_by_question:
            print("  [resume] Found completed result in previous file. Skipping and restoring scores.")
            results.append(resumed_by_question[question])
            save_results(results, timestamp)
            continue

        # ── A. Run the full RAG pipeline with robust retry ───────────────────
        q_start = time.monotonic()
        answer = ""
        contexts = []
        pipeline_error = None
        pipeline_latency = 0.0

        max_pipeline_attempts = 10
        pipeline_attempt = 0
        while pipeline_attempt < max_pipeline_attempts:
            pipeline_attempt += 1
            try:
                p_run_start = time.monotonic()
                answer, contexts = await run_pipeline(graph, question)
                pipeline_latency = time.monotonic() - p_run_start
                pipeline_error = None
                break
            except Exception as e:
                pipeline_error = str(e)
                print(f"  [pipeline] Attempt {pipeline_attempt} failed: {e}")
                wait_time = parse_rate_limit_time(pipeline_error)
                if wait_time > 0:
                    print(f"  [pipeline] Rate limit detected. Pausing for {wait_time}s...")
                    SLEEP_BETWEEN_QUESTIONS_SEC = max(SLEEP_BETWEEN_QUESTIONS_SEC, int(wait_time) + 30)
                    print(f"  [pipeline] Increased base sleep between questions to {SLEEP_BETWEEN_QUESTIONS_SEC}s")
                    await asyncio.sleep(wait_time)
                else:
                    sleep_time = min(300, 5 * (2 ** pipeline_attempt))
                    print(f"  [pipeline] Generic error. Sleeping {sleep_time}s before retry...")
                    await asyncio.sleep(sleep_time)

        if pipeline_error:
            print(f"  [pipeline] FAILED all {max_pipeline_attempts} attempts. Moving on.")
            results.append({
                "question":          question,
                "ground_truth":      ground_truth,
                "query_type":        query_type,
                "source_doc":        source_doc,
                "answer":            "",
                "contexts":          "",
                "context_count":     0,
                "faithfulness":      None,
                "answer_relevancy":  None,
                "context_precision": None,
                "context_recall":    None,
                "pipeline_error":    pipeline_error,
                "ragas_error":       None,
                "latency_pipeline_s": None,
                "latency_ragas_s":   None,
            })
            save_results(results, timestamp)
            if idx < len(rows) - 1:
                _sleep_with_countdown(SLEEP_BETWEEN_QUESTIONS_SEC)
            continue

        print(f"  ✓ Pipeline done in {pipeline_latency:.1f}s | contexts={len(contexts)} | answer_len={len(answer)}")

        # ── B. RAGAS scoring with robust retry (blocking, run in thread) ────
        ragas_start = time.monotonic()
        ragas_error = None
        scores = {
            "faithfulness":       None,
            "answer_relevancy":   None,
            "context_precision":  None,
            "context_recall":     None,
        }

        if not contexts:
            print("  [ragas] WARNING: No contexts retrieved — skipping RAGAS scoring for this question.")
            ragas_error = "no_contexts"
        elif not answer:
            print("  [ragas] WARNING: Empty answer — skipping RAGAS scoring.")
            ragas_error = "no_answer"
        else:
            max_ragas_attempts = 10
            ragas_attempt = 0
            while ragas_attempt < max_ragas_attempts:
                ragas_attempt += 1
                try:
                    scores = await asyncio.to_thread(
                        score_with_ragas,
                        question,
                        answer,
                        contexts,
                        ground_truth,
                        eval_llm,
                        embeddings,
                    )
                    ragas_error = None
                    def fmt(val):
                        return f"{val:.3f}" if val is not None else "None"
                    print(f"  📊 faithfulness={fmt(scores['faithfulness'])}  "
                          f"relevancy={fmt(scores['answer_relevancy'])}  "
                          f"ctx_precision={fmt(scores['context_precision'])}  "
                          f"ctx_recall={fmt(scores['context_recall'])}")
                    break
                except Exception as e:
                    ragas_error = str(e)
                    print(f"  [ragas] Attempt {ragas_attempt} failed: {e}")
                    wait_time = parse_rate_limit_time(ragas_error)
                    if wait_time > 0:
                        print(f"  [ragas] Rate limit detected. Pausing for {wait_time}s...")
                        SLEEP_BETWEEN_QUESTIONS_SEC = max(SLEEP_BETWEEN_QUESTIONS_SEC, int(wait_time) + 30)
                        print(f"  [ragas] Increased base sleep between questions to {SLEEP_BETWEEN_QUESTIONS_SEC}s")
                        await asyncio.sleep(wait_time)
                    else:
                        sleep_time = min(300, 5 * (2 ** ragas_attempt))
                        print(f"  [ragas] Generic error. Sleeping {sleep_time}s before retry...")
                        await asyncio.sleep(sleep_time)

        ragas_latency = time.monotonic() - ragas_start

        results.append({
            "question":           question,
            "ground_truth":       ground_truth,
            "query_type":         query_type,
            "source_doc":         source_doc,
            "answer":             answer,
            "contexts":           " ||| ".join(contexts),
            "context_count":      len(contexts),
            "faithfulness":       scores["faithfulness"],
            "answer_relevancy":   scores["answer_relevancy"],
            "context_precision":  scores["context_precision"],
            "context_recall":     scores["context_recall"],
            "pipeline_error":     None,
            "ragas_error":        ragas_error,
            "latency_pipeline_s": round(pipeline_latency, 2),
            "latency_ragas_s":    round(ragas_latency, 2),
        })

        # ── C. Save incrementally ────────────────────────────────────────────
        save_results(results, timestamp)

        # ── D. Rate-limit sleep (skip after last question) ────────────────────
        if idx < len(rows) - 1:
            _sleep_with_countdown(SLEEP_BETWEEN_QUESTIONS_SEC)

    # ── 8. Final output ───────────────────────────────────────────────────────
    elapsed = time.monotonic() - wall_start
    save_results(results, timestamp)
    print_summary(results, elapsed)

    # ── 9. Clean up eval vectors from Qdrant ──────────────────────────────────
    print("\n[cleanup] Removing evaluation vectors from Qdrant…")
    cleanup_eval_vectors(vectorstore)


def _sleep_with_countdown(seconds: int, label: str = "rate-limit cooldown") -> None:
    """Sleep with a live countdown so you can see progress."""
    print(f"  ⏳ {label} — sleeping {seconds}s", end="", flush=True)
    step = max(1, seconds // 10)
    remaining = seconds
    while remaining > 0:
        chunk = min(step, remaining)
        time.sleep(chunk)
        remaining -= chunk
        if remaining > 0:
            print(f" …{remaining}s", end="", flush=True)
    print(" ✓")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
