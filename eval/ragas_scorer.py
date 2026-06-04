import asyncio
from database import insert_one
from database import logger
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from rag_pipeline.llm import get_eval_llm
from typing import cast
from ragas.evaluation import EvaluationResult
from rag_pipeline.chunking import get_embedding_model

eval_llm=get_eval_llm()
embeddings=get_embedding_model()
eval_lock = asyncio.Lock()

async def score_response(question: str, answer: str, contexts: list[str]) -> dict:
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    })
    
    async with eval_lock:
        import os
        old_tracing = os.environ.get("LANGCHAIN_TRACING_V2")
        old_smith = os.environ.get("LANGSMITH_TRACING")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        try:
            result = cast(EvaluationResult, await asyncio.to_thread(
                evaluate,
                dataset, 
                metrics=[faithfulness, answer_relevancy], 
                llm=eval_llm, 
                embeddings=embeddings
            ))
        finally:
            if old_tracing is not None:
                os.environ["LANGCHAIN_TRACING_V2"] = old_tracing
            else:
                os.environ.pop("LANGCHAIN_TRACING_V2", None)
                
            if old_smith is not None:
                os.environ["LANGSMITH_TRACING"] = old_smith
            else:
                os.environ.pop("LANGSMITH_TRACING", None)

    return {
        "faithfulness": result["faithfulness"][0],
        "answer_relevancy": result["answer_relevancy"][0]
    }
async def evaluate_and_save_to_db(
    session_id: str,
    user_id: str,
    message_id: str,
    question: str,
    answer: str,
    contexts: list[str],
    confidence_score: float,
    retry_count: int,
    latency_ms: int
):
    """
    Asynchronous background runner.
    Calculates RAGAS metrics and inserts a record directly into 'eval_results'.
    """
    import math

    def sanitize_float(val):
        if val is None:
            return None
        try:
            if math.isnan(val) or math.isinf(val):
                return None
        except (TypeError, ValueError):
            pass
        return val

    faithfulness_val = None
    relevancy_val = None
    context_precision_val = None
    status = "success"
    try:
        # 1. Run standard RAGAS evaluation
        scores = await score_response(question, answer, contexts)
        if scores:
            faithfulness_val = sanitize_float(scores.get("faithfulness"))
            relevancy_val = sanitize_float(scores.get("answer_relevancy"))
    except Exception as e:
        status = "failed"
        logger.error(f"RAGAS evaluation scoring failed: {e}", exc_info=True)
    try:
        # 2. Package database payload to match 'eval_results' schema
        payload = {
            "message_id": message_id,
            "session_id": session_id,
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "faithfulness": faithfulness_val,
            "answer_relevancy": relevancy_val,
            "confidence_score": sanitize_float(confidence_score),
            "retry_count": retry_count,
            "latency_ms": latency_ms,
            "status": status
        }
        
        # 3. Log directly to your Supabase table
        insert_one("eval_results", payload)
        logger.info(f"Successfully saved evaluation record for message {message_id} with status: {status}")
        
    except Exception as e:
        logger.error(f"Failed to insert evaluation record into eval_results: {e}", exc_info=True)
    