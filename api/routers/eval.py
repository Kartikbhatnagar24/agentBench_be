from fastapi import APIRouter
from api.services.eval import EvalService

router = APIRouter(prefix="/eval", tags=["Evaluation"])
eval_service = EvalService()

@router.get("/metrics")
async def get_metrics(session_id: str):
    """
    Get all evaluation rows for a session, useful for dashboards.

    Expected Return (JSON list of dicts):
    [
        {
            "message_id": "message-uuid-string",
            "session_id": "session-uuid-string",
            "user_id": "user-uuid-string",
            "question": "User question?",
            "answer": "Assistant answer",
            "faithfulness": 0.95,
            "answer_relevancy": 0.88,
            "confidence_score": 0.92,
            "retry_count": 0,
            "latency_ms": 1250,
            "status": "success"
        }
    ]
    """
    return eval_service.get_metrics_by_session(session_id)

@router.get("/metrics/{message_id}")
async def get_single_metric(message_id: str):
    """
    Get a single query breakdown by message_id.

    Expected Return (JSON dict):
    {
        "message_id": "message-uuid-string",
        "session_id": "session-uuid-string",
        "user_id": "user-uuid-string",
        "question": "User question?",
        "answer": "Assistant answer",
        "faithfulness": 0.95,
        "answer_relevancy": 0.88,
        "confidence_score": 0.92,
        "retry_count": 0,
        "latency_ms": 1250,
        "status": "success"
    }
    """
    return eval_service.get_metric_by_message(message_id)

@router.get("/summary")
async def get_summary(user_id: str):
    """
    Get aggregate user stats: avg scores, total queries, retry rate.

    Expected Return (JSON dict):
    {
        "user_id": "user-uuid-string",
        "total_queries": 15,
        "avg_faithfulness": 0.8923,
        "avg_answer_relevancy": 0.9145,
        "avg_confidence_score": 0.8842,
        "avg_latency_ms": 1320.5,
        "retry_rate": 0.1333
    }
    """
    return eval_service.get_user_summary(user_id)
