from database import find_many, find_one
from fastapi import HTTPException
from typing import Optional

class EvalService:
    def get_metrics_by_session(self, session_id: str):
        """
        Fetch all evaluation records for a given session.
        """
        return find_many("eval_results", {"session_id": session_id})

    def get_metric_by_message(self, message_id: str):
        """
        Fetch a single evaluation record by message ID.
        """
        result = find_one("eval_results", {"message_id": message_id})
        if not result:
            raise HTTPException(status_code=404, detail=f"Evaluation record for message_id '{message_id}' not found.")
        return result

    def get_user_summary(self, user_id: str):
        """
        Fetch all evaluation records for a user and calculate aggregate metrics.
        """
        results = find_many("eval_results", {"user_id": user_id})
        total_queries = len(results)

        if total_queries == 0:
            return {
                "user_id": user_id,
                "total_queries": 0,
                "avg_faithfulness": 0.0,
                "avg_answer_relevancy": 0.0,
                "avg_confidence_score": 0.0,
                "avg_latency_ms": 0.0,
                "retry_rate": 0.0
            }

        # Filter out None values to calculate accurate averages
        faithfulness_vals = [float(r["faithfulness"]) for r in results if r.get("faithfulness") is not None]
        relevancy_vals = [float(r["answer_relevancy"]) for r in results if r.get("answer_relevancy") is not None]
        confidence_vals = [float(r["confidence_score"]) for r in results if r.get("confidence_score") is not None]
        latency_vals = [float(r["latency_ms"]) for r in results if r.get("latency_ms") is not None]

        avg_faithfulness = sum(faithfulness_vals) / len(faithfulness_vals) if faithfulness_vals else 0.0
        avg_relevancy = sum(relevancy_vals) / len(relevancy_vals) if relevancy_vals else 0.0
        avg_confidence = sum(confidence_vals) / len(confidence_vals) if confidence_vals else 0.0
        avg_latency = sum(latency_vals) / len(latency_vals) if latency_vals else 0.0

        # Calculate retry rate: fraction of queries where retry_count > 0
        queries_with_retries = sum(
            1 for r in results if int(r.get("retry_count") or 0) > 0
        )
        retry_rate = queries_with_retries / total_queries

        return {
            "user_id": user_id,
            "total_queries": total_queries,
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_answer_relevancy": round(avg_relevancy, 4),
            "avg_confidence_score": round(avg_confidence, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "retry_rate": round(retry_rate, 4)
        }
