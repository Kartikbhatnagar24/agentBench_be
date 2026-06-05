from collections import Counter
from statistics import mean
from typing import Any, Optional

from database import find_many


class AnalysisService:
    def get_overview(self, user_id: str):
        results = find_many("eval_results", {"user_id": user_id}, order_by="created_at", descending=False)
        sessions = find_many("chat_sessions", {"user_id": user_id})

        session_info = {
            str(session.get("id")): {
                "title": session.get("first_message") or "Untitled conversation",
                "created_at": session.get("created_at") or "",
            }
            for session in sessions
        }

        return {
            "user_id": user_id,
            "summary": self._build_summary(results, len(sessions)),
            "status_breakdown": self._build_status_breakdown(results),
            "session_breakdown": self._build_session_breakdown(results, session_info),
            "weak_queries": self._build_weak_queries(results),
            "recent_metrics": self._build_recent_metrics(results),
        }

    def _build_summary(self, results: list[dict[str, Any]], total_sessions: int):
        total_queries = len(results)
        
        faithfulness = self._avg(results, "faithfulness")
        relevancy = self._avg(results, "answer_relevancy")
        confidence = self._avg(results, "confidence_score")
        
        # R_adjusted = mean relevancy of rows where query_type = "factual" only
        factual_rows = [row for row in results if not self._is_summary_query(row)]
        
        if len(factual_rows) > 0:
            r_adjusted = self._avg(factual_rows, "answer_relevancy")
            pipeline_score_raw = (faithfulness * 0.5) + (confidence * 0.3) + (r_adjusted * 0.2)
        else:
            # If no factual queries exist, renormalize
            pipeline_score_raw = (faithfulness * 0.625) + (confidence * 0.375)
            
        pipeline_score = round(pipeline_score_raw * 100, 1)
        
        return {
            "total_sessions": total_sessions,
            "total_queries": total_queries,
            "avg_faithfulness": faithfulness,
            "avg_answer_relevancy": relevancy,
            "avg_confidence_score": confidence,
            "avg_latency_ms": round(self._avg(results, "latency_ms"), 2),
            "retry_rate": self._retry_rate(results),
            "failure_rate": self._failure_rate(results),
            "pipeline_score": pipeline_score,
        }

    def _build_status_breakdown(self, results: list[dict[str, Any]]):
        counts = Counter(str(row.get("status") or "unknown") for row in results)
        return [{"status": status, "count": count} for status, count in counts.items()]

    def _build_session_breakdown(self, results: list[dict[str, Any]], session_info: dict[str, dict[str, Any]]):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in results:
            session_id = str(row.get("session_id") or "")
            if not session_id:
                continue
            grouped.setdefault(session_id, []).append(row)

        rows = []
        for session_id, records in grouped.items():
            meta = session_info.get(session_id) or {}
            created_at = meta.get("created_at") or ""

            rows.append({
                "session_id": session_id,
                "title": meta.get("title") or "Untitled conversation",
                "created_at": created_at,
                "total_queries": len(records),
                "avg_faithfulness": self._avg(records, "faithfulness"),
                "avg_answer_relevancy": self._avg(records, "answer_relevancy"),
                "avg_confidence_score": self._avg(records, "confidence_score"),
                "avg_latency_ms": round(self._avg(records, "latency_ms"), 2),
                "retry_rate": self._retry_rate(records),
                "queries": [
                    {
                        "message_id": row.get("message_id"),
                        "question": row.get("question") or "",
                        "answer": row.get("answer") or "",
                        "faithfulness": self._float_or_zero(row.get("faithfulness")),
                        "answer_relevancy": self._float_or_zero(row.get("answer_relevancy")),
                        "confidence_score": self._float_or_zero(row.get("confidence_score")),
                        "retry_count": int(row.get("retry_count") or 0),
                        "latency_ms": self._float_or_zero(row.get("latency_ms")),
                        "status": row.get("status") or "unknown",
                        "is_summary": self._is_summary_query(row),
                    }
                    for row in records
                ]
            })

        return sorted(
            rows,
            key=lambda item: item.get("created_at") or "",
            reverse=True,
        )

    def _build_weak_queries(self, results: list[dict[str, Any]]):
        ranked = sorted(
            results,
            key=lambda row: self._quality_score(row),
        )
        return [
            {
                "message_id": row.get("message_id"),
                "session_id": row.get("session_id"),
                "question": row.get("question") or "",
                "answer": row.get("answer") or "",
                "faithfulness": self._float_or_zero(row.get("faithfulness")),
                "answer_relevancy": self._float_or_zero(row.get("answer_relevancy")),
                "confidence_score": self._float_or_zero(row.get("confidence_score")),
                "retry_count": int(row.get("retry_count") or 0),
                "latency_ms": self._float_or_zero(row.get("latency_ms")),
                "status": row.get("status") or "unknown",
                "is_summary": self._is_summary_query(row),
            }
            for row in ranked[:5]
        ]

    def _build_recent_metrics(self, results: list[dict[str, Any]]):
        recent = results[-10:]
        return [
            {
                "message_id": row.get("message_id"),
                "session_id": row.get("session_id"),
                "quality_score": self._quality_score(row),
                "faithfulness": self._float_or_zero(row.get("faithfulness")),
                "answer_relevancy": self._float_or_zero(row.get("answer_relevancy")),
                "confidence_score": self._float_or_zero(row.get("confidence_score")),
                "latency_ms": self._float_or_zero(row.get("latency_ms")),
                "status": row.get("status") or "unknown",
            }
            for row in recent
        ]

    def _quality_score(self, row: dict[str, Any]) -> float:
        values = [
            self._optional_float(row.get("faithfulness")),
            self._optional_float(row.get("answer_relevancy")),
            self._optional_float(row.get("confidence_score")),
        ]
        present = [value for value in values if value is not None]
        if not present:
            return 0.0
        return round(mean(present), 4)

    def _avg(self, results: list[dict[str, Any]], key: str) -> float:
        values = [self._optional_float(row.get(key)) for row in results]
        present = [value for value in values if value is not None]
        if not present:
            return 0.0
        return round(mean(present), 4)

    def _retry_rate(self, results: list[dict[str, Any]]) -> float:
        if not results:
            return 0.0
        retried = sum(1 for row in results if int(row.get("retry_count") or 0) > 0)
        return round(retried / len(results), 4)

    def _failure_rate(self, results: list[dict[str, Any]]) -> float:
        if not results:
            return 0.0
        failed = sum(1 for row in results if str(row.get("status") or "").lower() == "failed")
        return round(failed / len(results), 4)

    def _optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _float_or_zero(self, value: Any) -> float:
        return self._optional_float(value) or 0.0

    def _is_summary_query(self, row: dict[str, Any]) -> bool:
        is_summary = row.get("is_summary")
        if is_summary is not None:
            return bool(is_summary)
        question_lower = (row.get("question") or "").lower()
        summary_keywords = ["explain", "summarize", "what does", "what is in", "contains", "overview", "tell me about"]
        return any(kw in question_lower for kw in summary_keywords)
