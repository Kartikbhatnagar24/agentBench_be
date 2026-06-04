from collections import Counter
from statistics import mean
from typing import Any, Optional

from database import find_many


class AnalysisService:
    def get_overview(self, user_id: str):
        results = find_many("eval_results", {"user_id": user_id})
        sessions = find_many("chat_sessions", {"user_id": user_id})

        session_titles = {
            str(session.get("id")): session.get("first_message") or "Untitled conversation"
            for session in sessions
        }

        return {
            "user_id": user_id,
            "summary": self._build_summary(results, len(sessions)),
            "status_breakdown": self._build_status_breakdown(results),
            "session_breakdown": self._build_session_breakdown(results, session_titles),
            "weak_queries": self._build_weak_queries(results),
            "recent_metrics": self._build_recent_metrics(results),
        }

    def _build_summary(self, results: list[dict[str, Any]], total_sessions: int):
        total_queries = len(results)
        return {
            "total_sessions": total_sessions,
            "total_queries": total_queries,
            "avg_faithfulness": self._avg(results, "faithfulness"),
            "avg_answer_relevancy": self._avg(results, "answer_relevancy"),
            "avg_confidence_score": self._avg(results, "confidence_score"),
            "avg_latency_ms": round(self._avg(results, "latency_ms"), 2),
            "retry_rate": self._retry_rate(results),
            "failure_rate": self._failure_rate(results),
        }

    def _build_status_breakdown(self, results: list[dict[str, Any]]):
        counts = Counter(str(row.get("status") or "unknown") for row in results)
        return [{"status": status, "count": count} for status, count in counts.items()]

    def _build_session_breakdown(self, results: list[dict[str, Any]], session_titles: dict[str, str]):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in results:
            session_id = str(row.get("session_id") or "")
            if not session_id:
                continue
            grouped.setdefault(session_id, []).append(row)

        rows = []
        for session_id, records in grouped.items():
            rows.append({
                "session_id": session_id,
                "title": session_titles.get(session_id, "Untitled conversation"),
                "total_queries": len(records),
                "avg_faithfulness": self._avg(records, "faithfulness"),
                "avg_answer_relevancy": self._avg(records, "answer_relevancy"),
                "avg_confidence_score": self._avg(records, "confidence_score"),
                "avg_latency_ms": round(self._avg(records, "latency_ms"), 2),
                "retry_rate": self._retry_rate(records),
            })

        return sorted(
            rows,
            key=lambda item: (
                item["avg_faithfulness"] + item["avg_answer_relevancy"] + item["avg_confidence_score"]
            ) / 3,
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
