"""
agents/utils.py
---------------
Shared utilities for all pipeline agents.
"""

from typing import Any
from agents.models.pipeline import QueryRewriteSchema, FactCheckSchema

def _validate_queries(raw: Any, attempt: int) -> tuple[list[str] | None, str | None]:
    """
    Validate that the LLM returned queries, either as a Pydantic object or dict.

    Returns:
        (parsed_value, None)   on success
        (None, error_message)  on failure
    """
    if isinstance(raw, QueryRewriteSchema) or hasattr(raw, "queries"):
        queries = raw.queries
    elif isinstance(raw, dict) and "queries" in raw:
        queries = raw["queries"]
    else:
        return None, f"Expected QueryRewriteSchema or dict with 'queries' key (attempt {attempt}): {raw}"

    if not isinstance(queries, list) or not queries:
        return None, f"'queries' is empty or not a list (attempt {attempt}): {queries}"
    if not all(isinstance(q, str) for q in queries):
        return None, f"Non-string item in queries (attempt {attempt}): {queries}"
    return queries, None


def _validate_factcheck(raw: Any, attempt: int) -> tuple[dict | None, str | None]:
    """
    Validate the fact-check structure from a Pydantic object or dict.

    Returns:
        (parsed_value, None)   on success
        (None, error_message)  on failure
    """
    if isinstance(raw, FactCheckSchema) or (hasattr(raw, "claims") and hasattr(raw, "overall_confidence")):
        if hasattr(raw, "model_dump"):
            raw_dict = raw.model_dump()
        else:
            raw_dict = raw.dict()
    elif isinstance(raw, dict):
        raw_dict = raw
    else:
        return None, f"Expected FactCheckSchema or dict, got {type(raw).__name__} (attempt {attempt})"

    if "claims" not in raw_dict or "overall_confidence" not in raw_dict:
        return None, f"Missing required keys 'claims'/'overall_confidence' (attempt {attempt}): {list(raw_dict.keys())}"
    if not isinstance(raw_dict["claims"], list):
        return None, f"'claims' must be a list (attempt {attempt}): {raw_dict['claims']}"
    if not isinstance(raw_dict["overall_confidence"], (int, float)):
        return None, f"'overall_confidence' must be a number (attempt {attempt}): {raw_dict['overall_confidence']}"

    bad_claims = [
        c for c in raw_dict["claims"]
        if not isinstance(c, dict)
        or "claim" not in c
        or "supported" not in c
        or "confidence" not in c
    ]
    if bad_claims:
        return None, f"Malformed claim objects (attempt {attempt}): {bad_claims}"

    return raw_dict, None



