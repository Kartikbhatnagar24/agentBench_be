from typing import Any, Callable
async def retry_llm_invoke(
    chain,
    inputs: dict,
    validator: Callable[[Any, int], tuple[Any, str | None]],
    *,
    max_attempts: int = 3,
    error_prefix: str = "LLM invoke",
) -> Any:
    """
    Invoke a LangChain chain with automatic retry on bad output.

    Args:
        chain:          Any callable/chain supporting `.ainvoke(inputs)`.
        inputs:         Dict of inputs to pass to the chain.
        validator:      A callable(raw, attempt) -> (value | None, error | None).
                        Return (parsed_value, None) on success,
                        or (None, error_message) to trigger a retry.
        max_attempts:   Maximum number of LLM calls before raising.
        error_prefix:   Label used in the final ValueError message.

    Returns:
        The validated, parsed output from the chain.

    Raises:
        ValueError: If all attempts are exhausted without a valid response.
    """
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            raw = await chain.ainvoke(inputs)
            value, error = validator(raw, attempt)
            if error is None:
                return value          # ✅ valid output
            last_error = error        # ❌ retry
        except Exception as e:
            last_error = f"Exception during invoke/parse (attempt {attempt}): {e}"

    raise ValueError(
        f"{error_prefix} failed after {max_attempts} attempt(s). "
        f"Last error: {last_error}"
    )
