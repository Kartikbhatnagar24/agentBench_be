from agents.models.pipeline import PipelineState, FactCheckSchema
from rag_pipeline.llm import get_llm
from agents.prompts import factcheck_prompt
from utils.validate_agents import _validate_factcheck
from utils.retry_agent import retry_llm_invoke

llm = get_llm()


async def factcheck_agent(state: PipelineState) -> PipelineState:
    # Pass ALL retrieved chunks to the fact-checker so it can robustly verify claims
    all_chunks_formatted = "\n\n".join([
        f"[CHUNK:{i}] {chunk['text']}"
        for i, chunk in enumerate(state.retrieved_chunks)
    ])

    # Invoke with automatic retry on bad output
    result: dict = await retry_llm_invoke(
        chain=factcheck_prompt | llm.with_structured_output(FactCheckSchema),
        inputs={"answer": state.synthesized_answer, "sources": all_chunks_formatted},
        validator=_validate_factcheck,
        error_prefix="Fact-check",
    )

    # Map numeric indexes back to actual Qdrant chunk_ids and document filenames
    mapped_claims = []
    for claim in result.get("claims", []):
        idx_val = claim.get("source_chunk_index")
        source_chunk_id = None
        source_document = "Unknown"
        
        if idx_val is not None:
            try:
                # Handle cases where LLM returned "CHUNK:0" or 0
                clean_idx = int(str(idx_val).replace("CHUNK:", "").strip())
                if clean_idx < len(state.retrieved_chunks):
                    source_chunk_id = state.retrieved_chunks[clean_idx].get("chunk_id")
                    source_document = state.retrieved_chunks[clean_idx]["metadata"].get("filename", "Unknown")
            except ValueError:
                pass
        
        # Append rich metadata to the verified claim
        mapped_claims.append({
            "claim": claim["claim"],
            "supported": claim["supported"],
            "source_chunk_index": idx_val,
            "source_chunk_id": source_chunk_id,      # Actual Qdrant UUID
            "source_document": source_document,      # filename metadata (e.g., mtp_2.pdf)
            "confidence": claim["confidence"],
            "reasoning": claim["reasoning"]
        })

    state.verified_claims = mapped_claims
    state.confidence_score = float(result["overall_confidence"])
    print("####################################")
    print("Fact check")
    for i,cl in enumerate(state.verified_claims):
        print(f"Claim {i}")
        print("Claim:",cl["claim"])
        print("Supported:",cl["supported"])
        print("Source Chunk Index:",cl["source_chunk_index"])
        print("Source Chunk ID:",cl["source_chunk_id"])
        print("Source Document:",cl["source_document"])
        print("Confidence:",cl["confidence"])
        print("Reasoning:",cl["reasoning"])
        print("###################################")
    print("###################################")

    # Build final answer — append warning if any claims are unsupported
    unsupported = [c for c in mapped_claims if not c["supported"]]
    if unsupported:
        warning = f"\n\n⚠️ {len(unsupported)} claim(s) could not be fully verified against sources."
        state.final_answer = state.synthesized_answer + warning
    else:
        state.final_answer = state.synthesized_answer

    return state