from database import logger
from agents.models.pipeline import PipelineState, QueryRewriteSchema
from utils.validate_agents import _validate_queries
from utils.retry_agent import retry_llm_invoke
from rag_pipeline.llm import get_llm
from agents.prompts import rewrite_prompt
from rag_pipeline.qdrant import asearch_by_session_with_score
from rag_pipeline.ranker import ranker

llm = get_llm()

def make_retrieval_agent(vectorstore):
    """Factory that binds `vectorstore` into the node closure."""

    async def retrieval_agent(state: PipelineState) -> PipelineState:
        # Detect if we're running against the RAGbench eval collection.
        # RAGbench chunk IDs are sequential integers scoped per-document, so they
        # are NOT globally unique. We need a composite key to deduplicate them.
        # In production, chunk IDs are UUIDs and are globally unique on their own.
        is_ragbench = getattr(vectorstore, "collection_name", "") == "ragbench_eval"
        # Step 0: Identify failed claims from previous attempts to build feedback
        failed_claims = [c for c in state.verified_claims if not c["supported"]]
        
        feedback_instruction = ""
        feedback_context = ""
        
        if state.verified_claims and failed_claims:
            state.retry_count += 1
            feedback_instruction = (
                "\n## CRITICAL RETRY GUIDELINE:\n"
                "This is a RETRY attempt. The previous answer failed verification because "
                "certain claims could not be supported by retrieved text. You MUST generate "
                "query variants specifically tailored to find and retrieve documents that "
                "can verify, clarify, or address the failed claims listed in the human message."
            )
            
            claims_list = "\n".join([
                f"- Claim: \"{c['claim']}\"\n  Reasoning: {c['reasoning']}"
                for c in failed_claims
            ])
            feedback_context = (
                f"### PREVIOUS ATTEMPT FAILURE DETAILS:\n"
                f"The following claims made in the previous answer were UNSUPPORTED by the retrieved context:\n"
                f"{claims_list}\n\n"
                f"Please focus your alternative search queries on retrieving documentation to verify or address these specific points."
            )

        # Step 1: generate query variants — retried automatically on bad output
        queries: list[str] = await retry_llm_invoke(
            chain=rewrite_prompt | llm.with_structured_output(QueryRewriteSchema),
            inputs={
                "question": state.original_query,
                "chat_history": state.chat_history,
                "feedback_instruction": feedback_instruction,
                "feedback_context": feedback_context
            },
            validator=_validate_queries,
            error_prefix="Query rewrite",
        )

        state.rewritten_queries = queries
        print(f"Rewritten queries (Retry Count: {state.retry_count}): {queries}")
        
        # Step 2: retrieve for each variant, deduplicate by chunk id
        seen_ids = set()
        all_chunks = []

        # Seed pool with previously retrieved chunks so we don't lose them
        if state.retrieved_chunks:
            for chunk in state.retrieved_chunks:
                cid = chunk.get("chunk_id")
                if is_ragbench:
                    metadata = chunk.get("metadata") or {}
                    doc_identifier = metadata.get("filename") or metadata.get("document_id") or metadata.get("file_id") or "Unknown"
                    unique_id = f"{doc_identifier}_{cid}"
                else:
                    unique_id = str(cid)
                if unique_id not in seen_ids:
                    seen_ids.add(unique_id)
                    all_chunks.append(chunk)

        for query in state.rewritten_queries:
            chunks = await asearch_by_session_with_score(vectorstore, query, state.session_id, k=8)
            for chunk, score in chunks:
                chunk_id = chunk.metadata.get("chunk_id")
                if is_ragbench:
                    doc_identifier = chunk.metadata.get("filename") or chunk.metadata.get("document_id") or chunk.metadata.get("file_id") or "Unknown"
                    unique_id = f"{doc_identifier}_{chunk_id}"
                else:
                    unique_id = str(chunk_id)

                if unique_id not in seen_ids:
                    seen_ids.add(unique_id)
                    all_chunks.append({
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                        "score": float(score),
                        "chunk_id": chunk_id,
                    })
        if all_chunks:
            pairs = [[state.original_query, chunk["text"]] for chunk in all_chunks]
            ranking_results = ranker.invoke(pairs)
            
            # Attach scores back to chunks
            for i, (original_query, chunk_text) in enumerate(pairs):
                all_chunks[i]["score"] = ranking_results[i]

        # Step 3: rank by score, keep top 8 to allow a richer context on retries
        state.retrieved_chunks = sorted(all_chunks, key=lambda x: x["score"], reverse=True)[:8]
        

        return state

    return retrieval_agent