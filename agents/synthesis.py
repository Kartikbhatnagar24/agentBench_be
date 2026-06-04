import re

from agents.models.pipeline import PipelineState
from rag_pipeline.llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from agents.prompts import synthesis_prompt

llm=get_llm()

async def synthesis_agent(state: PipelineState) -> PipelineState:
    # format chunks with unique tags for citation
    formatted = "\n\n".join([
        f"[CHUNK:{i}] {c['text']}" 
        for i, c in enumerate(state.retrieved_chunks)
    ])
    
    # Run the chain using astream and attach a specific tag so parent graph streaming can filter it
    chain = (synthesis_prompt | llm | StrOutputParser()).with_config(
        tags=["synthesis_stream"]
    )
    
    chunks = []
    async for chunk in chain.astream({
        "chunks": formatted,
        "question": state.original_query
    }):
        chunks.append(chunk)
        
    answer = "".join(chunks)
    
    # extract which citation indices were actually used, keeping only valid, in-bounds indices
    used = list(set(int(x) for x in re.findall(r'\[CHUNK:(\d+)\]', answer)))
    valid_used = sorted([x for x in used if x < len(state.retrieved_chunks)])
    
    state.synthesized_answer = answer
    state.citations = valid_used
    return state