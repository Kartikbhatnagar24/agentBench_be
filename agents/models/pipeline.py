from pydantic import BaseModel
from typing import Optional

class PipelineState(BaseModel):
    # input
    original_query: str
    chat_history: str=""
    session_id: str = ""
    
    # Agent 1 output
    rewritten_queries: list       # 2-3 variants
    retrieved_chunks: list[dict]      # chunk text + metadata + score
    
    # Agent 2 output  
    synthesized_answer: str = ""
    citations: list[int] = []              # indices of chunks actually used
    
    # Agent 3 output
    verified_claims: list[dict] = []       # {claim, supported: bool, source_chunk}
    confidence_score: float = 0.0
    final_answer: str = ""
    retry_count:int=0


from typing import List
from pydantic import Field

class QueryRewriteSchema(BaseModel):
    queries: List[str] = Field(
        description="Exactly 3 semantically diverse alternative search queries."
    )

class ClaimVerification(BaseModel):
    claim: str = Field(description="The exact sentence or phrase from the answer being evaluated.")
    supported: bool = Field(description="True if the claim is directly supported by the source chunks, False otherwise.")
    source_chunk_index: Optional[int] = Field(description="The integer index of the supporting source chunk, or null if unsupported.")
    confidence: float = Field(description="Continuous float between 0.0 and 1.0 representing confidence in support.")
    reasoning: str = Field(description="One sentence explaining your verdict.")

class FactCheckSchema(BaseModel):
    claims: List[ClaimVerification] = Field(description="List of all extracted and verified claims.")
    overall_confidence: float = Field(description="Overall confidence score for the entire answer.")



