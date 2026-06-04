from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv
from rag_pipeline.chunking import get_embedding_model
load_dotenv()

# module-level references, set during lifespan
qdrant_client: Optional[QdrantClient] = None
vectorstore: Optional[QdrantVectorStore] = None

COLLECTION_NAME = "agentbench_docs"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2


def _index_chunks(vectorstore, chunks: list) -> None:
    """Background task: upsert chunks into Qdrant. Errors are logged, not raised."""
    try:
        vectorstore.add_documents(chunks)
    except Exception as e:
        if "doesn't exist" in str(e) or "Not found" in str(e):
            print(f"[background] Collection {vectorstore.collection_name} not found. Recreating...")
            from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
            vectorstore.client.create_collection(
                collection_name=vectorstore.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            vectorstore.client.create_payload_index(
                collection_name=vectorstore.collection_name,
                field_name="metadata.session_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            vectorstore.client.create_payload_index(
                collection_name=vectorstore.collection_name,
                field_name="metadata.file_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            vectorstore.add_documents(chunks)
        else:
            print("[background] Qdrant indexing failed:", e)

def search_by_session(vectorstore, query: str, session_id: str, k: int = 5) -> str:
    """Search Qdrant for relevant chunks, filtering by the given session_id."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build the Qdrant filter
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.session_id",
                    match=MatchValue(value=session_id),
                )
            ]
        )
        
        # Perform similarity search with the filter
        docs = vectorstore.similarity_search(
            query=query, 
            k=k, 
            filter=qdrant_filter
        )
        
        # Combine the text of the retrieved documents
        if not docs:
            return ""
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error during Qdrant search: {e}")
        return ""


async def asearch_by_session_with_score(vectorstore, query: str, session_id: str, k: int = 5):
    """Search Qdrant asynchronously for relevant chunks, filtering by session_id, returning chunks and scores."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.session_id",
                match=MatchValue(value=session_id),
            )
        ]
    )
    
    return await vectorstore.asimilarity_search_with_score(
        query=query,
        k=k,
        filter=qdrant_filter
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, vectorstore

    # --- startup ---
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        api_key=os.getenv("QDRANT_API"),
        timeout=60
    )

    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

    from qdrant_client.models import PayloadSchemaType
    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.session_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass
    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.file_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=get_embedding_model()
    )
    
    app.state.vectorstore = vectorstore

    yield  # app runs here

    # --- shutdown ---
    if qdrant_client:
        qdrant_client.close()