from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()

_embeddings_instance = None

def embedding_model():
    """
    This function is used to load the embedding model from the HuggingFace Hub.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        hf_token = os.getenv("HF_TOKEN")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"token": hf_token},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={
                "prompt": "Represent this sentence for searching relevant passages: ",
                "normalize_embeddings": True,
            },
            show_progress=True,
        )
    return _embeddings_instance

def chunking_sementic(text:str):
    embeddings = embedding_model()
    try:
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        docs = splitter.create_documents([text])
        
        # Guard: Split any semantic chunk that exceeds 1200 characters to prevent
        # context truncation in BAAI/bge-small-en-v1.5 (which has a 512-token limit).
        sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        final_docs = []
        for doc in docs:
            if len(doc.page_content) > 1200:
                sub_docs = sub_splitter.create_documents(
                    [doc.page_content],
                    metadatas=[doc.metadata] * len(sub_splitter.split_text(doc.page_content))
                )
                final_docs.extend(sub_docs)
            else:
                final_docs.append(doc)
        return final_docs
    except Exception as e:
        print(f"Error during chunking: {e}")
        # Robust fallback splitter
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return fallback_splitter.create_documents([text])

# Public aliases used by other modules
chunk_semantic = chunking_sementic
get_embedding_model = embedding_model
