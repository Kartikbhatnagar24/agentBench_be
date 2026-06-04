from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()

def embedding_model():
    """
    This function is used to load the embedding model from the HuggingFace Hub.
    """
    hf_token = os.getenv("HF_TOKEN")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token":hf_token},
        show_progress=True,
    )
    return embeddings

def chunking_sementic(text:str):
    embeddings = embedding_model()
    try:
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        docs = splitter.create_documents([text])
        
        # Guard: Split any semantic chunk that exceeds 1200 characters to prevent
        # context truncation in all-MiniLM-L6-v2 (which has a 512-token limit).
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
