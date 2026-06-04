# pyrefly: ignore [missing-import]
from datetime import datetime
import asyncio
import uuid
from typing import Optional
from fastapi import UploadFile, File, HTTPException, Request, BackgroundTasks
from database import storage_upload, storage_download, storage_list, storage_delete
from rag_pipeline.chunking import chunk_semantic
from rag_pipeline.qdrant import _index_chunks
from utils.file_parser import (
    validate_file_extension,
    extract_text,
    parse_storage_item_name,
    format_size_kb,
    extract_text_preview,
)



class AttachmentService:
    def _session_prefix(self, user_id: str, session_id: str) -> str:
        return f"Rag_testing/{user_id}/{session_id}"

    async def upload_document(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
        user_id: str,
        session_id: str,
        file: UploadFile = File(...),
    ):
        validate_file_extension(file.filename)

        content = await file.read()
        text = await asyncio.to_thread(extract_text, file.filename, content)

        file_id = str(uuid.uuid4())

        # --- Chunking + Qdrant indexing ---
        chunks = chunk_semantic(text)
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "file_id": file_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "filename": file.filename,
                }
            )

        if chunks:
            # Synchronously wait for Qdrant indexing to complete so that
            # queries run right after upload will have access to the indexed documents.
            await asyncio.to_thread(
                _index_chunks, request.app.state.vectorstore, chunks
            )

        # --- Supabase Storage upload ---
        prefix = self._session_prefix(user_id, session_id)
        storage_path = f"{prefix}/{file_id}_{file.filename}"
        content_type = (
            "application/pdf"
            if file.filename and file.filename.endswith(".pdf")
            else "text/plain"
        )
        try:
            storage_upload("rag_project", storage_path, content, content_type)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload to Supabase Storage: {str(e)}",
            )

        return {
            "file_id": file_id,
            "filename": file.filename,
            "path": storage_path,
            "chunk_count": len(chunks),
            "content": text[:5000] if text else "",
        }

    def get_documents(self, user_id: str, session_id: str):
        docs = []
        prefix = self._session_prefix(user_id, session_id)
        try:
            for item in storage_list("rag_project", prefix):
                name = item.get("name")
                if not name or name == ".emptyFolderPlaceholder":
                    continue

                file_id, name_part = parse_storage_item_name(name)

                metadata = item.get("metadata", {}) or {}
                size_kb = format_size_kb(metadata.get("size", 0))

                file_bytes = storage_download("rag_project", f"{prefix}/{name}")
                text_content = extract_text_preview(file_bytes, name_part)

                docs.append(
                    {
                        "id": file_id,
                        "name": name_part,
                        "size": size_kb,
                        "content": text_content,
                        "uploaded_at": datetime.now(),
                    }
                )
        except Exception as e:
            print("Error listing documents:", e)
        return docs

    def delete_document(
        self, doc_id: str, user_id: str, session_id: str, request: Optional[Request] = None
    ):
        prefix = self._session_prefix(user_id, session_id)
        errors: list[str] = []

        # --- Step 1: Remove vectors from Qdrant ---
        if request is not None:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                vectorstore = request.app.state.vectorstore
                vectorstore.client.delete(
                    collection_name=vectorstore.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.file_id",
                                match=MatchValue(value=doc_id),
                            )
                        ]
                    ),
                )
            except Exception as e:
                errors.append(f"Qdrant cleanup failed: {e}")

        # --- Step 2: Remove file from Supabase Storage ---
        if not errors:
            try:
                deleted = False
                for item in storage_list("rag_project", prefix):
                    name = item.get("name")
                    if name and name.startswith(f"{doc_id}_"):
                        storage_delete("rag_project", [f"{prefix}/{name}"])
                        deleted = True
                        break
                if not deleted:
                    errors.append("File not found in storage.")
            except Exception as e:
                errors.append(f"Supabase storage cleanup failed: {e}")

        if errors:
            raise HTTPException(
                status_code=500,
                detail=f"Delete failed: {'; '.join(errors)}"
            )
