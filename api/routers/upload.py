from fastapi import APIRouter, File, UploadFile, HTTPException, Request, BackgroundTasks
from api.services.upload import AttachmentService

router = APIRouter(prefix="/chat",tags=["chat"])
service = AttachmentService()

@router.post('/upload')
async def upload(request: Request, background_tasks: BackgroundTasks, user_id: str, session_id: str, file: UploadFile = File(...)):
    """
    Upload and index a plain text (.txt) or PDF (.pdf) document.

    Expected Return (JSON dict):
    {
        "id": "file-uuid-string",
        "name": "filename.ext",
        "path": "Rag_testing/{user_id}/{session_id}/filename_{uuid}.ext",
        "content": "Full parsed text content of the uploaded document..."
    }
    """
    return await service.upload_document(request, background_tasks, user_id, session_id, file)

@router.get('/documents')
async def get_documents(user_id: str, session_id: str):
    """
    Retrieve all uploaded and indexed documents for a specific user session.

    Expected Return (JSON list of dicts):
    [
        {
            "id": "file-uuid-string",
            "name": "filename",
            "size": "15.4 KB",
            "content": "Snippet of text contents up to 5000 characters...",
            "uploaded_at": "Today"
        }
    ]
    """
    return service.get_documents(user_id, session_id)

@router.delete('/documents/{doc_id}')
async def delete_document(request: Request, doc_id: str, user_id: str, session_id: str):
    """
    Delete an indexed document from RAG storage.

    Expected Return (JSON dict):
    {
        "status": "success"
    }
    """
    service.delete_document(doc_id, user_id, session_id, request)
    return {"status": "success"}