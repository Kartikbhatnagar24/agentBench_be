
from database import logger
from api.models.chat_session import ChatSession
from api.services.chat_session import ChatSessionService
from fastapi import APIRouter, WebSocket, Request, WebSocketDisconnect, Depends
from utils.user_validation import get_current_user




router = APIRouter(prefix="/chat",tags=["chat"])

chat_session_service=ChatSessionService()

@router.post("/create-session")
async def create_session(session:ChatSession, current_user: dict = Depends(get_current_user)):
    """
    Create a new chat session thread.

    Expected Return (JSON dict):
    {
        "id": "session-uuid-string",
        "user_id": "user-uuid-string",
        "first_message": "Hello Agent",
        "messages": [
            {
                "id": "message-uuid-string",
                "role": "user",
                "content": "Hello Agent",
                "timestamp": "2026-05-29 17:25:52.123456",
                "is_streaming": false
            }
        ],
        "documents": []
    }
    """
    return chat_session_service.create_chat_session(session)

@router.get("/sessions")
async def get_sessions(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get all chat sessions for a specific user.

    Expected Return (JSON list of dicts):
    [
        {
            "id": "session-uuid-string",
            "user_id": "user-uuid-string",
            "first_message": "Hello Agent",
            "messages": [
                {
                    "id": "message-uuid-string",
                    "role": "user",
                    "content": "Hello Agent",
                    "timestamp": "2026-05-29 17:25:52.123456",
                    "is_streaming": false
                }
            ],
            "documents": []
        }
    ]
    """
    return chat_session_service.get_chat_sessions(user_id)

@router.websocket("/sessions/{session_id}/ws")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time RAG chat streaming.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001)
        return
    try:
        from utils.user_validation import verify_token
        verify_token(token)
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    try:
        while True:
            await chat_session_service.websocket_streaming(websocket,session_id)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
        logger.error(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        logger.error(f"WebSocket error: {e}")
    try:
        await websocket.close()
    except:
        pass
            
    

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    """
    Delete a chat session by ID. Also purges Qdrant vectors and Supabase
    storage files belonging to this session. All steps must succeed.
    """
    from fastapi import HTTPException
    errors: list[str] = []

    # Step 1: Delete all Qdrant vectors for this session
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        vectorstore = request.app.state.vectorstore
        vectorstore.client.delete(
            collection_name=vectorstore.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.session_id",
                        match=MatchValue(value=session_id),
                    )
                ]
            ),
        )
    except Exception as e:
        errors.append(f"Qdrant cleanup failed: {e}")

    # Step 2: Delete all Supabase storage files for this session
    if not errors:
        try:
            from database import storage_list, storage_delete
            prefix = f"Rag_testing/{user_id}/{session_id}"
            items = storage_list("rag_project", prefix) or []
            paths = [
                f"{prefix}/{item['name']}"
                for item in items
                if item.get("name") and item["name"] != ".emptyFolderPlaceholder"
            ]
            if paths:
                storage_delete("rag_project", paths)
        except Exception as e:
            errors.append(f"Supabase storage cleanup failed: {e}")

    # Step 3: Delete the session record from the database
    if not errors:
        try:
            res = chat_session_service.delete_session(session_id)
            if not res:
                errors.append("Session not found in database.")
        except Exception as e:
            errors.append(f"Database delete failed: {e}")

    if errors:
        raise HTTPException(
            status_code=500,
            detail=f"Session delete failed: {'; '.join(errors)}"
        )

    return {"status": "success", "message": "Chat session deleted successfully"}
