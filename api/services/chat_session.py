from utils.eval import get_state_val
from typing import Optional
from database import logger
from utils.chat import build_and_format_history
from rag_pipeline.llm import get_llm
from fastapi import WebSocket
from database import delete_one
from database import insert_one, find_many, find_one, update_one
from datetime import datetime
from api.models.chat_session import ChatSession
from fastapi import HTTPException
import uuid
import asyncio
from rag_pipeline.qdrant import search_by_session
from agents.graph.graph import build_graph
from agents.models.pipeline import PipelineState


class ChatSessionService:
    def create_chat_session(self, session: ChatSession):
        session_data = {
            "user_id": session.user_id,
            "first_message": session.first_message,
            "messages": [{
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": session.first_message,
                "timestamp": str(datetime.now()),
                "is_streaming": False
            }],
            "documents": []
        }
        result = insert_one("chat_sessions", session_data)
        return result
    def get_chat_sessions(self, user_id: str):
        return find_many("chat_sessions", {"user_id": user_id}, order_by="created_at", descending=True)

    def save_interaction(self, session_id: str, user_text: str, assistant_text: str, assistant_msg_id:Optional[str] ):
        """
        Saves both the user's message and the assistant's reply into the chat session.
        Used by the WebSocket endpoint after streaming the reply.
        """
        session = find_one("chat_sessions", {"id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        messages = session.get("messages", [])
        
        # Check if the user message is a duplicate of the initial first_message
        is_duplicate_first = (
            len(messages) == 1
            and messages[0].get("role") == "user"
            and messages[0].get("content") == user_text
        )
        
        if not is_duplicate_first:
            messages.append({
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": user_text,
                "timestamp": str(datetime.now()),
                "is_streaming": False
            })
            
        msg_id = assistant_msg_id if assistant_msg_id else str(uuid.uuid4())
        messages.append({
            "id": msg_id,
            "role": "assistant",
            "content": assistant_text,
            "timestamp": str(datetime.now()),
            "is_streaming": False
        })
        
        updated_session = update_one("chat_sessions", {"messages": messages}, record_id=session_id)
        return updated_session
    
    def delete_session(self,session_id:str):
        return delete_one("chat_sessions",{"id":session_id})
    
    def get_session_by_id(self,session_id:str):
        return find_one("chat_sessions",{"id":session_id})
    
    def _initialize_pipeline_state(self, user_message: str, session: Optional[dict], session_id: str) -> PipelineState:
        formatted_history = build_and_format_history(session)
        return PipelineState(
            original_query=user_message,
            chat_history=formatted_history,
            session_id=session_id,
            rewritten_queries=[],
            retrieved_chunks=[],
            synthesized_answer="",
            citations=[],
            verified_claims=[],
            confidence_score=0.0,
            final_answer="",
            retry_count=0
        )

    async def _stream_graph_events(
        self,
        websocket: WebSocket,
        graph,
        initial_state: PipelineState
    ) -> tuple[str, Optional[dict]]:
        final_answer = ""
        result_state = None
        try:
            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event.get("event")
                
                # A. Listen to chat model stream tokens from the synthesis node
                if kind == "on_chat_model_stream":
                    tags = event.get("tags", [])
                    if "synthesis_stream" in tags:
                        content = event["data"]["chunk"].content
                        if content:
                            final_answer += content
                            await websocket.send_text(content)
                
                # B. Listen to the end of the factcheck agent to catch the verified final answer 
                # (which adds warnings if any claims were unsupported)
                elif kind == "on_chain_end":
                    if event.get("name") == "factcheck_agent":
                        result_state = event["data"]["output"]
                        if isinstance(result_state, dict):
                            actual_final = result_state.get("final_answer") or result_state.get("synthesized_answer", "")
                        else:
                            actual_final = getattr(result_state, "final_answer", "") or getattr(result_state, "synthesized_answer", "")
                        
                        # If the factchecker appended a verification warning at the end of the text,
                        # stream the warning block to the user so it's visible on the UI.
                        if actual_final and len(actual_final) > len(final_answer):
                            diff = actual_final[len(final_answer):]
                            await websocket.send_text(diff)
                            final_answer = actual_final
                        elif actual_final:
                            final_answer = actual_final
                            
        except Exception as e:
            logger.error(f"Error during agent pipeline execution: {e}")
            await websocket.send_text("\n\n⚠️ An error occurred while processing your request through the agent pipeline.")
            final_answer = "Error during pipeline execution."
            result_state = None
            
        return final_answer, result_state

    def _trigger_background_eval(
        self,
        session_id: str,
        session: Optional[dict],
        user_message: str,
        final_answer: str,
        assistant_msg_id: str,
        result_state: Optional[dict],
        latency_ms: int
    ) -> None:
        raw_chunks = get_state_val(result_state, "retrieved_chunks", [])
        retrieved_chunks = raw_chunks if isinstance(raw_chunks, list) else []
        contexts = []
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                contexts.append(chunk.get("text", ""))
            else:
                contexts.append(getattr(chunk, "text", ""))
                
        raw_confidence = get_state_val(result_state, "confidence_score", 0.0)
        confidence_score = float(raw_confidence) if raw_confidence is not None else 0.0
        
        raw_retry = get_state_val(result_state, "retry_count", 0)
        retry_count = int(raw_retry) if raw_retry is not None else 0
        
        user_id = ""
        if session:
            raw_user_id = session.get("user_id")
            if isinstance(raw_user_id, str):
                user_id = raw_user_id
        
        # 11. Trigger evaluation asynchronously in the background
        from eval.ragas_scorer import evaluate_and_save_to_db
        asyncio.create_task(
            evaluate_and_save_to_db(
                session_id=session_id,
                user_id=user_id,
                message_id=assistant_msg_id,
                question=user_message,
                answer=final_answer,
                contexts=contexts,
                confidence_score=confidence_score,
                retry_count=retry_count,
                latency_ms=latency_ms
            )
        )

    async def websocket_streaming(self, websocket: WebSocket, session_id: str):
        llm = get_llm(streaming=True)
    
        # Wait for user message
        user_message = await websocket.receive_text()
        start_time = datetime.now()
        
        # Retrieve context using Qdrant (accessing via app state)
        vectorstore = websocket.app.state.vectorstore
        session = self.get_session_by_id(session_id)
        
        # 1. Initialize PipelineState
        initial_state = self._initialize_pipeline_state(user_message, session, session_id)
        
        # 2. Build graph and execute streaming events
        graph = build_graph(vectorstore)
        final_answer, result_state = await self._stream_graph_events(websocket, graph, initial_state)
        
        # 3. Signal completed transmission
        await websocket.send_text("[DONE]")
        
        # 4. Save the full interaction to DB
        assistant_msg_id = str(uuid.uuid4())
        self.save_interaction(session_id, user_message, final_answer, assistant_msg_id=assistant_msg_id)
        
        # 5. Extract metadata and raw text context for background RAGAS evaluation
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        self._trigger_background_eval(
            session_id=session_id,
            session=session,
            user_message=user_message,
            final_answer=final_answer,
            assistant_msg_id=assistant_msg_id,
            result_state=result_state,
            latency_ms=latency_ms
        )

        