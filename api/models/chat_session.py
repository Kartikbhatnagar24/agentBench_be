from pydantic import BaseModel

class ChatSession(BaseModel):
    user_id:str
    first_message:str
    messages:list[dict] = []
    documents:list[dict] = []