from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage

def build_and_foramt_history(session: dict | None) -> str:
    history: list[BaseMessage] = []
    if session:
        # Get only the last 5 messages to avoid overly long contexts
        for msg in session.get("messages", [])[-5:]:
            if msg["role"]=="user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        
    formatted_history = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in history
        ])
    return formatted_history