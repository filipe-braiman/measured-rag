import uuid

def new_conversation_id():
    return str(uuid.uuid4())[:8]

def get_query_id(chat_history):
    user_turns = sum(1 for msg in chat_history if msg.get("role") == "user")
    return user_turns + 1