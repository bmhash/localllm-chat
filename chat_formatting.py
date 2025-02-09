from typing import List, Dict

def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """
    Format chat messages for the model input.
    Uses a clear delimiter format to separate different message types.
    """
    formatted = ""
    
    # Add initial instruction to be concise
    formatted += "### System: You are a helpful AI assistant. Always provide direct and concise responses. Stay on topic and only answer what is asked.\n\n"
    
    # Handle system message if present
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    if system_messages:
        formatted += f"### System: {system_messages[0]['content']}\n\n"
    
    # Handle conversation messages
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"### Human: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            formatted += f"### Assistant: {msg['content']}\n\n"
    
    # Add final prompt for assistant
    formatted += "### Assistant: "
    
    return formatted
