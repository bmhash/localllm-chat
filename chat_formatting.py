from typing import List, Dict, Optional

def create_system_prompt(model_name: Optional[str] = None) -> str:
    """
    Create a system prompt that helps maintain focus and context.
    Optionally includes model-specific instructions.
    """
    base_prompt = (
        "You are a helpful AI assistant. Follow these key principles:\n"
        "1. Be direct and concise - focus on providing clear, actionable information\n"
        "2. Maintain context - reference relevant parts of the conversation when appropriate\n"
        "3. Stay on topic - address the current question while considering previous context\n"
        "4. Be precise - provide specific details rather than general statements\n"
    )
    
    if model_name:
        # Add model-specific instructions if needed
        model_specific = {
            "codellama": "\nSpecialize in providing code-focused responses with explanations.",
            "deepseek-coder": "\nFocus on detailed code analysis and implementation.",
            "mistral": "\nBalance between technical depth and general assistance.",
        }
        base_prompt += model_specific.get(model_name.lower(), "")
    
    return base_prompt

def format_chat_messages(
    messages: List[Dict[str, str]], 
    current_model: Optional[str] = None,
    previous_model: Optional[str] = None,
    max_context_length: int = 4096
) -> str:
    """
    Format chat messages for model input with enhanced context handling.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        current_model: Name of the current model being used
        previous_model: Name of the previous model (if switching models)
        max_context_length: Maximum context length to maintain
        
    Returns:
        Formatted string ready for model input
    """
    formatted = []
    
    # Add system instruction with model awareness
    system_prompt = create_system_prompt(current_model)
    formatted.append(f"### System: {system_prompt}\n")
    
    # Handle model transition if applicable
    if previous_model and current_model and previous_model != current_model:
        transition_note = (
            f"Note: Conversation continuing from {previous_model} to {current_model}. "
            "Maintain context while adapting to new capabilities."
        )
        formatted.append(f"### System: {transition_note}\n")
    
    # Handle system message if present
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    if system_messages:
        formatted.append(f"### System: {system_messages[0]['content']}\n")
    
    # Process conversation messages with enhanced formatting
    conversation_messages = [msg for msg in messages if msg["role"] in ["user", "assistant"]]
    
    # Ensure we're within context length by keeping most recent messages
    total_length = sum(len(msg["content"]) for msg in conversation_messages)
    if total_length > max_context_length:
        # Keep system messages and trim conversation history
        while total_length > max_context_length and len(conversation_messages) > 1:
            removed_msg = conversation_messages.pop(0)
            total_length -= len(removed_msg["content"])
    
    # Format conversation messages with clear separation and focus
    for msg in conversation_messages:
        if msg["role"] == "user":
            formatted.append(f"### Human: {msg['content']}\n")
        elif msg["role"] == "assistant":
            # Add emphasis to assistant's key points
            content = msg["content"]
            if ":" in content:  # If there are key points or sections
                content = content.replace(":", ":\n")  # Add line breaks for readability
            formatted.append(f"### Assistant: {content}\n")
    
    # Add final prompt for assistant with focus reminder
    if current_model:
        formatted.append(f"### Assistant ({current_model}): ")
    else:
        formatted.append("### Assistant: ")
    
    return "\n".join(formatted)

def extract_key_points(message: str) -> List[str]:
    """
    Extract key points from a message to maintain focus in long conversations.
    Useful for summarizing context when switching models or handling long threads.
    
    Args:
        message: The message content to analyze
        
    Returns:
        List of key points extracted from the message
    """
    points = []
    
    # Split on common delimiter patterns
    delimiters = [":", ".", "\n"]
    current_point = ""
    
    for char in message:
        current_point += char
        if char in delimiters and len(current_point.strip()) > 10:  # Minimum length for a key point
            points.append(current_point.strip())
            current_point = ""
    
    # Add any remaining content
    if current_point.strip():
        points.append(current_point.strip())
    
    return points

def create_context_summary(messages: List[Dict[str, str]], max_points: int = 3) -> str:
    """
    Create a brief summary of the conversation context.
    Useful when switching models or when context length is constrained.
    
    Args:
        messages: List of previous messages
        max_points: Maximum number of key points to include
        
    Returns:
        Summarized context string
    """
    key_points = []
    
    # Process messages in reverse to get most recent context first
    for msg in reversed(messages):
        if len(key_points) >= max_points:
            break
            
        if msg["role"] == "assistant":
            points = extract_key_points(msg["content"])
            key_points.extend(points[:max_points - len(key_points)])
    
    if key_points:
        return "Previous context:\n- " + "\n- ".join(key_points)
    return ""
