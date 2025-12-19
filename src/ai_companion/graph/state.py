from langgraph.graph import MessageState

class AICompanionState(MessageState):
    summary: str
    workflow: str
    audio_buffer: bytes
    image_path: str
    current_activity: str
    apply_activity: bool
    memory_context: str
