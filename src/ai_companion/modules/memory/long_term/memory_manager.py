
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import uuid
from datetime import datetime
from langchain_groq import ChatGroq

from settings import settings
from ai_companion.modules.memory.long_term.vector_store import get_vector_store, VectorStore
from ai_companion.core.prompts import MEMORY_ANALYSIS_PROMPT
from langchain_core.messages import HumanMessage, BaseMessage


class MemoryAnalysis(BaseModel):
    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_memory: Optional[str] = Field(..., description="The formatted memory to be stored")


class MemoryManager:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.logger = logging.getLogger(__name__)
        self.llm = ChatGroq(
            model = settings.SMALL_TEXT_MODEL_NAME,
            api_key = settings.GROQ_API_KEY,
            temperature=0.2,
            max_retries=2
        ).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message):
        prompt = MEMORY_ANALYSIS_PROMPT.format(message)
        return await self.llm.ainvoke(prompt)

    async def extract_and_store_memories(self, message:BaseMessage):
        if message.type != "human":
            return 

        analysis = await self.analyze_memory(message.content)

        if analysis.is_important and analysis.formatted_memory:
            similar = self.vector_store.find_similar_memory(analysis.formatted_memory)

            if similar:
                self.logger.info(f"Similar memory already exists: {analysis.formatted_memory}")
                return 

            self.logger.info(f"Storing memory : {analysis.formatted_memory}")
            self.vector_store.store_memory(
                text = analysis.formatted_memory,
                metadata = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat()
                }
            )

    def get_relevant_memories(self, context):
        memories = self.vector_store.search_memories(context, top_k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(f"Memory: {memory.text}, Score: {memory.score:.2f}")

        return [memory.text for memory in memories]

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        if not memories:
            return ""

        return "\n".join(f"- {memory}" for memory in memories)

def get_memory_manager() -> MemoryManager:
    return MemoryManager()
    
    