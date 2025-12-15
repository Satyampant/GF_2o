
import os
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from settings import settings

@dataclass
class Memory:
    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")
    
    @property
    def timestamp(self) -> Optional[str]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None
    

class VectorStore:
    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    EMBEDDING_MODEL = "all-miniLM-L6-v2"
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9

    _instance: Optional["VectorStore"] = None
    _initialized: bool = False

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._validate_env_vars()
            self.model = SentenceTransformer(self.EMBEDDING_MODEL)
            self._client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            self._initialized = True

    def _validate_env_vars(self):
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing env variables: {', '.join(missing_vars)}")
        
    def _collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)
    
    def _create_collection(self):
        sample_embedding = self.model.encode("sample text")
        self._client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE
            )
        )


    def find_similar_memory(self, text:str) -> Optional[Memory]:
        if not self._collection_exists():
            return None
        
        results = self.search_memories(text, top_k=1)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None
    
    def store_memory(self, text: str, metadata: dict):
        if not self._collection_exists():
            self._create_collection()

        similar_memory = self.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id

        
        embedding = self.model.encode(text).tolist()
        point = PointStruct(
            id=metadata["id"],
            vector=embedding,
            payload={
                'text': text,
                **metadata
            }
        )
        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point]
        )

    def search_memories(self, text: str, top_k: int = 5) -> list[Memory]:
        if not self._collection_exists():
            return []
        
        embedding = self.model.encode(text).tolist()
        search_result = self._client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k
        )

        return [
            Memory(
                text=hit.payload['text'],
                metadata={k: v for k, v in hit.payload.items() if k != 'text'},
                score=hit.score  # Convert cosine distance to similarity
            )
            for hit in search_result
        ]
    

@lru_cache()
def get_vector_store() -> VectorStore:
    return VectorStore()
        
