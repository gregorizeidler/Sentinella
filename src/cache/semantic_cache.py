"""Semantic cache using embeddings"""

import hashlib
import json
from typing import Optional, Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
import faiss


class SemanticCache:
    """
    Semantic cache that finds similar cached responses using embeddings
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.threshold = 0.85  # Similarity threshold
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
    
    async def initialize(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize semantic cache"""
        # Connect to Redis
        self.redis_client = await redis.from_url(
            f"redis://{redis_host}:{redis_port}/3",
            decode_responses=True,
        )
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
    
    async def get_similar(
        self,
        prompt: str,
        model: str,
        temperature: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find similar cached response
        
        Returns cached response if similarity > threshold, else None
        """
        if not self.embedding_model or not self.redis_client:
            return None
        
        # Generate embedding for prompt
        prompt_embedding = self.embedding_model.encode([prompt])[0]
        prompt_embedding = prompt_embedding / np.linalg.norm(prompt_embedding)  # Normalize
        
        # Search in FAISS index
        if self.index.ntotal == 0:
            return None
        
        # Reshape for FAISS
        query_vector = prompt_embedding.reshape(1, -1).astype('float32')
        
        # Search top 5 similar
        k = min(5, self.index.ntotal)
        similarities, indices = self.index.search(query_vector, k)
        
        # Check if any similarity exceeds threshold
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= self.threshold:
                # Get cached response from Redis
                cache_key = f"semantic_cache:{idx}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    response = json.loads(cached_data)
                    # Verify model and temperature match
                    if response.get("model") == model and response.get("temperature") == temperature:
                        return response
        
        return None
    
    async def store(
        self,
        prompt: str,
        response: Dict[str, Any],
        model: str,
        temperature: Optional[float] = None,
    ):
        """Store response in semantic cache"""
        if not self.embedding_model or not self.redis_client or not self.index:
            return
        
        # Generate embedding
        prompt_embedding = self.embedding_model.encode([prompt])[0]
        prompt_embedding = prompt_embedding / np.linalg.norm(prompt_embedding)
        
        # Add to FAISS index
        idx = self.index.ntotal
        self.index.add(prompt_embedding.reshape(1, -1).astype('float32'))
        
        # Store in Redis
        cache_key = f"semantic_cache:{idx}"
        cache_data = {
            "prompt": prompt,
            "response": response,
            "model": model,
            "temperature": temperature,
            "embedding": prompt_embedding.tolist(),
        }
        await self.redis_client.setex(
            cache_key,
            3600 * 24,  # 24 hours TTL
            json.dumps(cache_data),
        )

