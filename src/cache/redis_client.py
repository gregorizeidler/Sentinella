"""
Redis Cache Client - Response Caching Layer

This module provides Redis-based caching to reduce costs and latency
by caching LLM responses.
"""

import json
import hashlib
import os
from typing import Optional, Dict, Any, List
import redis.asyncio as redis
from redis.asyncio import Redis


class RedisCache:
    """
    Redis-based cache for LLM responses
    
    Cache Strategy:
    - Key: hash(model + messages + temperature)
    - TTL: Configurable (default 1 hour)
    - Benefits: Cost reduction, latency improvement
    """
    
    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.password = os.getenv("REDIS_PASSWORD")
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.default_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
        self.client: Optional[Redis] = None
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
        }
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = await redis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password,
                decode_responses=True,
            )
            # Test connection
            await self.client.ping()
        except Exception as e:
            print(f"Warning: Redis connection failed: {e}. Caching disabled.")
            self.client = None
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
    
    def generate_key(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate cache key from request parameters
        
        Key includes:
        - Model name
        - Message content
        - Temperature (if provided)
        """
        # Create deterministic string representation
        key_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        
        # Hash for consistent key length
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return f"sentinella:cache:{key_hash}"
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response
        
        Returns:
            Cached response dict or None if not found
        """
        if not self.client:
            return None
        
        try:
            cached_value = await self.client.get(key)
            if cached_value:
                self.stats["hits"] += 1
                return json.loads(cached_value)
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            print(f"Cache get error: {e}")
            self.stats["misses"] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """
        Cache a response
        
        Args:
            key: Cache key
            value: Response to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if not self.client:
            return
        
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value)
            await self.client.setex(key, ttl, serialized_value)
            self.stats["sets"] += 1
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete a cached entry"""
        if not self.client:
            return
        
        try:
            await self.client.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    async def clear(self, pattern: str = "sentinella:cache:*"):
        """Clear all cache entries matching pattern"""
        if not self.client:
            return
        
        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.client.delete(*keys)
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    async def health_check(self) -> bool:
        """Check Redis connection health"""
        if not self.client:
            return False
        
        try:
            await self.client.ping()
            return True
        except Exception:
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "hit_rate": round(hit_rate * 100, 2),
            "connected": self.client is not None,
        }

