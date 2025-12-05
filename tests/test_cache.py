"""Tests for Redis cache"""

import pytest
from src.cache.redis_client import RedisCache


def test_cache_key_generation():
    """Test cache key generation"""
    cache = RedisCache()
    
    messages = [{"role": "user", "content": "Hello"}]
    key1 = cache.generate_key("gpt-4", messages, 0.7)
    key2 = cache.generate_key("gpt-4", messages, 0.7)
    key3 = cache.generate_key("gpt-3.5-turbo", messages, 0.7)
    
    # Same inputs should generate same key
    assert key1 == key2
    
    # Different model should generate different key
    assert key1 != key3
    
    # Key should start with prefix
    assert key1.startswith("sentinella:cache:")


@pytest.mark.asyncio
async def test_cache_operations():
    """Test cache get/set operations"""
    cache = RedisCache()
    
    # Note: This test requires Redis to be running
    # In CI/CD, use a test Redis instance or mock
    
    # Test without connection (should not crash)
    result = await cache.get("test_key")
    assert result is None
    
    # Test stats
    stats = await cache.get_stats()
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats

