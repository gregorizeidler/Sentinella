"""Rate limiting implementation"""

import time
from typing import Optional
import redis.asyncio as redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException

from src.models.tenant import Tenant


class RateLimiter:
    """Rate limiter with Redis backend"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.limiter = Limiter(key_func=get_remote_address)
    
    async def connect(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Connect to Redis"""
        self.redis_client = await redis.from_url(
            f"redis://{redis_host}:{redis_port}/2",
            decode_responses=True,
        )
    
    async def disconnect(self):
        """Disconnect"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def check_rate_limit(
        self,
        tenant: Tenant,
        window: str = "minute",
    ) -> tuple[bool, Optional[str]]:
        """
        Check if tenant has exceeded rate limit
        
        Returns:
            (allowed, error_message)
        """
        if not self.redis_client:
            return True, None
        
        if not tenant.is_active:
            return False, "Tenant is inactive"
        
        # Get limit for window
        if window == "minute":
            limit = tenant.rate_limit_per_minute
            key_suffix = f":{int(time.time() / 60)}"
        elif window == "hour":
            limit = tenant.rate_limit_per_hour
            key_suffix = f":{int(time.time() / 3600)}"
        elif window == "day":
            limit = tenant.rate_limit_per_day
            key_suffix = f":{int(time.time() / 86400)}"
        else:
            return True, None
        
        key = f"rate_limit:{tenant.id}:{window}{key_suffix}"
        
        # Get current count
        current = await self.redis_client.get(key)
        count = int(current) if current else 0
        
        if count >= limit:
            return False, f"Rate limit exceeded: {limit} requests per {window}"
        
        # Increment counter
        await self.redis_client.incr(key)
        await self.redis_client.expire(key, 3600 if window == "minute" else 86400)
        
        return True, None
    
    async def check_cost_limit(
        self,
        tenant: Tenant,
        cost: float,
    ) -> tuple[bool, Optional[str]]:
        """Check if cost would exceed daily limit"""
        if not tenant.daily_cost_limit:
            return True, None
        
        if not self.redis_client:
            return True, None
        
        today = time.strftime("%Y-%m-%d")
        key = f"cost:{tenant.id}:{today}"
        
        current_cost = await self.redis_client.get(key)
        total_cost = float(current_cost) if current_cost else 0.0
        
        if total_cost + cost > tenant.daily_cost_limit:
            return False, f"Daily cost limit exceeded: ${tenant.daily_cost_limit}"
        
        # Increment cost
        await self.redis_client.incrbyfloat(key, cost)
        await self.redis_client.expire(key, 86400)
        
        return True, None

