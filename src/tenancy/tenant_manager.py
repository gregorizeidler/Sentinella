"""Tenant management and storage"""

import os
import json
from typing import Optional, Dict, Any
from datetime import datetime
import redis.asyncio as redis

from src.models.tenant import Tenant, TenantUsage


class TenantManager:
    """Manages tenants and their configurations"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.tenants_cache: Dict[str, Tenant] = {}
    
    async def connect(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Connect to Redis"""
        self.redis_client = await redis.from_url(
            f"redis://{redis_host}:{redis_port}/1",
            decode_responses=True,
        )
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        """Get tenant by API key"""
        # Check cache first
        for tenant in self.tenants_cache.values():
            if tenant.api_key == api_key:
                return tenant
        
        # Check Redis
        if self.redis_client:
            tenant_data = await self.redis_client.get(f"tenant:api_key:{api_key}")
            if tenant_data:
                tenant_dict = json.loads(tenant_data)
                tenant = Tenant(**tenant_dict)
                self.tenants_cache[tenant.id] = tenant
                return tenant
        
        return None
    
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        # Check cache
        if tenant_id in self.tenants_cache:
            return self.tenants_cache[tenant_id]
        
        # Check Redis
        if self.redis_client:
            tenant_data = await self.redis_client.get(f"tenant:{tenant_id}")
            if tenant_data:
                tenant_dict = json.loads(tenant_data)
                tenant = Tenant(**tenant_dict)
                self.tenants_cache[tenant_id] = tenant
                return tenant
        
        return None
    
    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create a new tenant"""
        if self.redis_client:
            tenant_dict = tenant.model_dump()
            tenant_dict["created_at"] = tenant.created_at.isoformat()
            await self.redis_client.set(
                f"tenant:{tenant.id}",
                json.dumps(tenant_dict),
            )
            await self.redis_client.set(
                f"tenant:api_key:{tenant.api_key}",
                json.dumps(tenant_dict),
            )
        
        self.tenants_cache[tenant.id] = tenant
        return tenant
    
    async def update_tenant(self, tenant: Tenant) -> Tenant:
        """Update tenant"""
        if self.redis_client:
            tenant_dict = tenant.model_dump()
            tenant_dict["created_at"] = tenant.created_at.isoformat()
            await self.redis_client.set(
                f"tenant:{tenant.id}",
                json.dumps(tenant_dict),
            )
            await self.redis_client.set(
                f"tenant:api_key:{tenant.api_key}",
                json.dumps(tenant_dict),
            )
        
        self.tenants_cache[tenant.id] = tenant
        return tenant
    
    async def track_usage(
        self,
        tenant_id: str,
        tokens: int,
        cost: float,
        model: str,
    ):
        """Track tenant usage"""
        if not self.redis_client:
            return
        
        today = datetime.now().date().isoformat()
        key = f"usage:{tenant_id}:{today}"
        
        # Increment counters
        await self.redis_client.hincrby(key, "request_count", 1)
        await self.redis_client.hincrby(key, "total_tokens", tokens)
        await self.redis_client.hincrbyfloat(key, "total_cost", cost)
        await self.redis_client.hincrby(key, f"model:{model}", 1)
        
        # Set expiration (keep for 90 days)
        await self.redis_client.expire(key, 90 * 24 * 3600)
    
    async def get_usage(self, tenant_id: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Get tenant usage for a date"""
        if not self.redis_client:
            return {}
        
        if not date:
            date = datetime.now().date().isoformat()
        
        key = f"usage:{tenant_id}:{date}"
        data = await self.redis_client.hgetall(key)
        
        return {
            "request_count": int(data.get("request_count", 0)),
            "total_tokens": int(data.get("total_tokens", 0)),
            "total_cost": float(data.get("total_cost", 0.0)),
        }

