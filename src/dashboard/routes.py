"""Dashboard API routes"""

import os
from fastapi import APIRouter, HTTPException, Header
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from src.tenancy.tenant_manager import TenantManager
from src.rate_limiter.limiter import RateLimiter
from src.router.router import SmartRouter

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

tenant_manager = TenantManager()
rate_limiter = RateLimiter()
smart_router = SmartRouter()


def verify_admin_key(admin_key: Optional[str] = Header(None, alias="X-Admin-Key")) -> str:
    """Verify admin API key"""
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Admin key not configured")
    if admin_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return admin_key


@router.get("/stats")
async def get_stats(admin_key: str = Header(None, alias="X-Admin-Key")):
    """Get overall gateway statistics"""
    verify_admin_key(admin_key)
    
    # Get router stats
    router_stats = smart_router.get_stats()
    
    return {
        "total_requests": router_stats["total_requests"],
        "model_distribution": router_stats["model_distribution"],
        "available_models": router_stats["model_count"],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/tenants")
async def list_tenants(admin_key: str = Header(None, alias="X-Admin-Key")):
    """List all tenants"""
    verify_admin_key(admin_key)
    
    # Get all tenants from cache (in production, fetch from database)
    tenants = []
    if tenant_manager.tenants_cache:
        for tenant in tenant_manager.tenants_cache.values():
            tenants.append({
                "id": tenant.id,
                "name": tenant.name,
                "is_active": tenant.is_active,
                "created_at": tenant.created_at.isoformat() if hasattr(tenant.created_at, 'isoformat') else str(tenant.created_at),
            })
    
    return {
        "tenants": tenants,
        "total": len(tenants),
    }


@router.get("/tenants/{tenant_id}/usage")
async def get_tenant_usage(
    tenant_id: str,
    days: int = 7,
    admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """Get tenant usage statistics"""
    verify_admin_key(admin_key)
    
    usage = await tenant_manager.get_usage(tenant_id)
    
    return {
        "tenant_id": tenant_id,
        "period_days": days,
        "usage": usage,
    }


@router.get("/models/performance")
async def get_model_performance(admin_key: str = Header(None, alias="X-Admin-Key")):
    """Get model performance metrics"""
    verify_admin_key(admin_key)
    
    router_stats = smart_router.get_stats()
    
    models_list = []
    for model, count in router_stats.get("model_distribution", {}).items():
        model_config = smart_router.models.get(model)
        avg_latency = model_config.avg_latency_ms if model_config else 0
        models_list.append({
            "name": model,
            "request_count": count,
            "avg_latency_ms": avg_latency,
        })
    
    return {
        "models": models_list,
    }


@router.get("/health/detailed")
async def detailed_health(admin_key: str = Header(None, alias="X-Admin-Key")):
    """Detailed health check"""
    verify_admin_key(admin_key)
    
    cache_health = await tenant_manager.redis_client.ping() if tenant_manager.redis_client else False
    
    return {
        "status": "healthy",
        "cache": "connected" if cache_health else "disconnected",
        "router": "operational",
        "timestamp": datetime.now().isoformat(),
    }

