"""Tenant model for multi-tenancy support"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Tenant(BaseModel):
    """Tenant model"""
    id: str
    name: str
    api_key: str
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    
    # Rate limiting config
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000
    
    # Cost limits
    daily_cost_limit: Optional[float] = None
    monthly_cost_limit: Optional[float] = None
    
    # Routing preferences
    default_model: Optional[str] = None
    allowed_models: list[str] = Field(default_factory=list)
    routing_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Webhook config
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TenantUsage(BaseModel):
    """Tenant usage tracking"""
    tenant_id: str
    date: datetime
    request_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    model_usage: Dict[str, int] = Field(default_factory=dict)
    error_count: int = 0

