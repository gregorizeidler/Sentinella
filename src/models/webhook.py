"""Webhook models"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class WebhookEventType(str, Enum):
    """Webhook event types"""
    COMPLETION = "completion"
    ERROR = "error"
    COST_THRESHOLD = "cost_threshold"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MODEL_CHANGED = "model_changed"


class WebhookPayload(BaseModel):
    """Webhook payload structure"""
    event_type: WebhookEventType
    tenant_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)


class WebhookDelivery(BaseModel):
    """Webhook delivery tracking"""
    id: str
    tenant_id: str
    event_type: WebhookEventType
    payload: Dict[str, Any]
    url: str
    status: str  # pending, delivered, failed
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

