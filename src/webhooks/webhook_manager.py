"""Webhook delivery manager"""

import asyncio
import hmac
import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime
import httpx

from src.models.webhook import WebhookEventType, WebhookPayload, WebhookDelivery
from src.models.tenant import Tenant


class WebhookManager:
    """Manage webhook deliveries"""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delays = [1, 5, 30]  # seconds
    
    async def send_webhook(
        self,
        tenant: Tenant,
        event_type: WebhookEventType,
        data: Dict[str, Any],
    ):
        """Send webhook to tenant's webhook URL"""
        if not tenant.webhook_url:
            return
        
        payload = WebhookPayload(
            event_type=event_type,
            tenant_id=tenant.id,
            data=data,
        )
        
        # Sign payload
        signature = self._sign_payload(payload.model_dump_json(), tenant.webhook_secret)
        
        # Send with retries
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        tenant.webhook_url,
                        json=payload.model_dump(),
                        headers={
                            "X-Sentinella-Signature": signature,
                            "X-Sentinella-Event": event_type.value,
                        },
                    )
                    response.raise_for_status()
                    return  # Success
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delays[attempt])
                else:
                    # Log failure
                    print(f"Webhook delivery failed after {self.max_retries} attempts: {e}")
    
    def _sign_payload(self, payload: str, secret: Optional[str]) -> str:
        """Sign webhook payload"""
        if not secret:
            return ""
        
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        
        return f"sha256={signature}"

