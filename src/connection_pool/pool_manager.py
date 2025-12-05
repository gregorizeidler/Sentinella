"""HTTP connection pool manager"""

from typing import Dict, Optional
import httpx
from httpx import AsyncClient, Limits, Timeout


class ConnectionPoolManager:
    """Manage HTTP connection pools for LLM providers"""
    
    def __init__(self):
        self.clients: Dict[str, AsyncClient] = {}
        self.default_limits = Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        self.default_timeout = Timeout(30.0, connect=10.0)
    
    def get_client(self, provider: str) -> AsyncClient:
        """Get or create HTTP client for provider"""
        if provider not in self.clients:
            self.clients[provider] = AsyncClient(
                limits=self.default_limits,
                timeout=self.default_timeout,
                http2=True,  # Enable HTTP/2
            )
        return self.clients[provider]
    
    async def close_all(self):
        """Close all connection pools"""
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

