"""Sentinella Python SDK Client"""

import httpx
from typing import Optional, List, Dict, Any, AsyncIterator
import json


class SentinellaClient:
    """Python SDK for Sentinella AI Gateway"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send chat completion request
        
        Args:
            messages: List of messages with 'role' and 'content'
            model: Model to use (optional, auto-selected if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            session_id: Session ID for conversational memory
        
        Returns:
            Completion response
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
        }
        
        if model:
            payload["model"] = model
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stream:
            payload["stream"] = True
        if session_id:
            payload["session_id"] = session_id
        
        response = await self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion
        
        Yields content chunks
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        
        if model:
            payload["model"] = model
        
        async with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            content = chunk["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        response = await self.client.get("/v1/models")
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics"""
        response = await self.client.get("/metrics")
        response.raise_for_status()
        return response.json()
    
    async def render_template(
        self,
        template_id: str,
        variables: Dict[str, Any],
    ) -> str:
        """Render a prompt template"""
        response = await self.client.post(
            f"/v1/templates/{template_id}/render",
            json={"variables": variables},
        )
        response.raise_for_status()
        data = response.json()
        return data["rendered"]
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

