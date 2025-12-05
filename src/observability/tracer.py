"""
LangFuse Tracer - Observability and Evaluation Integration

This module integrates LangFuse for comprehensive observability:
- Request tracing
- Performance metrics
- Cost tracking
- Evaluation integration
"""

import os
from typing import Optional, List, Dict, Any
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from langfuse.api.resources.commons.types.observation_level import ObservationLevel


class LangFuseTracer:
    """
    LangFuse tracer for observability
    
    Features:
    - Automatic request tracing
    - Error tracking
    - Performance metrics
    - Cost calculation
    """
    
    def __init__(self):
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        self.client: Optional[Langfuse] = None
        self.enabled = bool(self.secret_key and self.public_key)
    
    async def initialize(self):
        """Initialize LangFuse client"""
        if not self.enabled:
            print("LangFuse not configured. Observability disabled.")
            return
        
        try:
            self.client = Langfuse(
                secret_key=self.secret_key,
                public_key=self.public_key,
                host=self.host,
            )
        except Exception as e:
            print(f"LangFuse initialization failed: {e}")
            self.enabled = False
    
    def start_trace(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Start a new trace
        
        Returns:
            Trace object or None if LangFuse is disabled
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            trace = self.client.trace(
                name="chat_completion",
                user_id=user_id,
                metadata={
                    "model": model,
                    "message_count": len(messages),
                },
            )
            return trace
        except Exception as e:
            print(f"Failed to start trace: {e}")
            return None
    
    def trace_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response: Optional[Dict[str, Any]] = None,
        cached: bool = False,
        latency_ms: float = 0.0,
        trace_id: Optional[str] = None,
    ):
        """Trace a completion request"""
        if not self.enabled or not self.client:
            return
        
        try:
            # Create or get trace
            if trace_id:
                trace = self.client.trace(id=trace_id)
            else:
                trace = self.client.trace(name="chat_completion")
            
            # Create generation span
            generation = trace.generation(
                name="llm_completion",
                model=model,
                model_parameters={
                    "temperature": 0.7,  # Could be extracted from request
                },
                input=messages,
                output=response.get("choices", []) if response else None,
                metadata={
                    "cached": cached,
                    "latency_ms": latency_ms,
                },
                level=ObservationLevel.DEFAULT,
            )
            
            # Add usage information
            if response and "usage" in response:
                usage = response["usage"]
                generation.usage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
            
            # Flush to LangFuse
            self.client.flush()
        except Exception as e:
            print(f"Failed to trace completion: {e}")
    
    def trace_error(
        self,
        error: str,
        model: str,
        trace_id: Optional[str] = None,
    ):
        """Trace an error"""
        if not self.enabled or not self.client:
            return
        
        try:
            if trace_id:
                trace = self.client.trace(id=trace_id)
            else:
                trace = self.client.trace(name="chat_completion_error")
            
            trace.generation(
                name="llm_error",
                model=model,
                level=ObservationLevel.ERROR,
                metadata={"error": error},
            )
            
            self.client.flush()
        except Exception as e:
            print(f"Failed to trace error: {e}")
    
    def trace_fallback(
        self,
        from_model: str,
        to_model: str,
        trace_id: Optional[str] = None,
    ):
        """Trace a fallback event"""
        if not self.enabled or not self.client:
            return
        
        try:
            if trace_id:
                trace = self.client.trace(id=trace_id)
            else:
                trace = self.client.trace(name="fallback")
            
            trace.event(
                name="model_fallback",
                metadata={
                    "from_model": from_model,
                    "to_model": to_model,
                },
            )
            
            self.client.flush()
        except Exception as e:
            print(f"Failed to trace fallback: {e}")
    
    def trace_success(
        self,
        model: str,
        latency_ms: float,
        trace_id: Optional[str] = None,
    ):
        """Trace a successful request"""
        if not self.enabled or not self.client:
            return
        
        try:
            if trace_id:
                trace = self.client.trace(id=trace_id)
            else:
                trace = self.client.trace(name="success")
            
            trace.event(
                name="request_success",
                metadata={
                    "model": model,
                    "latency_ms": latency_ms,
                },
            )
            
            self.client.flush()
        except Exception as e:
            print(f"Failed to trace success: {e}")

