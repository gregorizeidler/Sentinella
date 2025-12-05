"""
Fallback Chain - Automatic Retry and Fallback Logic

This module implements intelligent fallback strategies when primary models fail
or exceed performance thresholds.
"""

import os
import asyncio
import time
from typing import Dict, Any, Optional, List
import litellm
from litellm import completion


class FallbackChain:
    """
    Implements fallback chain logic:
    1. Retry primary model (transient errors)
    2. Fallback to secondary model (provider issues)
    3. Fallback to tertiary model (last resort)
    """
    
    def __init__(self):
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "30"))
        self.fallback_chain = self._get_fallback_chain()
    
    def _get_fallback_chain(self) -> List[str]:
        """Get fallback chain from environment or use defaults"""
        chain_str = os.getenv("FALLBACK_CHAIN", "gpt-4,claude-3-5-sonnet,gpt-3.5-turbo")
        return [model.strip() for model in chain_str.split(",")]
    
    async def execute_with_fallback(
        self,
        litellm_params: Dict[str, Any],
        tracer: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute LLM request with automatic fallback
        
        Args:
            litellm_params: Parameters for LiteLLM completion
            tracer: LangFuse tracer instance
            trace_id: Optional trace ID for observability
            
        Returns:
            Response dictionary from LiteLLM
            
        Raises:
            Exception: If all fallback attempts fail
        """
        primary_model = litellm_params.get("model")
        
        # Try primary model first
        try:
            response = await self._execute_request(litellm_params, tracer, trace_id)
            return response
        except Exception as primary_error:
            # Log primary failure
            if tracer:
                tracer.trace_error(
                    trace_id=trace_id,
                    error=f"Primary model failed: {str(primary_error)}",
                    model=primary_model,
                )
            
            # Determine fallback models
            fallback_models = self._get_fallback_models(primary_model)
            
            # Try fallback models
            for fallback_model in fallback_models:
                try:
                    fallback_params = litellm_params.copy()
                    fallback_params["model"] = fallback_model
                    
                    if tracer:
                        tracer.trace_fallback(
                            trace_id=trace_id,
                            from_model=primary_model,
                            to_model=fallback_model,
                        )
                    
                    response = await self._execute_request(fallback_params, tracer, trace_id)
                    return response
                except Exception as fallback_error:
                    # Continue to next fallback
                    if tracer:
                        tracer.trace_error(
                            trace_id=trace_id,
                            error=f"Fallback model {fallback_model} failed: {str(fallback_error)}",
                            model=fallback_model,
                        )
                    continue
            
            # All fallbacks exhausted
            raise Exception(
                f"All models failed. Primary: {primary_model}, "
                f"Fallbacks: {fallback_models}. Last error: {str(fallback_error)}"
            )
    
    async def _execute_request(
        self,
        params: Dict[str, Any],
        tracer: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single LLM request with timeout"""
        start_time = time.time()
        
        try:
            # Use asyncio timeout
            response = await asyncio.wait_for(
                self._call_litellm(params),
                timeout=self.timeout_seconds,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Log successful request
            if tracer:
                tracer.trace_success(
                    trace_id=trace_id,
                    model=params["model"],
                    latency_ms=latency_ms,
                )
            
            return response
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {self.timeout_seconds}s")
        except Exception as e:
            raise Exception(f"LLM request failed: {str(e)}")
    
    async def _call_litellm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call LiteLLM completion (wrapped for async)"""
        # LiteLLM completion is sync, so we run it in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: completion(**params),
        )
        return response
    
    def _get_fallback_models(self, primary_model: str) -> List[str]:
        """
        Get fallback models for a given primary model
        
        Strategy: Use models from different providers to avoid provider-wide issues
        """
        # Provider-based fallback mapping
        fallback_map = {
            "gpt-4": ["claude-3-5-sonnet", "gpt-3.5-turbo", "gpt-4o-mini"],
            "gpt-3.5-turbo": ["claude-3-haiku", "gpt-4o-mini"],
            "gpt-4o-mini": ["claude-3-haiku", "gpt-3.5-turbo"],
            "claude-3-5-sonnet": ["gpt-4", "claude-3-haiku", "gpt-3.5-turbo"],
            "claude-3-opus": ["gpt-4", "claude-3-5-sonnet"],
            "claude-3-haiku": ["gpt-4o-mini", "gpt-3.5-turbo"],
        }
        
        # Use configured fallback chain or provider-specific defaults
        if primary_model in fallback_map:
            return fallback_map[primary_model]
        else:
            # Default fallback chain
            return self.fallback_chain

