"""
Sentinella AI Gateway - Main FastAPI Application

This module provides a unified API gateway for LLM providers with intelligent
routing, caching, multi-tenancy, streaming, and comprehensive observability.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

import litellm
from src.cache.redis_client import RedisCache
from src.cache.semantic_cache import SemanticCache
from src.observability.tracer import LangFuseTracer
from src.router.router import SmartRouter
from src.router.fallback import FallbackChain
from src.tenancy.tenant_manager import TenantManager
from src.rate_limiter.limiter import RateLimiter
from src.streaming.stream_handler import StreamHandler
from fastapi.responses import StreamingResponse
from src.memory.memory_manager import MemoryManager
from src.webhooks.webhook_manager import WebhookManager
from src.models.webhook import WebhookEventType
from src.prompts.template_manager import TemplateManager
from src.tools.tool_manager import ToolManager
from src.finetuning.finetune_manager import FineTuneManager
from src.connection_pool.pool_manager import ConnectionPoolManager

# Initialize LiteLLM
litellm.set_verbose = os.getenv("LOG_LEVEL", "INFO") == "DEBUG"

# Initialize components
cache = RedisCache()
semantic_cache = SemanticCache()
tracer = LangFuseTracer()
router = SmartRouter()
fallback = FallbackChain()
tenant_manager = TenantManager()
rate_limiter = RateLimiter()
memory_manager = MemoryManager()
webhook_manager = WebhookManager()
template_manager = TemplateManager()
tool_manager = ToolManager()
finetune_manager = FineTuneManager()
connection_pool = ConnectionPoolManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    await cache.connect()
    await semantic_cache.initialize()
    await tracer.initialize()
    await tenant_manager.connect()
    await rate_limiter.connect()
    await memory_manager.connect()
    await template_manager.connect()
    yield
    # Shutdown
    await cache.disconnect()
    await tenant_manager.disconnect()
    await rate_limiter.disconnect()
    await memory_manager.disconnect()
    await template_manager.disconnect()
    await connection_pool.close_all()


app = FastAPI(
    title="Sentinella AI Gateway",
    description="Enterprise AI Gateway with intelligent routing and evaluation",
    version="0.1.0",
    lifespan=lifespan,
)

# Include dashboard routes
from src.dashboard.routes import router as dashboard_router
app.include_router(dashboard_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(None, description="Model to use (optional, will be auto-selected)")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = Field(False, description="Stream responses")
    session_id: Optional[str] = Field(None, description="Session ID for conversational memory")
    template_id: Optional[str] = Field(None, description="Prompt template ID")
    template_variables: Optional[Dict[str, Any]] = Field(None, description="Template variables")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Function calling tools")
    max_latency_ms: Optional[float] = Field(None, description="Maximum acceptable latency")
    max_cost_per_1k: Optional[float] = Field(None, description="Maximum cost per 1k tokens")


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    choices: list[dict]
    usage: dict
    latency_ms: float
    cached: bool


async def get_tenant_from_api_key(api_key: Optional[str]) -> Optional[Any]:
    """Get tenant from API key"""
    if not api_key:
        return None
    
    # Try to get tenant by API key
    tenant = await tenant_manager.get_tenant_by_api_key(api_key)
    
    # Fallback to simple API key check if multi-tenancy not configured
    if not tenant:
        expected_key = os.getenv("SENTINELLA_API_KEY")
        if api_key == expected_key:
            # Create default tenant
            from src.models.tenant import Tenant
            from datetime import datetime
            tenant = Tenant(
                id="default",
                name="Default Tenant",
                api_key=api_key,
                created_at=datetime.now(),
            )
            await tenant_manager.create_tenant(tenant)
    
    return tenant


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    cache_status = await cache.health_check()
    return {
        "status": "healthy",
        "cache": "connected" if cache_status else "disconnected",
        "version": "0.1.0",
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Header(None, alias="X-API-Key"),
):
    """
    Unified chat completions endpoint with full feature support
    
    Features:
    - Multi-tenancy with rate limiting
    - Intelligent ML-based routing
    - Semantic caching
    - Streaming support
    - Conversational memory
    - Prompt templates
    - Function calling
    - Webhooks
    """
    start_time = time.time()
    
    # Get tenant
    tenant = await get_tenant_from_api_key(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not tenant.is_active:
        raise HTTPException(status_code=403, detail="Tenant is inactive")
    
    # Check rate limits
    allowed, error = await rate_limiter.check_rate_limit(tenant, "minute")
    if not allowed:
        raise HTTPException(status_code=429, detail=error)
    
    # Prepare messages
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Handle prompt templates
    if request.template_id:
        rendered = await template_manager.render_template(
            tenant.id,
            request.template_id,
            request.template_variables or {},
        )
        if rendered:
            messages_dict = [{"role": "user", "content": rendered}]
    
    # Handle conversational memory
    if request.session_id:
        # Get conversation history
        history = await memory_manager.get_messages_for_llm(
            tenant.id,
            request.session_id,
        )
        # Prepend history to current messages
        messages_dict = history + messages_dict
    
    # Extract prompt text for caching
    prompt_text = " ".join([msg.get("content", "") for msg in messages_dict])
    
    # Determine model
    selected_model = request.model
    if not selected_model:
        # Check for fine-tuned models first
        finetuned_models = finetune_manager.get_available_finetuned_models(tenant.id)
        if finetuned_models and tenant.default_model:
            # Use fine-tuned model if available
            selected_model = finetune_manager.format_model_name(finetuned_models[0])
        else:
            # Use smart routing
            selected_model = router.select_model(
                messages_dict,
                max_latency_ms=request.max_latency_ms,
                max_cost_per_1k=request.max_cost_per_1k,
            )
    
    # Check semantic cache first
    semantic_cached = await semantic_cache.get_similar(
        prompt_text,
        selected_model,
        request.temperature,
    )
    
    if semantic_cached:
        # Track usage
        await tenant_manager.track_usage(
            tenant.id,
            semantic_cached.get("usage", {}).get("total_tokens", 0),
            0.0,  # Cached, no cost
            selected_model,
        )
        
        tracer.trace_completion(
            messages=messages_dict,
            model=selected_model,
            cached=True,
            latency_ms=(time.time() - start_time) * 1000,
        )
        
        return ChatCompletionResponse(
            id=semantic_cached.get("id", "cached"),
            model=selected_model,
            choices=semantic_cached.get("choices", []),
            usage=semantic_cached.get("usage", {}),
            latency_ms=(time.time() - start_time) * 1000,
            cached=True,
        )
    
    # Check regular cache
    cache_key = cache.generate_key(
        model=selected_model,
        messages=messages_dict,
        temperature=request.temperature,
    )
    cached_response = await cache.get(cache_key)
    
    if cached_response:
        await tenant_manager.track_usage(
            tenant.id,
            cached_response.get("usage", {}).get("total_tokens", 0),
            0.0,
            selected_model,
        )
        
        tracer.trace_completion(
            messages=messages_dict,
            model=selected_model,
            cached=True,
            latency_ms=(time.time() - start_time) * 1000,
        )
        
        return ChatCompletionResponse(
            id=cached_response.get("id", "cached"),
            model=selected_model,
            choices=cached_response.get("choices", []),
            usage=cached_response.get("usage", {}),
            latency_ms=(time.time() - start_time) * 1000,
            cached=True,
        )
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            StreamHandler.stream_completion(
                model=selected_model,
                messages=messages_dict,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
            media_type="text/event-stream",
        )
    
    # Prepare LiteLLM parameters
    litellm_params = {
        "model": selected_model,
        "messages": messages_dict,
        "temperature": request.temperature,
    }
    if request.max_tokens:
        litellm_params["max_tokens"] = request.max_tokens
    
    # Add tools if provided
    if request.tools:
        litellm_params["tools"] = request.tools
    
    # Start trace
    trace = tracer.start_trace(
        model=selected_model,
        messages=messages_dict,
        user_id=tenant.id,
    )
    
    # Attempt request with fallback
    response = None
    final_model = selected_model
    error_occurred = False
    
    try:
        response = await fallback.execute_with_fallback(
            litellm_params=litellm_params,
            tracer=tracer,
            trace_id=trace.id if trace else None,
        )
        final_model = response.get("model", selected_model)
        
        # Update router latency tracking
        latency_ms = (time.time() - start_time) * 1000
        router.update_latency(final_model, latency_ms)
        router.update_success(final_model, True)
        
    except Exception as e:
        error_occurred = True
        router.update_success(final_model, False)
        
        tracer.trace_error(
            trace_id=trace.id if trace else None,
            error=str(e),
            model=selected_model,
        )
        
        # Send error webhook
        await webhook_manager.send_webhook(
            tenant,
            WebhookEventType.ERROR,
            {"error": str(e), "model": selected_model},
        )
        
        raise HTTPException(status_code=500, detail=f"LLM request failed: {str(e)}")
    
    # Calculate cost (simplified)
    usage = response.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    model_config = router.models.get(final_model)
    cost = 0.0
    if model_config:
        cost = (total_tokens / 1000) * model_config.cost_per_1k_tokens
    
    # Check cost limit
    allowed, error = await rate_limiter.check_cost_limit(tenant, cost)
    if not allowed:
        await webhook_manager.send_webhook(
            tenant,
            WebhookEventType.COST_THRESHOLD,
            {"cost": cost, "limit": tenant.daily_cost_limit},
        )
        raise HTTPException(status_code=429, detail=error)
    
    # Track usage
    await tenant_manager.track_usage(
        tenant.id,
        total_tokens,
        cost,
        final_model,
    )
    
    # Cache successful response
    if response:
        await cache.set(cache_key, response, ttl=3600)
        await semantic_cache.store(
            prompt_text,
            response,
            final_model,
            request.temperature,
        )
    
    # Save to conversational memory
    if request.session_id:
        await memory_manager.add_message(
            tenant.id,
            request.session_id,
            "user",
            prompt_text,
        )
        if response.get("choices"):
            await memory_manager.add_message(
                tenant.id,
                request.session_id,
                "assistant",
                response["choices"][0].get("message", {}).get("content", ""),
            )
    
    # Complete trace
    latency_ms = (time.time() - start_time) * 1000
    tracer.trace_completion(
        messages=messages_dict,
        model=final_model,
        response=response,
        latency_ms=latency_ms,
        trace_id=trace.id if trace else None,
    )
    
    # Send completion webhook
    if not error_occurred:
        await webhook_manager.send_webhook(
            tenant,
            WebhookEventType.COMPLETION,
            {
                "model": final_model,
                "tokens": total_tokens,
                "cost": cost,
                "latency_ms": latency_ms,
            },
        )
    
    return ChatCompletionResponse(
        id=response.get("id", "unknown"),
        model=final_model,
        choices=response.get("choices", []),
        usage=usage,
        latency_ms=latency_ms,
        cached=False,
    )


@app.get("/v1/models")
async def list_models(api_key: str = Header(None, alias="X-API-Key")):
    """List available models"""
    tenant = await get_tenant_from_api_key(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    models = router.get_available_models()
    
    # Add fine-tuned models if tenant has any
    finetuned = finetune_manager.get_available_finetuned_models(tenant.id)
    for ft_model in finetuned:
        models.append({
            "id": finetune_manager.format_model_name(ft_model),
            "object": "model",
            "created": 1677610602,
            "owned_by": f"{ft_model.provider}-finetuned",
            "fine_tuned": True,
        })
    
    return {
        "data": models,
        "object": "list",
    }


@app.get("/metrics")
async def metrics(api_key: str = Header(None, alias="X-API-Key")):
    """Get gateway metrics"""
    tenant = await get_tenant_from_api_key(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    cache_stats = await cache.get_stats()
    usage = await tenant_manager.get_usage(tenant.id)
    
    return {
        "cache": cache_stats,
        "router": router.get_stats(),
        "tenant_usage": usage,
    }


# Template endpoints
@app.post("/v1/templates")
async def create_template(
    name: str,
    template: str,
    variables: Optional[List[str]] = None,
    api_key: str = Header(None, alias="X-API-Key"),
):
    """Create a prompt template"""
    tenant = await get_tenant_from_api_key(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    from src.models.prompt_template import PromptTemplate
    from datetime import datetime
    import uuid
    
    prompt_template = PromptTemplate(
        id=str(uuid.uuid4()),
        tenant_id=tenant.id,
        name=name,
        template=template,
        variables=variables or [],
    )
    
    created = await template_manager.create_template(prompt_template)
    return {"id": created.id, "name": created.name}


@app.get("/v1/templates/{template_id}")
async def get_template(
    template_id: str,
    api_key: str = Header(None, alias="X-API-Key"),
):
    """Get a template"""
    tenant = await get_tenant_from_api_key(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    template = await template_manager.get_template(tenant.id, template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return template.model_dump()


@app.post("/v1/templates/{template_id}/render")
async def render_template(
    template_id: str,
    variables: Dict[str, Any],
    api_key: str = Header(None, alias="X-API-Key"),
):
    """Render a template"""
    tenant = await get_tenant_from_api_key(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    rendered = await template_manager.render_template(
        tenant.id,
        template_id,
        variables,
    )
    
    if not rendered:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"rendered": rendered}


# Conversation endpoints
@app.get("/v1/conversations/{session_id}")
async def get_conversation(
    session_id: str,
    api_key: str = Header(None, alias="X-API-Key"),
):
    """Get conversation by session ID"""
    tenant = await get_tenant_from_api_key(api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    conversation = await memory_manager.get_conversation(tenant.id, session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation.model_dump()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.gateway.main:app",
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8000")),
        reload=True,
    )
