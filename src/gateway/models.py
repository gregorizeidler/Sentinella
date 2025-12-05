"""Pydantic models for API requests and responses"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""
    model: Optional[str] = Field(None, description="Model identifier (optional)")
    messages: list[ChatMessage] = Field(..., min_items=1)
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = Field(False)


class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""
    id: str
    model: str
    choices: list[Choice]
    usage: Usage
    latency_ms: float
    cached: bool

