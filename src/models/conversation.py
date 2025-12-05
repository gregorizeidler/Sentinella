"""Conversation and memory models"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """Single message in a conversation"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """Conversation with memory"""
    id: str
    tenant_id: str
    session_id: str
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Memory management
    summary: Optional[str] = None  # Summarized context for long conversations
    max_messages: int = 50  # Max messages before summarization
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    """Summary of conversation for memory"""
    conversation_id: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

