"""Prompt template model"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Prompt template with versioning"""
    id: str
    tenant_id: str
    name: str
    description: Optional[str] = None
    template: str  # Template string with {variables}
    variables: List[str] = Field(default_factory=list)  # List of variable names
    version: int = 1
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Optimization hints
    suggested_model: Optional[str] = None
    estimated_tokens: Optional[int] = None


class PromptVersion(BaseModel):
    """Version history for prompt templates"""
    template_id: str
    version: int
    template: str
    variables: List[str]
    created_at: datetime
    created_by: Optional[str] = None

