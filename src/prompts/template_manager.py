"""Prompt template manager"""

import re
from typing import Optional, Dict, Any, List
import redis.asyncio as redis
import json
from datetime import datetime

from src.models.prompt_template import PromptTemplate, PromptVersion


class TemplateManager:
    """Manage prompt templates with versioning"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.templates_cache: Dict[str, PromptTemplate] = {}
    
    async def connect(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Connect to Redis"""
        self.redis_client = await redis.from_url(
            f"redis://{redis_host}:{redis_port}/4",
            decode_responses=True,
        )
    
    async def disconnect(self):
        """Disconnect"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def create_template(self, template: PromptTemplate) -> PromptTemplate:
        """Create a new template"""
        if self.redis_client:
            # Extract variables from template
            variables = self._extract_variables(template.template)
            template.variables = variables
            
            # Store template
            template_dict = template.model_dump()
            template_dict["created_at"] = template.created_at.isoformat()
            template_dict["updated_at"] = template.updated_at.isoformat()
            
            await self.redis_client.set(
                f"template:{template.tenant_id}:{template.id}",
                json.dumps(template_dict),
            )
            
            # Store version history
            version = PromptVersion(
                template_id=template.id,
                version=template.version,
                template=template.template,
                variables=variables,
                created_at=template.created_at,
            )
            await self._store_version(version)
        
        self.templates_cache[f"{template.tenant_id}:{template.id}"] = template
        return template
    
    async def get_template(
        self,
        tenant_id: str,
        template_id: str,
    ) -> Optional[PromptTemplate]:
        """Get template by ID"""
        cache_key = f"{tenant_id}:{template_id}"
        
        # Check cache
        if cache_key in self.templates_cache:
            return self.templates_cache[cache_key]
        
        # Check Redis
        if self.redis_client:
            template_data = await self.redis_client.get(f"template:{cache_key}")
            if template_data:
                template_dict = json.loads(template_data)
                template = PromptTemplate(**template_dict)
                self.templates_cache[cache_key] = template
                return template
        
        return None
    
    async def render_template(
        self,
        tenant_id: str,
        template_id: str,
        variables: Dict[str, Any],
    ) -> Optional[str]:
        """Render template with variables"""
        template = await self.get_template(tenant_id, template_id)
        if not template:
            return None
        
        try:
            rendered = template.template.format(**variables)
            return rendered
        except KeyError as e:
            raise ValueError(f"Missing variable: {e}")
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template string"""
        pattern = r'\{(\w+)\}'
        variables = re.findall(pattern, template)
        return list(set(variables))  # Remove duplicates
    
    async def _store_version(self, version: PromptVersion):
        """Store template version"""
        if self.redis_client:
            version_dict = version.model_dump()
            version_dict["created_at"] = version.created_at.isoformat()
            await self.redis_client.lpush(
                f"template_versions:{version.template_id}",
                json.dumps(version_dict),
            )
            await self.redis_client.ltrim(
                f"template_versions:{version.template_id}",
                0,
                99,  # Keep last 100 versions
            )

