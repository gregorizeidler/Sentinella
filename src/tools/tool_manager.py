"""Tool and function calling manager"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field


class ToolFunction(BaseModel):
    """Tool function definition"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    handler: Optional[Callable] = None


class ToolManager:
    """Manage tools and function calling"""
    
    def __init__(self):
        self.tools: Dict[str, ToolFunction] = {}
    
    def register_tool(self, tool: ToolFunction):
        """Register a tool"""
        self.tools[tool.name] = tool
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tools formatted for LLM function calling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in self.tools.values()
        ]
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Execute a tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool = self.tools[tool_name]
        if tool.handler:
            if asyncio.iscoroutinefunction(tool.handler):
                return await tool.handler(**arguments)
            else:
                return tool.handler(**arguments)
        else:
            raise ValueError(f"Tool {tool_name} has no handler")
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a tool"""
        if tool_name not in self.tools:
            return None
        
        tool = self.tools[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

