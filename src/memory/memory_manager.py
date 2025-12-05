"""Conversational memory manager"""

from typing import Optional, List, Dict, Any
import redis.asyncio as redis
import json
from datetime import datetime

from src.models.conversation import Conversation, ConversationMessage, ConversationSummary


class MemoryManager:
    """Manage conversational memory"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Connect to Redis"""
        self.redis_client = await redis.from_url(
            f"redis://{redis_host}:{redis_port}/5",
            decode_responses=True,
        )
    
    async def disconnect(self):
        """Disconnect"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get_conversation(
        self,
        tenant_id: str,
        session_id: str,
    ) -> Optional[Conversation]:
        """Get conversation by session ID"""
        if not self.redis_client:
            return None
        
        key = f"conversation:{tenant_id}:{session_id}"
        data = await self.redis_client.get(key)
        
        if data:
            conv_dict = json.loads(data)
            # Convert message dicts to ConversationMessage objects
            conv_dict["messages"] = [
                ConversationMessage(**msg) if isinstance(msg, dict) else msg
                for msg in conv_dict["messages"]
            ]
            return Conversation(**conv_dict)
        
        return None
    
    async def create_conversation(
        self,
        tenant_id: str,
        session_id: str,
    ) -> Conversation:
        """Create a new conversation"""
        conversation = Conversation(
            id=f"{tenant_id}:{session_id}",
            tenant_id=tenant_id,
            session_id=session_id,
        )
        
        await self.save_conversation(conversation)
        return conversation
    
    async def save_conversation(self, conversation: Conversation):
        """Save conversation"""
        if not self.redis_client:
            return
        
        conversation.updated_at = datetime.now()
        key = f"conversation:{conversation.tenant_id}:{conversation.session_id}"
        
        conv_dict = conversation.model_dump()
        conv_dict["created_at"] = conversation.created_at.isoformat()
        conv_dict["updated_at"] = conversation.updated_at.isoformat()
        conv_dict["messages"] = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata,
            }
            for msg in conversation.messages
        ]
        
        await self.redis_client.setex(
            key,
            86400 * 30,  # 30 days TTL
            json.dumps(conv_dict),
        )
    
    async def add_message(
        self,
        tenant_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add message to conversation"""
        conversation = await self.get_conversation(tenant_id, session_id)
        
        if not conversation:
            conversation = await self.create_conversation(tenant_id, session_id)
        
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        
        conversation.messages.append(message)
        
        # Check if summarization is needed
        if len(conversation.messages) > conversation.max_messages:
            await self._summarize_conversation(conversation)
        
        await self.save_conversation(conversation)
    
    async def _summarize_conversation(self, conversation: Conversation):
        """Summarize old messages to save context"""
        # Keep last 10 messages, summarize the rest
        if len(conversation.messages) <= conversation.max_messages:
            return
        
        # Simple summarization: keep first and last messages
        # In production, use LLM to generate summary
        old_messages = conversation.messages[:-10]
        new_messages = conversation.messages[-10:]
        
        # Create summary
        summary_text = f"Previous conversation had {len(old_messages)} messages. "
        summary_text += f"Key topics: {', '.join(set([msg.role for msg in old_messages[:5]]))}"
        
        conversation.summary = summary_text
        conversation.messages = new_messages
    
    async def get_messages_for_llm(
        self,
        tenant_id: str,
        session_id: str,
        max_messages: int = 20,
    ) -> List[Dict[str, str]]:
        """Get formatted messages for LLM"""
        conversation = await self.get_conversation(tenant_id, session_id)
        
        if not conversation:
            return []
        
        # Include summary if exists
        messages = []
        if conversation.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {conversation.summary}",
            })
        
        # Get recent messages
        recent_messages = conversation.messages[-max_messages:]
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        return messages

