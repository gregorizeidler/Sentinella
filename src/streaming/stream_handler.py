"""Streaming response handler"""

from typing import AsyncGenerator, Dict, Any
from fastapi.responses import StreamingResponse
import json
import litellm


class StreamHandler:
    """Handle streaming responses"""
    
    @staticmethod
    async def stream_completion(
        model: str,
        messages: list[Dict[str, Any]],
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM completion
        
        Yields SSE-formatted chunks
        """
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                **kwargs,
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # Format as SSE
                    data = {
                        "id": chunk.id if hasattr(chunk, 'id') else "",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {"content": content},
                            "index": 0,
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            
            # Send final chunk
            final_data = {
                "id": "",
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "stream_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"

