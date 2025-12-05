"""
Smart Router - Intelligent Model Selection with ML-based routing

This module implements intelligent routing logic that selects the optimal
LLM model based on input characteristics, cost constraints, and quality requirements.
Now includes all major models from GPT, Gemini, Grok, DeepSeek, and more.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    provider: str
    cost_per_1k_tokens: float
    max_tokens: int
    best_for: List[str]
    avg_latency_ms: float = 0.0  # Average latency in milliseconds
    success_rate: float = 1.0  # Success rate (0.0 to 1.0)


class SmartRouter:
    """
    Intelligent router with ML-based routing and latency tracking
    """
    
    def __init__(self):
        self.models = self._initialize_all_models()
        self.stats = {
            "total_requests": 0,
            "model_selections": {},
            "latency_history": {},  # Track latency per model
            "success_history": {},  # Track success rate per model
        }
    
    def _initialize_all_models(self) -> Dict[str, ModelConfig]:
        """Initialize ALL available models from all providers"""
        models = {}
        
        # OpenAI Models
        models.update({
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                provider="openai",
                cost_per_1k_tokens=5.0,
                max_tokens=128000,
                best_for=["general", "reasoning", "code"],
                avg_latency_ms=1200.0,
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                provider="openai",
                cost_per_1k_tokens=0.15,
                max_tokens=128000,
                best_for=["simple", "quick", "cost-effective"],
                avg_latency_ms=800.0,
            ),
            "gpt-4-turbo": ModelConfig(
                name="gpt-4-turbo",
                provider="openai",
                cost_per_1k_tokens=10.0,
                max_tokens=128000,
                best_for=["complex", "reasoning", "long-context"],
                avg_latency_ms=1500.0,
            ),
            "gpt-4": ModelConfig(
                name="gpt-4",
                provider="openai",
                cost_per_1k_tokens=30.0,
                max_tokens=8192,
                best_for=["reasoning", "complex", "code"],
                avg_latency_ms=2000.0,
            ),
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                cost_per_1k_tokens=0.50,
                max_tokens=16384,
                best_for=["simple", "conversational", "cost-effective"],
                avg_latency_ms=600.0,
            ),
            "gpt-4-1106-preview": ModelConfig(
                name="gpt-4-1106-preview",
                provider="openai",
                cost_per_1k_tokens=10.0,
                max_tokens=128000,
                best_for=["long-context", "complex"],
            ),
            "gpt-4-0125-preview": ModelConfig(
                name="gpt-4-0125-preview",
                provider="openai",
                cost_per_1k_tokens=10.0,
                max_tokens=128000,
                best_for=["long-context", "complex"],
            ),
        })
        
        # Anthropic Claude Models
        models.update({
            "claude-3-5-sonnet-20241022": ModelConfig(
                name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                cost_per_1k_tokens=3.0,
                max_tokens=200000,
                best_for=["reasoning", "complex", "balanced"],
                avg_latency_ms=1400.0,
            ),
            "claude-3-opus-20240229": ModelConfig(
                name="claude-3-opus-20240229",
                provider="anthropic",
                cost_per_1k_tokens=15.0,
                max_tokens=200000,
                best_for=["complex", "reasoning", "high_quality"],
                avg_latency_ms=2500.0,
            ),
            "claude-3-sonnet-20240229": ModelConfig(
                name="claude-3-sonnet-20240229",
                provider="anthropic",
                cost_per_1k_tokens=3.0,
                max_tokens=200000,
                best_for=["reasoning", "balanced"],
            ),
            "claude-3-haiku-20240307": ModelConfig(
                name="claude-3-haiku-20240307",
                provider="anthropic",
                cost_per_1k_tokens=0.25,
                max_tokens=200000,
                best_for=["simple", "long_context", "fast"],
                avg_latency_ms=500.0,
            ),
        })
        
        # Google Gemini Models
        models.update({
            "gemini-pro": ModelConfig(
                name="gemini-pro",
                provider="google",
                cost_per_1k_tokens=0.50,
                max_tokens=32768,
                best_for=["general", "multimodal"],
                avg_latency_ms=1000.0,
            ),
            "gemini-pro-vision": ModelConfig(
                name="gemini-pro-vision",
                provider="google",
                cost_per_1k_tokens=0.50,
                max_tokens=16384,
                best_for=["vision", "multimodal"],
            ),
            "gemini-1.5-pro": ModelConfig(
                name="gemini-1.5-pro",
                provider="google",
                cost_per_1k_tokens=1.25,
                max_tokens=2097152,  # 2M tokens!
                best_for=["long-context", "complex", "multimodal"],
                avg_latency_ms=1800.0,
            ),
            "gemini-1.5-flash": ModelConfig(
                name="gemini-1.5-flash",
                provider="google",
                cost_per_1k_tokens=0.075,
                max_tokens=1048576,  # 1M tokens
                best_for=["fast", "long-context", "cost-effective"],
                avg_latency_ms=700.0,
            ),
            "gemini-ultra": ModelConfig(
                name="gemini-ultra",
                provider="google",
                cost_per_1k_tokens=5.0,
                max_tokens=32768,
                best_for=["complex", "high-quality"],
            ),
        })
        
        # Grok Models (xAI)
        models.update({
            "grok-beta": ModelConfig(
                name="grok-beta",
                provider="xai",
                cost_per_1k_tokens=0.50,
                max_tokens=8192,
                best_for=["general", "conversational"],
                avg_latency_ms=900.0,
            ),
            "grok-2": ModelConfig(
                name="grok-2",
                provider="xai",
                cost_per_1k_tokens=1.0,
                max_tokens=131072,
                best_for=["general", "long-context"],
            ),
        })
        
        # DeepSeek Models
        models.update({
            "deepseek-chat": ModelConfig(
                name="deepseek-chat",
                provider="deepseek",
                cost_per_1k_tokens=0.14,
                max_tokens=16384,
                best_for=["code", "cost-effective", "fast"],
                avg_latency_ms=600.0,
            ),
            "deepseek-coder": ModelConfig(
                name="deepseek-coder",
                provider="deepseek",
                cost_per_1k_tokens=0.14,
                max_tokens=16384,
                best_for=["code", "programming"],
                avg_latency_ms=650.0,
            ),
        })
        
        # Cohere Models
        models.update({
            "command": ModelConfig(
                name="command",
                provider="cohere",
                cost_per_1k_tokens=1.0,
                max_tokens=4096,
                best_for=["general", "instruction-following"],
            ),
            "command-light": ModelConfig(
                name="command-light",
                provider="cohere",
                cost_per_1k_tokens=0.50,
                max_tokens=4096,
                best_for=["simple", "fast"],
            ),
        })
        
        # Mistral Models
        models.update({
            "mistral-large": ModelConfig(
                name="mistral-large",
                provider="mistral",
                cost_per_1k_tokens=2.0,
                max_tokens=32000,
                best_for=["general", "reasoning"],
            ),
            "mistral-medium": ModelConfig(
                name="mistral-medium",
                provider="mistral",
                cost_per_1k_tokens=1.0,
                max_tokens=32000,
                best_for=["general", "balanced"],
            ),
            "mistral-small": ModelConfig(
                name="mistral-small",
                provider="mistral",
                cost_per_1k_tokens=0.20,
                max_tokens=32000,
                best_for=["simple", "cost-effective"],
            ),
        })
        
        # Meta Llama Models (via AWS Bedrock)
        models.update({
            "llama-3-70b": ModelConfig(
                name="llama-3-70b",
                provider="bedrock",
                cost_per_1k_tokens=0.65,
                max_tokens=8192,
                best_for=["general", "open-source"],
            ),
            "llama-3-8b": ModelConfig(
                name="llama-3-8b",
                provider="bedrock",
                cost_per_1k_tokens=0.10,
                max_tokens=8192,
                best_for=["simple", "cost-effective"],
            ),
        })
        
        return models
    
    def select_model(
        self,
        messages: List[Dict[str, Any]],
        max_latency_ms: Optional[float] = None,
        max_cost_per_1k: Optional[float] = None,
        preferred_provider: Optional[str] = None,
    ) -> str:
        """
        Select optimal model with ML-based routing and latency constraints
        
        Args:
            messages: Conversation messages
            max_latency_ms: Maximum acceptable latency
            max_cost_per_1k: Maximum cost per 1k tokens
            preferred_provider: Preferred provider name
        """
        self.stats["total_requests"] += 1
        
        # Extract prompt
        prompt = " ".join([msg.get("content", "") for msg in messages])
        prompt_length = len(prompt)
        
        # Analyze complexity
        complexity_score = self._analyze_complexity(prompt)
        task_type = self._detect_task_type(prompt)
        
        # Filter models by constraints
        candidates = self._filter_models(
            max_latency_ms=max_latency_ms,
            max_cost_per_1k=max_cost_per_1k,
            preferred_provider=preferred_provider,
        )
        
        # ML-based scoring
        scored_models = []
        for model_name, model_config in candidates.items():
            score = self._calculate_model_score(
                model_config,
                prompt_length,
                complexity_score,
                task_type,
            )
            scored_models.append((model_name, model_config, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[2], reverse=True)
        
        # Select best model
        selected = scored_models[0][0] if scored_models else "gpt-3.5-turbo"
        
        # Track selection
        self.stats["model_selections"][selected] = (
            self.stats["model_selections"].get(selected, 0) + 1
        )
        
        return selected
    
    def _filter_models(
        self,
        max_latency_ms: Optional[float] = None,
        max_cost_per_1k: Optional[float] = None,
        preferred_provider: Optional[str] = None,
    ) -> Dict[str, ModelConfig]:
        """Filter models by constraints"""
        filtered = {}
        
        for name, config in self.models.items():
            # Provider filter
            if preferred_provider and config.provider != preferred_provider:
                continue
            
            # Latency filter
            if max_latency_ms and config.avg_latency_ms > max_latency_ms:
                continue
            
            # Cost filter
            if max_cost_per_1k and config.cost_per_1k_tokens > max_cost_per_1k:
                continue
            
            filtered[name] = config
        
        return filtered if filtered else self.models
    
    def _calculate_model_score(
        self,
        model: ModelConfig,
        prompt_length: int,
        complexity: float,
        task_type: str,
    ) -> float:
        """
        Calculate ML-based score for model selection
        
        Factors:
        - Cost efficiency
        - Latency
        - Model capabilities (best_for match)
        - Historical performance
        """
        score = 0.0
        
        # Cost efficiency (lower is better, inverted)
        cost_score = 1.0 / (1.0 + model.cost_per_1k_tokens * 0.1)
        score += cost_score * 0.3
        
        # Latency (lower is better, inverted)
        latency_score = 1.0 / (1.0 + model.avg_latency_ms * 0.001)
        score += latency_score * 0.2
        
        # Capability match
        capability_score = 0.0
        if task_type == "code" and "code" in model.best_for:
            capability_score = 1.0
        elif task_type == "reasoning" and "reasoning" in model.best_for:
            capability_score = 1.0
        elif complexity < 0.3 and "simple" in model.best_for:
            capability_score = 1.0
        elif complexity > 0.7 and "complex" in model.best_for:
            capability_score = 1.0
        else:
            capability_score = 0.5
        
        score += capability_score * 0.3
        
        # Historical performance
        historical_score = model.success_rate
        score += historical_score * 0.2
        
        return score
    
    def update_latency(self, model: str, latency_ms: float):
        """Update latency tracking for a model"""
        if model not in self.stats["latency_history"]:
            self.stats["latency_history"][model] = []
        
        history = self.stats["latency_history"][model]
        history.append(latency_ms)
        
        # Keep only last 100 measurements
        if len(history) > 100:
            history.pop(0)
        
        # Update model config with average
        if model in self.models:
            avg_latency = np.mean(history) if history else self.models[model].avg_latency_ms
            self.models[model].avg_latency_ms = avg_latency
    
    def update_success(self, model: str, success: bool):
        """Update success rate tracking"""
        if model not in self.stats["success_history"]:
            self.stats["success_history"][model] = {"success": 0, "total": 0}
        
        stats = self.stats["success_history"][model]
        stats["total"] += 1
        if success:
            stats["success"] += 1
        
        # Update model config
        if model in self.models:
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 1.0
            self.models[model].success_rate = success_rate
    
    def _analyze_complexity(self, prompt: str) -> float:
        """Analyze prompt complexity (0.0 = simple, 1.0 = complex)"""
        score = 0.0
        
        # Length factor
        length_score = min(len(prompt) / 1000, 1.0) * 0.3
        score += length_score
        
        # Technical keywords
        technical_keywords = [
            "algorithm", "optimize", "architecture", "design pattern",
            "analyze", "evaluate", "compare", "explain why",
            "reasoning", "logic", "derive", "prove",
        ]
        keyword_count = sum(1 for kw in technical_keywords if kw.lower() in prompt.lower())
        keyword_score = min(keyword_count / 5, 1.0) * 0.4
        score += keyword_score
        
        # Question complexity
        complex_indicators = [
            "?", "how", "why", "what if", "compare",
            "difference", "explain", "analyze",
        ]
        indicator_count = sum(1 for ind in complex_indicators if ind.lower() in prompt.lower())
        indicator_score = min(indicator_count / 3, 1.0) * 0.3
        score += indicator_score
        
        return min(score, 1.0)
    
    def _detect_task_type(self, prompt: str) -> str:
        """Detect task type"""
        prompt_lower = prompt.lower()
        
        if "code" in prompt_lower or "```" in prompt_lower or "function" in prompt_lower:
            return "code"
        elif "reason" in prompt_lower or "think" in prompt_lower or "step by step" in prompt_lower:
            return "reasoning"
        elif "translate" in prompt_lower or "summarize" in prompt_lower:
            return "transformation"
        else:
            return "general"
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return [
            {
                "id": model.name,
                "object": "model",
                "created": 1677610602,
                "owned_by": model.provider,
                "max_tokens": model.max_tokens,
                "cost_per_1k": model.cost_per_1k_tokens,
                "avg_latency_ms": model.avg_latency_ms,
            }
            for model in self.models.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "total_requests": self.stats["total_requests"],
            "model_distribution": self.stats["model_selections"],
            "model_count": len(self.models),
        }
