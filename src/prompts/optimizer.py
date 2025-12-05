"""Prompt optimization utilities"""

from typing import List, Dict, Any
import re


class PromptOptimizer:
    """Optimize prompts for better results and cost efficiency"""
    
    @staticmethod
    def analyze_prompt(prompt: str) -> Dict[str, Any]:
        """Analyze prompt and provide optimization suggestions"""
        suggestions = []
        
        # Check length
        if len(prompt) > 2000:
            suggestions.append({
                "type": "length",
                "message": "Prompt is very long. Consider summarizing or breaking into parts.",
                "impact": "high",
            })
        
        # Check for redundant phrases
        redundant_phrases = [
            "please please",
            "very very",
            "really really",
            "I want you to I want you to",
        ]
        for phrase in redundant_phrases:
            if phrase.lower() in prompt.lower():
                suggestions.append({
                    "type": "redundancy",
                    "message": f"Redundant phrase detected: '{phrase}'",
                    "impact": "medium",
                })
        
        # Check for vague instructions
        vague_words = ["good", "better", "nice", "some", "things"]
        vague_count = sum(1 for word in vague_words if word.lower() in prompt.lower())
        if vague_count > 3:
            suggestions.append({
                "type": "vagueness",
                "message": "Prompt contains vague terms. Be more specific.",
                "impact": "high",
            })
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(prompt) // 4
        
        return {
            "length": len(prompt),
            "estimated_tokens": estimated_tokens,
            "suggestions": suggestions,
        }
    
    @staticmethod
    def optimize_prompt(prompt: str) -> str:
        """Apply basic optimizations to prompt"""
        optimized = prompt
        
        # Remove extra whitespace
        optimized = re.sub(r'\s+', ' ', optimized)
        optimized = optimized.strip()
        
        # Remove redundant phrases
        redundant = [
            (r'\bplease please\b', 'please'),
            (r'\bvery very\b', 'very'),
            (r'\breally really\b', 'really'),
        ]
        for pattern, replacement in redundant:
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        return optimized
    
    @staticmethod
    def suggest_improvements(prompt: str) -> List[str]:
        """Suggest specific improvements"""
        improvements = []
        
        # Check for clear instructions
        if not any(word in prompt.lower() for word in ["write", "create", "generate", "explain", "analyze"]):
            improvements.append("Add a clear action verb (write, create, explain, etc.)")
        
        # Check for context
        if len(prompt.split('.')) < 2:
            improvements.append("Add more context to help the model understand the task")
        
        # Check for examples
        if "example" not in prompt.lower() and "for example" not in prompt.lower():
            improvements.append("Consider adding examples to improve output quality")
        
        return improvements

