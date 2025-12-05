"""Tests for smart router"""

import pytest
from src.router.router import SmartRouter


def test_simple_query_routing():
    """Test that simple queries route to cost-effective models"""
    router = SmartRouter()
    messages = [{"role": "user", "content": "Hello"}]
    
    model = router.select_model(messages)
    assert model in ["gpt-4o-mini", "gpt-3.5-turbo"]


def test_complex_reasoning_routing():
    """Test that complex reasoning queries route to premium models"""
    router = SmartRouter()
    messages = [{"role": "user", "content": "Think step by step and explain the reasoning behind quantum computing"}]
    
    model = router.select_model(messages)
    assert model in ["gpt-4", "claude-3-5-sonnet"]


def test_code_routing():
    """Test that code-related queries route to GPT-4"""
    router = SmartRouter()
    messages = [{"role": "user", "content": "Write a Python function to sort a list"}]
    
    model = router.select_model(messages)
    assert model == "gpt-4"


def test_complexity_analysis():
    """Test complexity analysis"""
    router = SmartRouter()
    
    simple_prompt = "Hello"
    complex_prompt = "Analyze the architectural design patterns used in microservices and explain the trade-offs"
    
    simple_score = router._analyze_complexity(simple_prompt)
    complex_score = router._analyze_complexity(complex_prompt)
    
    assert complex_score > simple_score
    assert 0.0 <= simple_score <= 1.0
    assert 0.0 <= complex_score <= 1.0


def test_get_available_models():
    """Test getting available models"""
    router = SmartRouter()
    models = router.get_available_models()
    
    assert len(models) > 0
    assert all("id" in model for model in models)
    assert all("owned_by" in model for model in models)

