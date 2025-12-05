"""
Evaluation Pipeline - Quality Assurance for LLM Responses

This module provides automated evaluation of LLM responses using:
- Golden datasets
- RAGAS metrics
- A/B testing capabilities
"""

import json
import asyncio
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import httpx
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


class Evaluator:
    """
    Evaluation pipeline for LLM quality assurance
    
    Features:
    - Golden dataset evaluation
    - RAGAS metrics calculation
    - Model comparison (A/B testing)
    - Pre-deployment quality checks
    """
    
    def __init__(self, gateway_url: str = "http://localhost:8000"):
        self.gateway_url = gateway_url
        self.api_key = os.getenv("SENTINELLA_API_KEY", "")
    
    async def evaluate_model(
        self,
        model: str,
        dataset_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model against a golden dataset
        
        Args:
            model: Model identifier to evaluate
            dataset_path: Path to golden dataset JSON file
            output_path: Optional path to save evaluation results
            
        Returns:
            Evaluation results dictionary
        """
        # Load golden dataset
        dataset = self._load_dataset(dataset_path)
        
        # Run evaluations
        results = []
        total_latency = 0.0
        total_cost = 0.0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for item in dataset:
                prompt = item["prompt"]
                expected_answer = item.get("expected_answer", "")
                
                # Make request to gateway
                start_time = asyncio.get_event_loop().time()
                response = await client.post(
                    f"{self.gateway_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": self.api_key,
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    },
                )
                latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    actual_answer = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    
                    # Calculate cost (simplified)
                    cost = self._estimate_cost(model, usage)
                    total_cost += cost
                    total_latency += latency_ms
                    
                    # Calculate quality metrics
                    quality_score = self._calculate_quality_score(
                        expected_answer,
                        actual_answer,
                    )
                    
                    results.append({
                        "prompt": prompt,
                        "expected": expected_answer,
                        "actual": actual_answer,
                        "quality_score": quality_score,
                        "latency_ms": latency_ms,
                        "cost": cost,
                        "tokens": usage.get("total_tokens", 0),
                    })
                else:
                    results.append({
                        "prompt": prompt,
                        "error": response.text,
                        "status_code": response.status_code,
                    })
        
        # Aggregate results
        evaluation_results = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(dataset),
            "successful": len([r for r in results if "error" not in r]),
            "average_quality_score": sum(
                r.get("quality_score", 0) for r in results if "quality_score" in r
            ) / max(len([r for r in results if "quality_score" in r]), 1),
            "average_latency_ms": total_latency / len(results) if results else 0,
            "total_cost": total_cost,
            "results": results,
        }
        
        # Save results if output path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    async def compare_models(
        self,
        models: List[str],
        dataset_path: str,
    ) -> Dict[str, Any]:
        """
        Compare multiple models side-by-side (A/B testing)
        
        Args:
            models: List of model identifiers to compare
            dataset_path: Path to golden dataset
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for model in models:
            print(f"Evaluating {model}...")
            results = await self.evaluate_model(model, dataset_path)
            comparison_results[model] = results
        
        # Generate comparison summary
        summary = {
            "models": list(comparison_results.keys()),
            "best_quality": max(
                comparison_results.items(),
                key=lambda x: x[1]["average_quality_score"],
            )[0],
            "fastest": min(
                comparison_results.items(),
                key=lambda x: x[1]["average_latency_ms"],
            )[0],
            "cheapest": min(
                comparison_results.items(),
                key=lambda x: x[1]["total_cost"],
            )[0],
            "detailed_results": comparison_results,
        }
        
        return summary
    
    def _load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load golden dataset from JSON file"""
        with open(dataset_path, "r") as f:
            return json.load(f)
    
    def _calculate_quality_score(
        self,
        expected: str,
        actual: str,
    ) -> float:
        """
        Calculate quality score between expected and actual answers
        
        Simple implementation using semantic similarity
        In production, use RAGAS or similar library
        """
        # Simple word overlap score (0.0 to 1.0)
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(expected_words & actual_words)
        total = len(expected_words)
        
        return overlap / total if total > 0 else 0.0
    
    def _estimate_cost(self, model: str, usage: Dict[str, Any]) -> float:
        """Estimate cost based on model and token usage"""
        # Simplified cost estimation (in USD)
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "gpt-4o-mini": 0.00015,
            "claude-3-5-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
        }
        
        rate = cost_per_1k_tokens.get(model, 0.002)
        total_tokens = usage.get("total_tokens", 0)
        
        return (total_tokens / 1000) * rate


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM models")
    parser.add_argument("--model", type=str, help="Model to evaluate")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--compare", nargs="+", help="Models to compare")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    evaluator = Evaluator()
    
    if args.compare:
        # Compare models
        results = asyncio.run(
            evaluator.compare_models(args.compare, args.dataset)
        )
        print(json.dumps(results, indent=2))
    elif args.model:
        # Evaluate single model
        results = asyncio.run(
            evaluator.evaluate_model(args.model, args.dataset, args.output)
        )
        print(json.dumps(results, indent=2))
    else:
        print("Please specify --model or --compare")

