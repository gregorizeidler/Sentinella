"""Fine-tuned model manager"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class FineTunedModel(BaseModel):
    """Fine-tuned model definition"""
    id: str
    tenant_id: str
    base_model: str  # e.g., "gpt-3.5-turbo"
    fine_tuned_model_id: str  # Provider's fine-tuned model ID
    provider: str  # "openai", "anthropic", etc.
    created_at: str
    status: str  # "pending", "succeeded", "failed"
    training_file_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FineTuneManager:
    """Manage fine-tuned models"""
    
    def __init__(self):
        self.models: Dict[str, FineTunedModel] = {}
    
    def register_finetuned_model(self, model: FineTunedModel):
        """Register a fine-tuned model"""
        self.models[model.id] = model
    
    def get_finetuned_model(
        self,
        tenant_id: str,
        model_id: str,
    ) -> Optional[FineTunedModel]:
        """Get fine-tuned model"""
        for model in self.models.values():
            if model.tenant_id == tenant_id and model.id == model_id:
                return model
        return None
    
    def get_available_finetuned_models(self, tenant_id: str) -> List[FineTunedModel]:
        """Get all fine-tuned models for a tenant"""
        return [
            model for model in self.models.values()
            if model.tenant_id == tenant_id and model.status == "succeeded"
        ]
    
    def format_model_name(self, model: FineTunedModel) -> str:
        """Format fine-tuned model name for LiteLLM"""
        # LiteLLM format: provider/model-id
        return f"{model.provider}/{model.fine_tuned_model_id}"

