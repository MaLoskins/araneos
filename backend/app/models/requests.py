from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ProcessDataRequest(BaseModel):
    data: List[Dict[str, Any]]
    config: Dict[str, Any]


class ModelConfig(BaseModel):
    model_name: str
    hidden_channels: int
    lr: float
    epochs: int
    dropout: float
    extra_params: Optional[Dict[str, Any]] = None


class TrainRequest(BaseModel):
    session_id: str
    configuration: ModelConfig
