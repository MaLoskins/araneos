from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class GraphSummaryResponse(BaseModel):
    session_id: str
    graph: Dict[str, Any]
    stats: Dict[str, Any]


class GraphStatsResponse(BaseModel):
    node_count: int
    edge_count: int
    label_count: int
    labeled_nodes: int
    unique_labels: List[str]
    avg_degree: float
    max_degree: int
    degree_distribution: Dict[str, int]
    has_embeddings: bool


class TrainingMessage(BaseModel):
    status: Optional[str] = None
    message: Optional[str] = None
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    is_best_model: Optional[bool] = None
    test_accuracy: Optional[float] = None
    best_val_loss: Optional[float] = None
