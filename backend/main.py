"""
Araneos API entry point.

Uses the app factory from app/ package. Re-exports key names for backward
compatibility with existing tests.
"""
import logging
import sys
import os

# Ensure the backend directory is on the path for pipeline imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)

from app import create_app  # noqa: E402
from app.models.requests import ProcessDataRequest, ModelConfig, TrainRequest  # noqa: E402, F401
from app.dependencies import get_session_store  # noqa: E402

# Re-export for test backward compatibility
from pipeline.torch_geometric_builder import (  # noqa: E402, F401
    TorchGeometricGraphBuilder, split_data,
    GCNModel, GraphSageModel, GATModel, GINModel, ChebConvModel, ResidualGCNModel,
)

app = create_app()

# Expose session store internals for test backward compatibility
_store = get_session_store()
_sessions = getattr(_store, '_sessions', {})
_sessions_lock = getattr(_store, '_lock', __import__('threading').Lock())

# Re-export the train_gnn function so tests can import it from main
from app.routes.training import train_gnn  # noqa: E402, F401


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
