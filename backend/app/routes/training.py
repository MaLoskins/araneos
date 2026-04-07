import asyncio
import json
import logging
import torch
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.models.requests import TrainRequest
from app.services import training_service
from app.dependencies import get_session_store
from pipeline.torch_geometric_builder import TorchGeometricGraphBuilder, split_data

logger = logging.getLogger(__name__)
router = APIRouter(tags=["training"])


@router.post("/train-gnn")
async def train_gnn(request: TrainRequest, session_store=Depends(get_session_store)) -> StreamingResponse:
    """Train a GNN model using server-stored graph data. Streaming NDJSON response."""
    try:
        try:
            full_graph = session_store.get(request.session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found. Please reprocess your graph.")

        graph_builder = TorchGeometricGraphBuilder(full_graph)
        data = graph_builder.build_data()

        if data.y is None:
            raise HTTPException(status_code=400, detail="No labels found. Node classification requires labeled nodes.")

        unique_labels = torch.unique(data.y)
        num_classes = len(unique_labels) - (1 if -1 in unique_labels else 0)

        logger.info(f"[TRAIN] Session: {request.session_id}, Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}, Features: {data.num_node_features}, Classes: {num_classes}")

        if num_classes < 2:
            raise HTTPException(status_code=400, detail=f"Need at least 2 classes, found {num_classes}.")

        data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        device = training_service.get_device()
        data = data.to(device)

        config = request.configuration
        extra_params = config.extra_params or {}

        model = training_service.create_model(config.model_name, data.num_node_features, config.hidden_channels, num_classes, config.dropout, extra_params)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {config.model_name}")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        async def training_stream() -> AsyncGenerator[str, None]:
            loop = asyncio.get_event_loop()

            yield json.dumps({
                "status": "started",
                "message": f"Training {config.model_name}",
                "epoch": 0,
                "total_epochs": config.epochs,
            }) + "\n"

            best_val_loss = float('inf')

            for epoch in range(1, config.epochs + 1):
                train_loss, val_loss, val_acc, is_best = await loop.run_in_executor(
                    None, training_service.run_epoch, model, data, optimizer, criterion, scheduler, best_val_loss
                )
                if is_best:
                    best_val_loss = val_loss

                yield json.dumps({
                    "epoch": epoch,
                    "total_epochs": config.epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "is_best_model": is_best,
                }) + "\n"

            test_acc = await loop.run_in_executor(None, training_service.run_test, model, data)

            yield json.dumps({
                "status": "completed",
                "message": "Training completed",
                "test_accuracy": test_acc,
                "best_val_loss": best_val_loss,
            }) + "\n"

        return StreamingResponse(training_stream(), media_type="application/x-ndjson")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training GNN: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error training GNN: {e}")


# --- Async (Celery-backed) training endpoints ---

@router.post("/train-gnn/async")
def train_gnn_async(request: TrainRequest, session_store=Depends(get_session_store)):
    """Submit a training job to the background task queue. Returns task_id for polling."""
    try:
        session_store.get(request.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found. Please reprocess your graph.")

    from app.tasks.training import train_gnn_task

    config = request.configuration
    config_dict = {
        "model_name": config.model_name,
        "hidden_channels": config.hidden_channels,
        "lr": config.lr,
        "epochs": config.epochs,
        "dropout": config.dropout,
        "extra_params": config.extra_params,
    }

    task = train_gnn_task.delay(request.session_id, config_dict)
    return {"task_id": task.id, "status": "submitted"}


@router.get("/train-gnn/{task_id}/status")
def train_gnn_status(task_id: str):
    """Poll training progress. Returns latest metrics and accumulated NDJSON progress."""
    from app.tasks.training import train_gnn_task

    result = train_gnn_task.AsyncResult(task_id)

    # Read progress lines from Redis
    progress_lines = []
    try:
        import redis
        from app.config import get_settings
        settings = get_settings()
        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        raw_lines = r.lrange(f"training:{task_id}:progress", 0, -1)
        progress_lines = [json.loads(line) for line in raw_lines]
    except Exception:
        pass

    response = {
        "task_id": task_id,
        "state": result.state,
        "progress": progress_lines,
    }

    if result.state == "TRAINING" and result.info:
        response["current"] = result.info
    elif result.state == "SUCCESS" and result.result:
        response["result"] = result.result
    elif result.state == "FAILURE":
        response["error"] = str(result.result)

    return response
