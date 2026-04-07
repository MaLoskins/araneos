import json
import logging
import time
import torch

from app.celery_app import celery_app
from app.dependencies import get_session_store
from app.services import training_service
from pipeline.torch_geometric_builder import TorchGeometricGraphBuilder, split_data

logger = logging.getLogger(__name__)


def _get_redis_client():
    """Get a raw Redis client for progress updates."""
    import redis
    from app.config import get_settings
    settings = get_settings()
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


@celery_app.task(bind=True, name="train_gnn_task")
def train_gnn_task(self, session_id: str, model_config: dict):
    """
    Background Celery task for GNN training.

    Writes progress to Redis key `training:{task_id}:progress` as NDJSON lines.
    Final result is stored in Celery's result backend and also at `training:{task_id}:result`.
    """
    task_id = self.request.id
    redis_client = _get_redis_client()
    progress_key = f"training:{task_id}:progress"
    result_key = f"training:{task_id}:result"

    def emit(data: dict):
        """Append a progress line to Redis."""
        line = json.dumps(data)
        redis_client.rpush(progress_key, line)
        redis_client.expire(progress_key, 7200)

    try:
        session_store = get_session_store()
        full_graph = session_store.get(session_id)

        graph_builder = TorchGeometricGraphBuilder(full_graph)
        data = graph_builder.build_data()

        if data.y is None:
            emit({"status": "error", "message": "No labels found. Node classification requires labeled nodes."})
            return {"status": "error", "message": "No labels found."}

        unique_labels = torch.unique(data.y)
        num_classes = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if num_classes < 2:
            emit({"status": "error", "message": f"Need at least 2 classes, found {num_classes}."})
            return {"status": "error", "message": f"Need at least 2 classes, found {num_classes}."}

        data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        device = training_service.get_device()
        data = data.to(device)

        config_name = model_config["model_name"]
        hidden_ch = model_config["hidden_channels"]
        lr = model_config["lr"]
        epochs = model_config["epochs"]
        dropout = model_config["dropout"]
        extra_params = model_config.get("extra_params", {})

        model = training_service.create_model(config_name, data.num_node_features, hidden_ch, num_classes, dropout, extra_params)
        if model is None:
            emit({"status": "error", "message": f"Unsupported model: {config_name}"})
            return {"status": "error", "message": f"Unsupported model: {config_name}"}
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        emit({
            "status": "started",
            "message": f"Training {config_name}",
            "epoch": 0,
            "total_epochs": epochs,
        })

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, val_loss, val_acc, is_best = training_service.run_epoch(
                model, data, optimizer, criterion, scheduler, best_val_loss
            )
            if is_best:
                best_val_loss = val_loss

            emit({
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "is_best_model": is_best,
            })

            # Update Celery task state for polling
            self.update_state(
                state="TRAINING",
                meta={"epoch": epoch, "total_epochs": epochs, "val_accuracy": val_acc},
            )

        test_acc = training_service.run_test(model, data)
        elapsed = time.time() - start_time

        result = {
            "status": "completed",
            "message": "Training completed",
            "test_accuracy": test_acc,
            "best_val_loss": best_val_loss,
            "training_time": elapsed,
        }

        emit(result)
        redis_client.setex(result_key, 7200, json.dumps(result))

        return result

    except Exception as e:
        logger.error(f"Celery training task failed: {e}", exc_info=True)
        emit({"status": "error", "message": str(e)})
        return {"status": "error", "message": str(e)}
