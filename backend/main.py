from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Dict, List, Optional
import pandas as pd
import numpy as np
import json
import torch
import logging
from networkx.readwrite import json_graph

from DataFrameToGraph import DataFrameToGraph
from FeatureSpaceCreator import FeatureSpaceCreator
from TorchGeometricGraphBuilder import (
    TorchGeometricGraphBuilder, split_data,
    GCNModel, GraphSageModel, GATModel, GINModel, ChebConvModel, ResidualGCNModel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models ---

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

class TrainGNNRequest(BaseModel):
    graph: Dict[str, List]
    configuration: ModelConfig

# --- Model Factory ---

def _create_model(name: str, in_ch: int, hidden_ch: int, num_cls: int, dropout: float, extra: dict):
    """Factory function to create GNN models by name."""
    key = name.upper()
    if key in ('GCN',):
        return GCNModel(in_ch, hidden_ch, num_cls, dropout)
    if key in ('GRAPHSAGE', 'SAGE'):
        return GraphSageModel(in_ch, hidden_ch, num_cls, dropout)
    if key == 'GAT':
        heads = extra.get('heads', 8)
        return GATModel(in_ch, hidden_ch // heads, num_cls, heads, dropout)
    if key == 'GIN':
        return GINModel(in_ch, hidden_ch, num_cls, dropout)
    if key in ('CHEBCONV', 'CHEB'):
        return ChebConvModel(in_ch, hidden_ch, num_cls, extra.get('K', 3), dropout)
    if key in ('RESIDUALGCN', 'RESGCN'):
        return ResidualGCNModel(in_ch, hidden_ch, num_cls, dropout)
    return None

SUPPORTED_MODELS = {'GCN', 'GRAPHSAGE', 'SAGE', 'GAT', 'GIN', 'CHEBCONV', 'CHEB', 'RESIDUALGCN', 'RESGCN'}

# --- Endpoints ---

@app.post("/process-data")
def process_data(req: ProcessDataRequest):
    df = pd.DataFrame(req.data)
    config = req.config

    nodes_config = config.get("nodes", [])
    relationships = config.get("relationships", [])
    graph_type = config.get("graph_type", "directed")
    label_column = config.get("label_column", "")
    use_feature_space = config.get("use_feature_space", False)
    feature_space_config = config.get("feature_space_config", {})
    user_features = config.get("features", [])

    # Build graph
    graph_config = {"nodes": nodes_config, "relationships": relationships, "graph_type": graph_type}
    graph_data = json_graph.node_link_data(
        DataFrameToGraph(df, graph_config, graph_type=graph_type).get_graph()
    )

    # Attach labels
    if label_column and label_column in df.columns:
        for node in graph_data["nodes"]:
            node_id = str(node["id"])
            for nc in nodes_config:
                matching = df.loc[df[nc["id"]].astype(str) == node_id]
                if not matching.empty:
                    node.setdefault("features", {})
                    node["features"]["label"] = str(matching[label_column].values[0])
                    break

    # Generate embeddings
    feature_data = None
    if use_feature_space and feature_space_config:
        logger.info("Generating embeddings with FeatureSpaceCreator.")
        feature_data = FeatureSpaceCreator(config=feature_space_config, device="cuda").process(df)

        for feat in user_features:
            node_id_col = feat.get("node_id_column")
            col_name = feat.get("column_name")
            feat_type = feat.get("type", "text").lower()

            if not node_id_col or not col_name:
                logger.warning(f"Skipping feature {feat} - missing required fields.")
                continue

            feature_col = f"{col_name}_{'embedding' if feat_type == 'text' else 'feature'}"
            if feature_col not in feature_data.columns:
                logger.warning(f"Feature column '{feature_col}' not found. Skipping.")
                continue

            if node_id_col not in feature_data.columns:
                if node_id_col not in df.columns:
                    logger.error(f"Column '{node_id_col}' not in CSV. Cannot attach features.")
                    continue
                feature_data[node_id_col] = df[node_id_col]

            for _, row in feature_data.iterrows():
                if pd.isnull(row[node_id_col]):
                    continue
                node_id_str = str(row[node_id_col])
                val = row[feature_col].tolist() if isinstance(row[feature_col], np.ndarray) else row[feature_col]

                for n in graph_data["nodes"]:
                    if str(n["id"]) == node_id_str:
                        n.setdefault("features", {})
                        n["features"][feature_col] = val
                        break

        logger.info("Feature embeddings attached to graph nodes.")
    else:
        logger.info("No advanced embeddings requested.")

    # NetworkX 3.x uses 'edges' key; older versions used 'links'
    edge_key = 'edges' if 'edges' in graph_data else 'links'
    edge_list = graph_data.get(edge_key, [])
    logger.info(f"Returning graph with {len(graph_data.get('nodes', []))} nodes, {len(edge_list)} edges (key='{edge_key}')")
    if edge_list:
        logger.info(f"Sample edge: {edge_list[0]}")

    return {
        "graph": graph_data,
        "featureDataCsv": feature_data.to_csv(index=False) if feature_data is not None else None,
    }


@app.post("/train-gnn")
async def train_gnn(request: TrainGNNRequest) -> StreamingResponse:
    """Train a GNN model and stream back metrics."""
    try:
        graph_builder = TorchGeometricGraphBuilder(request.graph)
        data = graph_builder.build_data()

        if data.y is None:
            raise HTTPException(status_code=400, detail="No labels found. Node classification requires labeled nodes.")

        unique_labels = torch.unique(data.y)
        num_classes = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # Diagnostic logging
        logger.info(f"[TRAIN] Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}, Features per node: {data.num_node_features}")
        logger.info(f"[TRAIN] Labels: {unique_labels.tolist()}, Num classes: {num_classes}")
        label_counts = {l.item(): (data.y == l).sum().item() for l in unique_labels}
        logger.info(f"[TRAIN] Label distribution: {label_counts}")
        logger.info(f"[TRAIN] Feature sample (node 0, first 10): {data.x[0][:10].tolist()}")
        logger.info(f"[TRAIN] Feature std: {data.x.std().item():.6f}, mean: {data.x.mean().item():.6f}")

        if num_classes < 2:
            raise HTTPException(status_code=400, detail=f"Need at least 2 classes, found {num_classes}.")

        data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)

        config = request.configuration
        in_channels = data.num_node_features
        extra_params = config.extra_params or {}

        # Create model
        model = _create_model(config.model_name, in_channels, config.hidden_channels, num_classes, config.dropout, extra_params)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {config.model_name}. Available: {', '.join(sorted(SUPPORTED_MODELS))}")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        async def training_stream() -> AsyncGenerator[str, None]:
            yield json.dumps({
                "status": "started",
                "message": f"Training {config.model_name} model",
                "epoch": 0,
                "total_epochs": config.epochs,
            }) + "\n"

            best_val_loss = float('inf')

            for epoch in range(1, config.epochs + 1):
                # Train
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss = loss.item()

                # Validate
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
                    pred = out[data.val_mask].argmax(dim=1)
                    val_acc = (pred == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

                scheduler.step(val_loss)
                is_best = val_loss < best_val_loss
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

            # Test
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                test_pred = out[data.test_mask].argmax(dim=1)
                test_acc = (test_pred == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
