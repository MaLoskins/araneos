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
import uuid
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

# --- Server-Side Session Store ---
# Stores full graph data keyed by session ID. Keeps heavy data (embeddings)
# on the backend so the frontend only needs lightweight summaries.

_sessions: Dict[str, Dict] = {}

def _store_session(graph_data: dict) -> str:
    """Store graph data server-side, return session ID."""
    session_id = str(uuid.uuid4())[:8]
    _sessions[session_id] = graph_data
    # Keep max 10 sessions to avoid unbounded memory
    if len(_sessions) > 10:
        oldest = next(iter(_sessions))
        del _sessions[oldest]
    logger.info(f"Stored session {session_id} ({len(graph_data.get('nodes', []))} nodes)")
    return session_id

def _get_session(session_id: str) -> dict:
    """Retrieve graph data by session ID."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Please reprocess your graph.")
    return _sessions[session_id]

def _make_viz_node(node: dict) -> dict:
    """Strip embeddings from a node for visualization (keep labels, types, skip float arrays)."""
    viz = {"id": node["id"]}
    if "type" in node:
        viz["type"] = node["type"]
    feats = node.get("features", {})
    # Only include non-embedding features for viz
    viz_feats = {}
    for k, v in feats.items():
        if isinstance(v, (list, np.ndarray)):
            continue  # Skip embedding vectors
        viz_feats[k] = v
    if viz_feats:
        viz["features"] = viz_feats
    # Promote label to top level for easy access
    if "label" in feats:
        viz["label"] = feats["label"]
    return viz


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

class TrainRequest(BaseModel):
    session_id: str
    configuration: ModelConfig


# --- Model Factory ---

def _create_model(name: str, in_ch: int, hidden_ch: int, num_cls: int, dropout: float, extra: dict):
    key = name.upper()
    if key == 'GCN':
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
    """Process CSV data into a graph. Stores full graph server-side, returns lightweight summary."""
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
    if use_feature_space and feature_space_config:
        logger.info("Generating embeddings with FeatureSpaceCreator.")
        feature_data = FeatureSpaceCreator(config=feature_space_config, device="cuda").process(df)

        for feat in user_features:
            node_id_col = feat.get("node_id_column")
            col_name = feat.get("column_name")
            feat_type = feat.get("type", "text").lower()

            if not node_id_col or not col_name:
                continue

            feature_col = f"{col_name}_{'embedding' if feat_type == 'text' else 'feature'}"
            if feature_col not in feature_data.columns:
                logger.warning(f"Feature column '{feature_col}' not found. Skipping.")
                continue

            if node_id_col not in feature_data.columns:
                if node_id_col not in df.columns:
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

    # Normalize edge key
    edge_key = 'edges' if 'edges' in graph_data else 'links'
    edges = graph_data.get(edge_key, [])

    # Store full graph server-side
    full_graph = {"nodes": graph_data["nodes"], "links": edges, "directed": graph_data.get("directed", False)}
    session_id = _store_session(full_graph)

    # Build lightweight response for frontend (no embedding vectors)
    viz_nodes = [_make_viz_node(n) for n in graph_data["nodes"]]
    viz_edges = [{"source": e.get("source"), "target": e.get("target"), "type": e.get("type", "")} for e in edges]

    # Compute stats
    labels = [n.get("features", {}).get("label") for n in graph_data["nodes"]]
    label_set = [l for l in labels if l is not None]
    has_embeddings = any(
        isinstance(v, (list, np.ndarray))
        for n in graph_data["nodes"]
        for v in n.get("features", {}).values()
    )

    logger.info(f"Session {session_id}: {len(viz_nodes)} nodes, {len(viz_edges)} edges, {len(set(label_set))} classes")

    return {
        "session_id": session_id,
        "graph": {"nodes": viz_nodes, "edges": viz_edges, "directed": graph_data.get("directed", False)},
        "stats": {
            "node_count": len(viz_nodes),
            "edge_count": len(viz_edges),
            "label_count": len(set(label_set)),
            "labeled_nodes": len(label_set),
            "has_embeddings": has_embeddings,
            "unique_labels": list(set(label_set)),
        },
    }


@app.get("/graph/{session_id}")
def get_graph(session_id: str):
    """Get lightweight visualization data for a session."""
    full_graph = _get_session(session_id)
    viz_nodes = [_make_viz_node(n) for n in full_graph["nodes"]]
    viz_edges = [{"source": e.get("source"), "target": e.get("target"), "type": e.get("type", "")} for e in full_graph["links"]]
    return {"nodes": viz_nodes, "edges": viz_edges, "directed": full_graph.get("directed", False)}


@app.get("/graph/{session_id}/stats")
def get_graph_stats(session_id: str):
    """Get graph statistics without transferring full data."""
    full_graph = _get_session(session_id)
    nodes = full_graph["nodes"]
    edges = full_graph["links"]

    labels = [n.get("features", {}).get("label") for n in nodes]
    label_set = [l for l in labels if l is not None]

    # Degree distribution
    degrees = {}
    for n in nodes:
        degrees[str(n["id"])] = 0
    for e in edges:
        src = str(e["source"]) if not isinstance(e["source"], dict) else str(e["source"]["id"])
        tgt = str(e["target"]) if not isinstance(e["target"], dict) else str(e["target"]["id"])
        degrees[src] = degrees.get(src, 0) + 1
        degrees[tgt] = degrees.get(tgt, 0) + 1

    deg_values = list(degrees.values())
    freq = {}
    for d in deg_values:
        freq[d] = freq.get(d, 0) + 1

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "label_count": len(set(label_set)),
        "labeled_nodes": len(label_set),
        "unique_labels": list(set(label_set)),
        "avg_degree": round(sum(deg_values) / max(len(deg_values), 1), 2),
        "max_degree": max(deg_values) if deg_values else 0,
        "degree_distribution": {str(k): v for k, v in sorted(freq.items())},
        "has_embeddings": any(isinstance(v, (list, np.ndarray)) for n in nodes for v in n.get("features", {}).values()),
    }


@app.post("/train-gnn")
async def train_gnn(request: TrainRequest) -> StreamingResponse:
    """Train a GNN model using server-stored graph data. Only the session ID is needed."""
    try:
        full_graph = _get_session(request.session_id)

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
        device = torch.device('cpu')
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device='cuda')
                device = torch.device('cuda')
            except Exception:
                logger.warning("CUDA available but unusable (kernel mismatch), falling back to CPU")
        data = data.to(device)

        config = request.configuration
        extra_params = config.extra_params or {}

        model = _create_model(config.model_name, data.num_node_features, config.hidden_channels, num_classes, config.dropout, extra_params)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {config.model_name}")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        async def training_stream() -> AsyncGenerator[str, None]:
            yield json.dumps({
                "status": "started",
                "message": f"Training {config.model_name}",
                "epoch": 0,
                "total_epochs": config.epochs,
            }) + "\n"

            best_val_loss = float('inf')

            for epoch in range(1, config.epochs + 1):
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss = loss.item()

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
