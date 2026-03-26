"""End-to-end test of the training pipeline with the test dataset."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from networkx.readwrite import json_graph
from DataFrameToGraph import DataFrameToGraph
from FeatureSpaceCreator import FeatureSpaceCreator
from TorchGeometricGraphBuilder import TorchGeometricGraphBuilder, split_data
import torch
import logging

logging.basicConfig(level=logging.WARNING)

# Step 1: Load CSV
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "test_social_graph.csv"))
print(f"CSV: {len(df)} rows, columns: {list(df.columns)}")

# Step 2: Build graph (same as /process-data endpoint)
nodes_config = [{"id": "user_id", "type": "user"}, {"id": "replied_to_user", "type": "user"}]
relationships = [{"source": "user_id", "target": "replied_to_user", "type": "replied_to"}]
graph_config = {"nodes": nodes_config, "relationships": relationships, "graph_type": "directed"}

g = DataFrameToGraph(df, graph_config, graph_type="directed").get_graph()
graph_data = json_graph.node_link_data(g)
print(f"\nGraph: {len(graph_data['nodes'])} nodes, {len(graph_data.get('links', graph_data.get('edges', [])))} edges")

# Step 3: Attach labels
label_column = "user_type"
for node in graph_data["nodes"]:
    node_id = str(node["id"])
    for nc in nodes_config:
        matching = df.loc[df[nc["id"]].astype(str) == node_id]
        if not matching.empty:
            node.setdefault("features", {})
            node["features"]["label"] = str(matching[label_column].values[0])
            break

# Check label distribution
labels = [n.get("features", {}).get("label") for n in graph_data["nodes"]]
from collections import Counter
label_dist = Counter(labels)
print(f"Labels: {dict(label_dist)}")
print(f"Nodes with label: {sum(1 for l in labels if l is not None)}/{len(labels)}")

# Step 4: Generate embeddings
feature_config = {"features": [
    {"column_name": "message", "type": "text", "embedding_method": "word2vec", "embedding_dim": 128}
]}
feature_data = FeatureSpaceCreator(config=feature_config, device="cpu").process(df)
print(f"\nEmbeddings shape: {feature_data.shape}")
print(f"Embedding columns: {[c for c in feature_data.columns if 'embedding' in c or 'feature' in c]}")

# Attach embeddings to nodes
feat_col = "message_embedding"
if feat_col in feature_data.columns:
    feature_data["user_id"] = df["user_id"]
    attached = 0
    for _, row in feature_data.iterrows():
        node_id_str = str(row["user_id"])
        val = row[feat_col].tolist() if isinstance(row[feat_col], np.ndarray) else row[feat_col]
        for n in graph_data["nodes"]:
            if str(n["id"]) == node_id_str:
                n.setdefault("features", {})
                n["features"][feat_col] = val
                attached += 1
                break
    print(f"Attached embeddings to {attached} rows (multiple per node is fine)")

# Check what features look like
sample_node = graph_data["nodes"][0]
feats = sample_node.get("features", {})
print(f"\nSample node '{sample_node['id']}' features:")
print(f"  label: {feats.get('label')}")
embedding = feats.get("message_embedding")
if embedding:
    if isinstance(embedding, list):
        print(f"  message_embedding: list of {len(embedding)} floats, first 5: {embedding[:5]}")
    elif isinstance(embedding, np.ndarray):
        print(f"  message_embedding: ndarray shape {embedding.shape}, first 5: {embedding[:5].tolist()}")
    else:
        print(f"  message_embedding: type={type(embedding)}, value={embedding}")
else:
    print(f"  message_embedding: MISSING!")
    print(f"  all feature keys: {list(feats.keys())}")

# Step 5: Build PyG data
# Simulate what the frontend sends: edges in the 'links' key
edge_key = 'links' if 'links' in graph_data else 'edges'
send_data = {"nodes": graph_data["nodes"], "links": graph_data.get(edge_key, [])}
print(f"\nSending to TorchGeometricGraphBuilder: {len(send_data['nodes'])} nodes, {len(send_data['links'])} links")

builder = TorchGeometricGraphBuilder(send_data)
data = builder.build_data()

print(f"\nPyG Data:")
print(f"  x shape: {data.x.shape} (nodes x features)")
print(f"  edge_index shape: {data.edge_index.shape}")
print(f"  y: {data.y}")
print(f"  unique labels: {torch.unique(data.y).tolist()}")
print(f"  label counts: { {l.item(): (data.y == l).sum().item() for l in torch.unique(data.y)} }")
print(f"  feature std: {data.x.std():.4f}, mean: {data.x.mean():.4f}")
print(f"  any NaN in features: {torch.isnan(data.x).any()}")
print(f"  any Inf in features: {torch.isinf(data.x).any()}")

# Check if all features are identical
unique_rows = len(torch.unique(data.x, dim=0))
print(f"  unique feature vectors: {unique_rows}/{data.x.shape[0]}")

if unique_rows == 1:
    print("\n*** ALL NODES HAVE IDENTICAL FEATURES - THIS IS THE BUG ***")
    print("The GNN cannot distinguish nodes, so it predicts majority class.")

# Step 6: Train all models
data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
print(f"\nSplit: train={data.train_mask.sum()}, val={data.val_mask.sum()}, test={data.test_mask.sum()}")

device = torch.device('cpu')
data = data.to(device)

num_classes = len(torch.unique(data.y)) - (1 if -1 in torch.unique(data.y) else 0)
in_channels = data.num_node_features

from TorchGeometricGraphBuilder import GCNModel, GraphSageModel, GATModel, GINModel, ChebConvModel, ResidualGCNModel

models = {
    "GCN": GCNModel(in_channels, 64, num_classes, 0.3),
    "GraphSAGE": GraphSageModel(in_channels, 64, num_classes, 0.3),
    "GAT": GATModel(in_channels, 8, num_classes, 8, 0.3),
    "GIN": GINModel(in_channels, 64, num_classes, 0.3),
    "ChebConv": ChebConvModel(in_channels, 64, num_classes, 3, 0.3),
    "ResidualGCN": ResidualGCNModel(in_channels, 64, num_classes, 0.3),
}

print(f"\n{'='*60}")
print(f"Training all models (200 epochs each, {in_channels} features, {num_classes} classes)")
print(f"{'='*60}")

for name, model in models.items():
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out[data.val_mask].argmax(dim=1)
                val_acc = (pred == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
                best_val_acc = max(best_val_acc, val_acc)

    # Final test
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred = out[data.test_mask].argmax(dim=1)
        test_acc = (test_pred == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    print(f"  {name:15s} | val_acc: {best_val_acc*100:.1f}% | test_acc: {test_acc*100:.1f}%")
