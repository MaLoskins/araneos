import logging
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph

from pipeline.dataframe_to_graph import DataFrameToGraph
from pipeline.feature_space_creator import FeatureSpaceCreator

logger = logging.getLogger(__name__)


def make_viz_node(node: dict) -> dict:
    """Strip embeddings from a node for visualization (keep labels, types, skip float arrays)."""
    viz = {"id": node["id"]}
    if "type" in node:
        viz["type"] = node["type"]
    feats = node.get("features", {})
    viz_feats = {}
    for k, v in feats.items():
        if isinstance(v, (list, np.ndarray)):
            continue
        viz_feats[k] = v
    if viz_feats:
        viz["features"] = viz_feats
    if "label" in feats:
        viz["label"] = feats["label"]
    return viz


def process_graph(df: pd.DataFrame, config: dict, session_store) -> dict:
    """Process CSV data into a graph. Stores full graph, returns lightweight summary."""
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

    # Store full graph
    full_graph = {"nodes": graph_data["nodes"], "links": edges, "directed": graph_data.get("directed", False)}
    session_id = session_store.store(full_graph)

    # Build lightweight response
    viz_nodes = [make_viz_node(n) for n in graph_data["nodes"]]
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


def get_viz_data(session_store, session_id: str) -> dict:
    """Get lightweight visualization data for a session."""
    full_graph = session_store.get(session_id)
    viz_nodes = [make_viz_node(n) for n in full_graph["nodes"]]
    viz_edges = [{"source": e.get("source"), "target": e.get("target"), "type": e.get("type", "")} for e in full_graph["links"]]
    return {"nodes": viz_nodes, "edges": viz_edges, "directed": full_graph.get("directed", False)}


def get_stats(session_store, session_id: str) -> dict:
    """Get graph statistics."""
    full_graph = session_store.get(session_id)
    nodes = full_graph["nodes"]
    edges = full_graph["links"]

    labels = [n.get("features", {}).get("label") for n in nodes]
    label_set = [l for l in labels if l is not None]

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
