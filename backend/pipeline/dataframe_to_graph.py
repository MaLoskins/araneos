import pandas as pd
import networkx as nx
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataFrameToGraph:
    """Converts a DataFrame and config into a NetworkX graph."""

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any], graph_type: str = 'directed'):
        self.df = df
        self.config = config
        self.graph_type = graph_type.lower()
        self.graph = self._initialize_graph()
        self.node_registry = {}
        self._validate_config()
        self._parse_dataframe()

    def _initialize_graph(self) -> nx.Graph:
        if self.graph_type == 'directed':
            return nx.MultiDiGraph()
        elif self.graph_type == 'undirected':
            return nx.MultiGraph()
        raise ValueError("graph_type must be 'directed' or 'undirected'.")

    def _validate_config(self):
        for key in ['nodes', 'relationships']:
            if key not in self.config:
                raise KeyError(f"Configuration missing required key: '{key}'")

        for n in self.config['nodes']:
            if 'id' not in n:
                raise KeyError("Each node configuration must have an 'id' key.")
            if 'type' not in n:
                logger.warning(f"Node configuration {n} missing 'type'. Defaulting to 'default'.")

        for r in self.config['relationships']:
            if 'source' not in r or 'target' not in r:
                raise KeyError("Each relationship configuration must have 'source' and 'target' keys.")
            if 'type' not in r:
                logger.warning(f"Relationship configuration {r} missing 'type'. Defaulting to 'default'.")

    def _parse_dataframe(self):
        """Create nodes and edges from DataFrame rows."""
        for idx, row in self.df.iterrows():
            for nc in self.config['nodes']:
                self._process_node(idx, row, nc)
            for rc in self.config['relationships']:
                self._process_edge(idx, row, rc)

    def _process_node(self, idx, row, nc):
        nid = row.get(nc['id'], None)
        if nid is None or pd.isnull(nid) or str(nid).strip() == '':
            logger.warning(f"Row {idx}: Missing or empty node ID for '{nc['id']}'. Skipping.")
            return

        nid_str = str(nid)
        ntype = nc.get('type', 'default')
        features = {c: row[c] for c in self.df.columns if c.endswith(('_embedding', '_feature'))}
        self._add_node(nid_str, ntype, features)

    def _process_edge(self, idx, row, rc):
        src_col, tgt_col = rc['source'], rc['target']
        rel_type = rc.get('type', 'default')
        src_id, tgt_id = row.get(src_col), row.get(tgt_col)

        if any(v is None or pd.isnull(v) or str(v).strip() == '' for v in [src_id, tgt_id]):
            logger.warning(f"Row {idx}: Missing source/target for '{rel_type}'. Skipping.")
            return

        src_str, tgt_str = str(src_id), str(tgt_id)

        if src_str not in self.node_registry:
            logger.warning(f"Row {idx}: Source node '{src_str}' not selected. Skipping.")
            return
        if tgt_str not in self.node_registry:
            logger.warning(f"Row {idx}: Target node '{tgt_str}' not selected. Skipping.")
            return

        self._add_edge(src_str, tgt_str, rel_type)

    def _add_node(self, node_id: str, node_type: str, features: dict):
        if node_id in self.node_registry:
            return
        self.node_registry[node_id] = {'type': node_type, 'features': features}
        self.graph.add_node(node_id, type=node_type, features=features)
        logger.info(f"Added node {node_id} (type='{node_type}', features={list(features.keys())})")

    def _add_edge(self, source_id: str, target_id: str, rel_type: str):
        if self.graph.has_edge(source_id, target_id, key=rel_type):
            return
        self.graph.add_edge(source_id, target_id, key=rel_type, type=rel_type)
        logger.info(f"Added edge {source_id} -> {target_id} (type='{rel_type}')")

    def get_graph(self) -> nx.Graph:
        return self.graph
