# TorchGeometricGraphBuilder.py
"""PyTorch Geometric graph builder with GNN architectures"""

import json,os,random,numpy as np,matplotlib.pyplot as plt,umap,torch,torch.nn as nn,torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,GINConv,ChebConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.naive_bayes import GaussianNB

# Reproducibility
def set_seed(seed:int=42):
    """Set random seed for reproducibility"""
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
set_seed(42)

# Device config
device=torch.device('cpu')
if torch.cuda.is_available():
    try:
        torch.zeros(1,device='cuda')
        device=torch.device('cuda')
    except Exception:
        print("CUDA available but unusable (kernel mismatch), falling back to CPU")
print(f"Using device: {device}")

# Naive Bayes Baseline
class NaiveBayesBaseline:
    """GaussianNB baseline for node classification"""
    def __init__(self):self.clf=GaussianNB()
    def fit(self,X,y):self.clf.fit(X,y)
    def predict(self,X):return self.clf.predict(X)

# Naive Bayes Helpers
def train_naive_bayes(model:NaiveBayesBaseline,data:Data):
    """Train Naive Bayes model on training set"""
    x_train,y_train=data.x[data.train_mask].cpu().numpy(),data.y[data.train_mask].cpu().numpy()
    valid_idx=y_train!=-1
    model.fit(x_train[valid_idx],y_train[valid_idx])
    return model

def validate_naive_bayes(model:NaiveBayesBaseline,data:Data):
    """Get validation accuracy for Naive Bayes"""
    x_val,y_val=data.x[data.val_mask].cpu().numpy(),data.y[data.val_mask].cpu().numpy()
    valid_idx=y_val!=-1
    return accuracy_score(y_val[valid_idx],model.predict(x_val[valid_idx]))

def test_naive_bayes(model:NaiveBayesBaseline,data:Data):
    """Evaluate Naive Bayes on test set"""
    x_test,y_test=data.x[data.test_mask].cpu().numpy(),data.y[data.test_mask].cpu().numpy()
    valid_idx=y_test!=-1
    x_test,y_test=x_test[valid_idx],y_test[valid_idx]
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    report=classification_report(y_test,y_pred,digits=4)
    print(f"Test Accuracy (Naive Bayes): {acc:.4f}\nClassification Report (Naive Bayes):\n{report}")
    return acc


# GNN Model Classes
class GCNModel(nn.Module):
    """Graph Convolutional Network for Node Classification"""
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,dropout:float=0.3):
        super(GCNModel,self).__init__()
        self.conv1=GCNConv(in_channels,hidden_channels)
        self.conv2=GCNConv(hidden_channels,out_channels)
        self.dropout=dropout

    def forward(self,x,edge_index):
        x=F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training)
        return self.conv2(x,edge_index)

class ResidualGCNModel(nn.Module):
    """Residual GCN for Node Classification"""
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,dropout:float=0.3):
        super(ResidualGCNModel,self).__init__()
        self.conv1=GCNConv(in_channels,hidden_channels)
        self.conv2=GCNConv(hidden_channels,hidden_channels)
        self.conv3=GCNConv(hidden_channels,out_channels)
        self.dropout=dropout
        self.residual_transform=nn.Linear(in_channels,out_channels) if in_channels!=out_channels else None

    def forward(self,x,edge_index):
        residual=x
        x=F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training)
        x=F.dropout(F.relu(self.conv2(x,edge_index)),p=self.dropout,training=self.training)
        x=self.conv3(x,edge_index)
        if self.residual_transform:residual=self.residual_transform(residual)
        return x+residual

class GraphSageModel(nn.Module):
    """GraphSAGE for Node Classification"""
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,dropout:float=0.3):
        super(GraphSageModel,self).__init__()
        self.conv1=SAGEConv(in_channels,hidden_channels)
        self.conv2=SAGEConv(hidden_channels,out_channels)
        self.dropout=dropout

    def forward(self,x,edge_index):
        return self.conv2(F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training),edge_index)

class GATModel(nn.Module):
    """Graph Attention Network for Node Classification"""
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,heads:int=8,dropout:float=0.6):
        super(GATModel,self).__init__()
        self.conv1=GATConv(in_channels,hidden_channels,heads=heads,dropout=dropout)
        self.conv2=GATConv(hidden_channels*heads,out_channels,heads=1,concat=False,dropout=dropout)
        self.dropout=dropout

    def forward(self,x,edge_index):
        return self.conv2(F.dropout(F.elu(self.conv1(x,edge_index)),p=self.dropout,training=self.training),edge_index)

class GINModel(nn.Module):
    """Graph Isomorphism Network for Node Classification"""
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,dropout:float=0.3):
        super(GINModel,self).__init__()
        self.conv1=GINConv(nn.Sequential(nn.Linear(in_channels,hidden_channels),nn.ReLU(),nn.Linear(hidden_channels,hidden_channels)))
        self.conv2=GINConv(nn.Sequential(nn.Linear(hidden_channels,hidden_channels),nn.ReLU(),nn.Linear(hidden_channels,hidden_channels)))
        self.linear=nn.Linear(hidden_channels,out_channels)
        self.dropout=dropout

    def forward(self,x,edge_index):
        x=F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training)
        return self.linear(F.relu(self.conv2(x,edge_index)))

class ChebConvModel(nn.Module):
    """Chebyshev Convolution Network for Node Classification"""
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,K:int=3,dropout:float=0.3):
        super(ChebConvModel,self).__init__()
        self.conv1=ChebConv(in_channels,hidden_channels,K=K)
        self.conv2=ChebConv(hidden_channels,out_channels,K=K)
        self.dropout=dropout

    def forward(self,x,edge_index):
        return self.conv2(F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training),edge_index)

# Graph Builder Class
class TorchGeometricGraphBuilder:
    """Builds PyG Data object from node-link JSON"""
    def __init__(self,data_json:dict):
        self.data_json=data_json
        self.node_id_map={}

    def build_data(self) -> Data:
        """
        Converts JSON node-link data into a torch_geometric.data.Data object.
        """
        # 1) Map node IDs to integers
        all_nodes = self.data_json.get("nodes", [])
        for idx, node_obj in enumerate(all_nodes):
            node_id = str(node_obj["id"])
            self.node_id_map[node_id] = idx

        # 2) Collect node features & labels
        # First pass: determine which feature keys exist across all nodes
        # so we can build consistent-length vectors
        feature_keys = []
        for node_obj in all_nodes:
            feats_dict = node_obj.get("features", {})
            for key in feats_dict:
                if key == "label":
                    continue
                if key not in feature_keys:
                    feature_keys.append(key)
        # Sort for deterministic ordering
        feature_keys.sort()

        node_features_list = []
        labels_list = []
        for node_obj in all_nodes:
            feats_dict = node_obj.get("features", {})
            feats_vector = []

            for key in feature_keys:
                val = feats_dict.get(key, None)
                if val is None:
                    # Will be padded below
                    continue
                elif isinstance(val, (list, np.ndarray)):
                    try:
                        feats_vector.extend([float(v) for v in val])
                    except (ValueError, TypeError):
                        print(f"Invalid values in '{key}' for node {node_obj['id']}. Using zeros.")
                        feats_vector.extend([0.0] * len(val))
                elif isinstance(val, (int, float)):
                    try:
                        feats_vector.append(float(val))
                    except (ValueError, TypeError):
                        feats_vector.append(0.0)
                # Skip string features (like labels) that aren't numeric

            # If no features found, add a dummy
            if not feats_vector:
                feats_vector.append(0.0)

            # Label
            label_val = feats_dict.get("label", None)
            labels_list.append(label_val)

            node_features_list.append(feats_vector)

        x = self._to_tensor(node_features_list, dtype=torch.float)
        x = self._normalize_features(x)

        # 4) Build edge_index
        all_links = self.data_json.get("links", self.data_json.get("edges", []))
        source_indices = []
        target_indices = []
        for link_obj in all_links:
            source_val = link_obj.get("source", {})
            target_val = link_obj.get("target", {})
            # Handle both dict format {"id": "x"} and plain string "x"
            s_id = str(source_val.get("id", "")) if isinstance(source_val, dict) else str(source_val)
            t_id = str(target_val.get("id", "")) if isinstance(target_val, dict) else str(target_val)
            s_idx = self.node_id_map.get(s_id)
            t_idx = self.node_id_map.get(t_id)

            if s_idx is None or t_idx is None:
                print(f"Warning: Edge from '{s_id}' to '{t_id}' contains undefined node IDs. Skipping.")
                continue

            source_indices.append(s_idx)
            target_indices.append(t_idx)

        if not source_indices or not target_indices:
            raise ValueError("No valid edges found in the dataset.")

        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

        # 6) Build y (labels)
        unique_labels = sorted(set(lbl for lbl in labels_list if lbl is not None))
        if unique_labels:
            label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
            y_data = []
            for lbl in labels_list:
                if lbl is None:
                    y_data.append(-1)  # unlabeled sentinel
                else:
                    y_data.append(label_to_idx[lbl])
            y = torch.tensor(y_data, dtype=torch.long)
        else:
            y = None

        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    @staticmethod
    def _to_tensor(list_of_lists,dtype=torch.float):
        """Pad lists with zeros and convert to Tensor"""
        max_len=max(len(row) for row in list_of_lists)
        return torch.tensor([row+[0.0]*(max_len-len(row)) for row in list_of_lists],dtype=dtype)

    @staticmethod
    def _normalize_features(x:torch.Tensor)->torch.Tensor:
        """Normalize features to zero mean and unit variance"""
        mean,std=x.mean(dim=0,keepdim=True),x.std(dim=0,keepdim=True)+1e-6
        return (x-mean)/std


# -------------------------- PCA Function ----------------------- #
def reduce_feature_dimensions(x:torch.Tensor,n_components:int=50)->torch.Tensor:
    """Reduce feature dimensions with PCA"""
    try:
        return torch.tensor(PCA(n_components=n_components,random_state=42).fit_transform(x.cpu().numpy()),
                           dtype=torch.float).to(x.device)
    except Exception as e:
        print(f"Error during PCA: {e}")
        raise e

# -------------------------- Data Splitting Function ------------------------ #
def split_data(data: Data, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Splits data into training, validation, and test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1."

    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    labels = data.y.numpy()

    labeled_mask = (labels != -1)
    labeled_indices = indices[labeled_mask]
    labeled_labels = labels[labeled_mask]

    if len(labeled_indices) == 0:
        raise ValueError("No labeled nodes found in the dataset.")

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        labeled_indices, labeled_labels, stratify=labeled_labels,
        test_size=(1 - train_ratio), random_state=42
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, stratify=y_temp,
        test_size=(1 - val_size), random_state=42
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


# ---------------------- GNN Training and Evaluation ----------------------- #
def train(model:nn.Module,data:Data,optimizer,criterion,epoch:int,edge_drop_prob:float=0.0):
    """Train GNN model for one epoch"""
    model.train();optimizer.zero_grad()
    
    # Edge dropout
    edge_index=data.edge_index[:,torch.rand(data.edge_index.size(1))>edge_drop_prob] if edge_drop_prob>0 else data.edge_index
    
    # Forward, backward, optimize
    out=model(data.x,edge_index)
    loss=criterion(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
    optimizer.step()
    return loss.item()

def validate(model:nn.Module,data:Data,criterion):
    """Validate GNN model on validation set"""
    model.eval()
    with torch.no_grad():
        return criterion(model(data.x,data.edge_index)[data.val_mask],data.y[data.val_mask]).item()

def test(model:nn.Module,data:Data):
    """Test GNN model on test set"""
    model.eval()
    with torch.no_grad():
        pred=model(data.x,data.edge_index).argmax(dim=1).cpu().numpy()
        y_true=data.y.cpu().numpy()
        
        test_mask=data.test_mask.cpu().numpy()
        y_pred,y_true_test=pred[test_mask],y_true[test_mask]
        
        acc=accuracy_score(y_true_test,y_pred)
        report=classification_report(y_true_test,y_pred,digits=4)
    
    print(f"Test Accuracy: {acc:.4f}\nClassification Report:\n{report}")
    return acc


# -------------------------- Visualization Function ------------------------ #
def visualize_embeddings(data: Data, model: nn.Module, title: str = "Node Embeddings"):
    """
    Visualizes node embeddings using UMAP.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device)).cpu().numpy()

    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                          c=data.y.cpu().numpy(), cmap='viridis', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()

# -------------------------- Class Distribution Function -------------------- #
def print_class_distribution(data: Data):
    """
    Prints the distribution of classes in the dataset (train, val, test).
    """
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    unique, counts = np.unique(y[train_mask], return_counts=True)
    print("Training Set Class Distribution:", dict(zip(unique, counts)))

    unique, counts = np.unique(y[val_mask], return_counts=True)
    print("Validation Set Class Distribution:", dict(zip(unique, counts)))

    unique, counts = np.unique(y[test_mask], return_counts=True)
    print("Test Set Class Distribution:", dict(zip(unique, counts)))


# -------------------------- Misclassification Analysis --------------------- #
def analyze_misclassifications(model: nn.Module, data: Data, model_name: str):
    """
    Analyzes misclassified nodes in the test set.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()

        test_mask = data.test_mask.cpu().numpy()
        misclassified = np.where((pred != y_true) & test_mask)[0]

        print(f"\n--- Misclassifications for {model_name} ---")
        print(f"Number of Misclassified Nodes: {len(misclassified)}")
        if len(misclassified) > 0:
            print("Sample Misclassifications (node indices):", misclassified[:10])


# -------------------------- Ensemble Method Function ----------------------- #
def ensemble_predictions(models: dict, data: Data):
    """
    Aggregates predictions from multiple (GNN) models via majority voting.
    (Naive Bayes can be included if adapted to produce logits or we treat
     NB predictions in the same numeric class ID format.)
    """
    model_preds = []
    for name, model in models.items():
        # For simplicity, assume GNN-like forward. If NB is included,
        # you would need to handle that separately (get class IDs).
        model.eval()
        with torch.no_grad():
            try:
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1).cpu().numpy()
                model_preds.append(pred)
            except Exception as e:
                print(f"Error getting predictions from model {name}: {e}")
                continue

    if not model_preds:
        raise ValueError("No model predictions available for ensemble.")

    model_preds = np.array(model_preds)  # shape: [num_models, num_nodes]
    ensemble_preds = []
    for i in range(model_preds.shape[1]):
        counts = np.bincount(model_preds[:, i])
        ensemble_preds.append(np.argmax(counts))
    ensemble_preds = np.array(ensemble_preds)
    return ensemble_preds


# ------------------------ Structural Features Function -------------------- #
def add_structural_features(data: Data):
    """
    Adds node degree as an additional feature.
    """
    deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
    deg = deg.unsqueeze(1)
    data.x = torch.cat([data.x, deg], dim=1)


# -------------------------- Main Execution Flow --------------------------- #
def main():
    """
    Main function to build graph data, define models, train, and evaluate.
    Includes a Naive Bayes baseline for comparison.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train multiple models on graph data.")
    parser.add_argument('--json_path', type=str, default='test_graph_data.json',
                        help='Path to the JSON file containing graph data.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs for each GNN.')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels for GNN models.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping (GNNs only).')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (for GNNs).')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate for GNN models.')
    parser.add_argument('--edge_drop_prob', type=float, default=0.0,
                        help='Probability of dropping edges during training (data augmentation).')
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components for feature dimensionality reduction.')
    args = parser.parse_args()

    # 1. Load JSON data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, args.json_path)

    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}. Please provide a valid path.")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # 2. Build PyG Data object
    builder = TorchGeometricGraphBuilder(data_json)
    try:
        data = builder.build_data()
    except Exception as e:
        print(f"Error building data: {e}")
        return

    # 3. Add Structural Features (e.g., node degree)
    try:
        add_structural_features(data)
    except Exception as e:
        print(f"Error adding structural features: {e}")
        return

    # 4. Apply PCA to reduce feature dimensions
    try:
        data.x = reduce_feature_dimensions(data.x, n_components=args.pca_components)
    except Exception as e:
        print(f"Error applying PCA: {e}")
        return

    # 5. Split data
    try:
        data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return

    data = data.to(device)

    print("\n==== Torch Geometric Data ====")
    print(data)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"x shape: {data.x.shape}")

    if data.y is None:
        print("No labels found. Exiting.")
        return

    print("\ny:", data.y)
    print("Unique label IDs:", torch.unique(data.y))
    print_class_distribution(data)

    # 6. Prepare label info
    unique_labels = torch.unique(data.y)
    num_classes = len(unique_labels) - (1 if -1 in unique_labels else 0)
    if num_classes < 2:
        print("Insufficient number of classes for classification. Exiting.")
        return

    in_channels = data.num_node_features
    hidden_channels = args.hidden_channels

    # 6.a Create the GNN models
    gnn_models = {
        "GCN": GCNModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, dropout=args.dropout),
        "GraphSAGE": GraphSageModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, dropout=args.dropout),
        "GAT": GATModel(in_channels=in_channels, hidden_channels=hidden_channels // 8, out_channels=num_classes, heads=8, dropout=args.dropout),
        "GIN": GINModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, dropout=args.dropout),
        "ChebConv": ChebConvModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, K=3, dropout=args.dropout),
        "ResidualGCN": ResidualGCNModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, dropout=args.dropout)
    }

    # 6.b Add Naive Bayes baseline
    # We'll store it in the same dictionary, but the logic to train/test differs.
    models = {
        "NaiveBayes": NaiveBayesBaseline()
    }
    models.update(gnn_models)

    # 7. GNN models => device; skip for Naive Bayes
    for name, model in models.items():
        if name != "NaiveBayes":
            model.to(device)

    # 8. Define optimizer and loss only for GNNs
    optimizers = {}
    criterions = {}
    for name, model in models.items():
        if name == "NaiveBayes":
            optimizers[name] = None
            criterions[name] = None
        else:
            optimizers[name] = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            criterions[name] = nn.CrossEntropyLoss()

    # 9. Class weights (skip for NaiveBayes)
    y_train = data.y[data.train_mask].cpu().numpy()
    classes = np.unique(y_train)
    if len(classes) > 1:
        try:
            class_weights_array = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train
            )
            class_weights = torch.tensor(class_weights_array, dtype=torch.float).to(device)
            for gnn_name in gnn_models.keys():
                criterions[gnn_name] = nn.CrossEntropyLoss(weight=class_weights)
        except Exception as e:
            print(f"Error computing class weights: {e}")
            return
    else:
        print("Only one class present in training data. Exiting.")
        return

    # 10. Define schedulers (skip NaiveBayes)
    from collections import defaultdict
    schedulers = {}
    for name, optimizer in optimizers.items():
        if name == "NaiveBayes":
            schedulers[name] = None
        else:
            schedulers[name] = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 11. Train / Fit each model
    best_val_losses = defaultdict(lambda: float('inf'))
    best_models = {}

    for name, model in models.items():
        if name == "NaiveBayes":
            print(f"\n=== Training {name} (single pass) ===")
            model = train_naive_bayes(model, data)
            val_acc = validate_naive_bayes(model, data)
            print(f"NaiveBayes Validation Accuracy: {val_acc:.4f}")
            best_models[name] = model
            continue

        # Otherwise, train GNN with epochs + early stopping
        print(f"\n=== Training {name} ===")
        optimizer = optimizers[name]
        criterion = criterions[name]
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, args.epochs + 1):
            try:
                loss = train(model, data, optimizer, criterion, epoch, edge_drop_prob=args.edge_drop_prob)
                val_loss = validate(model, data, criterion)
            except Exception as e:
                print(f"Error during training epoch {epoch} for {name}: {e}")
                break

            # Calc val acc
            model.eval()
            with torch.no_grad():
                try:
                    out = model(data.x, data.edge_index)
                    pred = out[data.val_mask].argmax(dim=1)
                    val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred.cpu())
                except Exception as e:
                    print(f"Error calculating validation accuracy for {name} at epoch {epoch}: {e}")
                    val_acc = 0.0

            print(f"Epoch {epoch:03d}/{args.epochs:03d}, "
                  f"Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping triggered for {name} at epoch {epoch}.")
                break

            schedulers[name].step(val_loss)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            best_models[name] = model

    # 12. Testing each model
    print("\n=== Testing Models ===")
    accuracies = {}
    for name, model in best_models.items():
        print(f"\n--- Testing {name} ---")
        try:
            if name == "NaiveBayes":
                acc = test_naive_bayes(model, data)
            else:
                acc = test(model, data)
            accuracies[name] = acc
        except Exception as e:
            print(f"Error testing model {name}: {e}")
            accuracies[name] = 0.0

    # 13. Display Accuracies
    print("\n=== Model Accuracies ===")
    for name, acc in accuracies.items():
        print(f"{name}: {acc:.4f}")

    # 14. Visualization (only for best GNN, ignoring NaiveBayes)
    if len(accuracies) > 0:
        # Find best among all models, or GNN-only, whichever you prefer
        best_model_name = max(accuracies, key=accuracies.get)
        best_model = best_models[best_model_name]
        if best_model_name != "NaiveBayes":
            print(f"\nVisualizing embeddings from the best model: {best_model_name}")
            try:
                visualize_embeddings(data, best_model, title=f"{best_model_name} Node Embeddings")
            except Exception as e:
                print(f"Error visualizing embeddings for {best_model_name}: {e}")
        else:
            print(f"\nBest model is NaiveBayes (no embedding visualization).")

    # 15. Analyze Misclassifications (skip or adapt for NaiveBayes)
    print("\n=== Analyzing Misclassifications ===")
    for name, model in best_models.items():
        if name == "NaiveBayes":
            continue
        try:
            analyze_misclassifications(model, data, name)
        except Exception as e:
            print(f"Error analyzing misclassifications for {name}: {e}")

    # 16. Ensemble (optional, typically for GNNs)
    print("\n=== Ensemble Method ===")
    try:
        # By default, ensemble_predictions is for GNNs returning logits.
        # If you want NB in an ensemble, you'd need special handling.
        gnn_only_dict = {k: v for k, v in best_models.items() if k != "NaiveBayes"}
        if not gnn_only_dict:
            print("No GNN models to ensemble.")
            return
        ensemble_preds = ensemble_predictions(gnn_only_dict, data)
        test_mask = data.test_mask.cpu().numpy()
        ensemble_acc = accuracy_score(data.y[test_mask].cpu(), ensemble_preds[test_mask])
        print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
        report = classification_report(data.y[test_mask].cpu(), ensemble_preds[test_mask], digits=4)
        print("Ensemble Classification Report:")
        print(report)
    except Exception as e:
        print(f"Error during ensemble prediction: {e}")


if __name__ == "__main__":
    main()
