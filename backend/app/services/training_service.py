import logging
import torch

from pipeline.torch_geometric_builder import (
    GCNModel, GraphSageModel, GATModel, GINModel, ChebConvModel, ResidualGCNModel,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {'GCN', 'GRAPHSAGE', 'SAGE', 'GAT', 'GIN', 'CHEBCONV', 'CHEB', 'RESIDUALGCN', 'RESGCN'}


def create_model(name: str, in_ch: int, hidden_ch: int, num_cls: int, dropout: float, extra: dict):
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


def run_epoch(model, data, optimizer, criterion, scheduler, best_val_loss):
    """Run a single training epoch. Returns (train_loss, val_loss, val_acc, is_best)."""
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
    return train_loss, val_loss, val_acc, is_best


def run_test(model, data):
    """Run final test evaluation. Returns test accuracy."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_pred = out[data.test_mask].argmax(dim=1)
        test_acc = (test_pred == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return test_acc


def get_device():
    """Detect best available device."""
    device = torch.device('cpu')
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device='cuda')
            device = torch.device('cuda')
        except Exception:
            logger.warning("CUDA available but unusable (kernel mismatch), falling back to CPU")
    return device
