# test_train_endpoint.py
import pytest
import json
import torch
from unittest.mock import MagicMock, patch
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app, TrainRequest
from app.dependencies import get_session_store
from app.storage.memory import InMemorySessionStore


@pytest.fixture
def session_store():
    """Create a fresh session store for each test."""
    store = InMemorySessionStore(max_sessions=50)
    return store


@pytest.fixture
def client(session_store):
    """TestClient with overridden session store dependency."""
    app.dependency_overrides[get_session_store] = lambda: session_store
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def valid_graph_data():
    return {
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}},
            {"source": {"id": "2"}, "target": {"id": "3"}},
            {"source": {"id": "1"}, "target": {"id": "3"}}
        ],
        "nodes": [
            {"id": "1", "features": {"label": "A", "text_embedding": [0.1, 0.2, 0.3], "user_followers_count_feature": "5"}},
            {"id": "2", "features": {"label": "B", "text_embedding": [0.4, 0.5, 0.6], "user_followers_count_feature": "10"}},
            {"id": "3", "features": {"label": "A", "text_embedding": [0.7, 0.8, 0.9], "user_followers_count_feature": "15"}}
        ]
    }


@pytest.fixture
def no_labels_graph_data():
    return {
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}},
            {"source": {"id": "2"}, "target": {"id": "3"}},
            {"source": {"id": "1"}, "target": {"id": "3"}}
        ],
        "nodes": [
            {"id": "1", "features": {"text_embedding": [0.1, 0.2, 0.3], "user_followers_count_feature": "5"}},
            {"id": "2", "features": {"text_embedding": [0.4, 0.5, 0.6], "user_followers_count_feature": "10"}},
            {"id": "3", "features": {"text_embedding": [0.7, 0.8, 0.9], "user_followers_count_feature": "15"}}
        ]
    }


@pytest.fixture
def single_class_graph_data():
    return {
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}},
            {"source": {"id": "2"}, "target": {"id": "3"}},
            {"source": {"id": "1"}, "target": {"id": "3"}}
        ],
        "nodes": [
            {"id": "1", "features": {"label": "A", "text_embedding": [0.1, 0.2, 0.3], "user_followers_count_feature": "5"}},
            {"id": "2", "features": {"label": "A", "text_embedding": [0.4, 0.5, 0.6], "user_followers_count_feature": "10"}},
            {"id": "3", "features": {"label": "A", "text_embedding": [0.7, 0.8, 0.9], "user_followers_count_feature": "15"}}
        ]
    }


def _store(session_store, graph_data):
    """Store graph data and return session_id."""
    return session_store.store(graph_data)


@pytest.fixture
def valid_model_config():
    return {
        "model_name": "GCN",
        "hidden_channels": 64,
        "dropout": 0.3,
        "lr": 0.01,
        "epochs": 5,
    }


# --- TestClient-based tests (HTTP layer, Depends resolved by FastAPI) ---

def test_train_gnn_missing_labels(client, session_store, no_labels_graph_data, valid_model_config):
    session_id = _store(session_store, no_labels_graph_data)
    response = client.post("/train-gnn", json={"session_id": session_id, "configuration": valid_model_config})
    assert response.status_code == 400
    assert "no labels" in response.json()["detail"].lower()


def test_train_gnn_invalid_model(client, session_store, valid_graph_data, valid_model_config):
    session_id = _store(session_store, valid_graph_data)
    invalid_config = valid_model_config.copy()
    invalid_config["model_name"] = "INVALID_MODEL"

    with patch("app.routes.training.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("app.routes.training.split_data") as mock_split_data:

        mock_instance = MockBuilder.return_value
        mock_data = MagicMock()
        mock_data.y = torch.tensor([0, 1])
        mock_instance.build_data.return_value = mock_data
        mock_split_data.return_value = mock_data

        response = client.post("/train-gnn", json={"session_id": session_id, "configuration": invalid_config})

    assert response.status_code == 400
    assert "unsupported model" in response.json()["detail"].lower()


def test_train_gnn_single_class(client, session_store, single_class_graph_data, valid_model_config):
    session_id = _store(session_store, single_class_graph_data)

    with patch("app.routes.training.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("torch.unique", return_value=torch.tensor([0])):

        mock_instance = MockBuilder.return_value
        mock_data = MagicMock()
        mock_data.y = torch.tensor([0, 0, 0])
        mock_instance.build_data.return_value = mock_data

        response = client.post("/train-gnn", json={"session_id": session_id, "configuration": valid_model_config})

    assert response.status_code == 400
    assert "2 classes" in response.json()["detail"].lower()


def test_train_gnn_graph_building_error(client, session_store, valid_graph_data, valid_model_config):
    session_id = _store(session_store, valid_graph_data)

    with patch("app.routes.training.TorchGeometricGraphBuilder") as MockBuilder:
        mock_instance = MockBuilder.return_value
        mock_instance.build_data.side_effect = ValueError("Error building graph: invalid node feature")

        response = client.post("/train-gnn", json={"session_id": session_id, "configuration": valid_model_config})

    assert response.status_code == 500
    assert "error building graph" in response.json()["detail"].lower()


def test_train_gnn_session_not_found(client, valid_model_config):
    response = client.post("/train-gnn", json={"session_id": "nonexistent", "configuration": valid_model_config})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# --- Async tests that call train_gnn directly (pass session_store explicitly) ---

@pytest.mark.asyncio
async def test_train_gnn_gcn_model(session_store, valid_graph_data):
    from app.routes.training import train_gnn

    session_id = _store(session_store, valid_graph_data)
    config = {"model_name": "GCN", "hidden_channels": 16, "dropout": 0.2, "lr": 0.01, "epochs": 2}
    request = TrainRequest(session_id=session_id, configuration=config)

    mock_data = MagicMock()
    mock_data.y = torch.tensor([0, 1])
    mock_data.num_node_features = 4

    with patch("app.routes.training.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("app.routes.training.training_service.create_model") as MockCreate, \
         patch("app.routes.training.split_data", return_value=mock_data), \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("torch.optim.Adam"), \
         patch("torch.nn.CrossEntropyLoss"), \
         patch("torch.optim.lr_scheduler.ReduceLROnPlateau"), \
         patch("torch.cuda.is_available", return_value=False):

        MockBuilder.return_value.build_data.return_value = mock_data
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        MockCreate.return_value = mock_model

        response = await train_gnn(request, session_store=session_store)
        assert isinstance(response, StreamingResponse)
        MockCreate.assert_called_once()


@pytest.mark.asyncio
async def test_train_gnn_graphsage_model(session_store, valid_graph_data):
    from app.routes.training import train_gnn

    session_id = _store(session_store, valid_graph_data)
    config = {"model_name": "GraphSAGE", "hidden_channels": 16, "dropout": 0.2, "lr": 0.01, "epochs": 2}
    request = TrainRequest(session_id=session_id, configuration=config)

    mock_data = MagicMock()
    mock_data.y = torch.tensor([0, 1])
    mock_data.num_node_features = 4

    with patch("app.routes.training.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("app.routes.training.training_service.create_model") as MockCreate, \
         patch("app.routes.training.split_data", return_value=mock_data), \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("torch.optim.Adam"), \
         patch("torch.nn.CrossEntropyLoss"), \
         patch("torch.optim.lr_scheduler.ReduceLROnPlateau"), \
         patch("torch.cuda.is_available", return_value=False):

        MockBuilder.return_value.build_data.return_value = mock_data
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        MockCreate.return_value = mock_model

        response = await train_gnn(request, session_store=session_store)
        assert isinstance(response, StreamingResponse)
        MockCreate.assert_called_once()


@pytest.mark.asyncio
async def test_train_gnn_gat_model_with_extra_params(session_store, valid_graph_data):
    from app.routes.training import train_gnn

    session_id = _store(session_store, valid_graph_data)
    config = {"model_name": "GAT", "hidden_channels": 16, "dropout": 0.2, "lr": 0.01, "epochs": 2, "extra_params": {"heads": 4}}
    request = TrainRequest(session_id=session_id, configuration=config)

    mock_data = MagicMock()
    mock_data.y = torch.tensor([0, 1])
    mock_data.num_node_features = 4

    with patch("app.routes.training.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("app.routes.training.training_service.create_model") as MockCreate, \
         patch("app.routes.training.split_data", return_value=mock_data), \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("torch.optim.Adam"), \
         patch("torch.nn.CrossEntropyLoss"), \
         patch("torch.optim.lr_scheduler.ReduceLROnPlateau"), \
         patch("torch.cuda.is_available", return_value=False):

        MockBuilder.return_value.build_data.return_value = mock_data
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        MockCreate.return_value = mock_model

        response = await train_gnn(request, session_store=session_store)
        assert isinstance(response, StreamingResponse)
        MockCreate.assert_called_once()


@pytest.mark.asyncio
async def test_train_gnn_streaming_response(session_store, valid_graph_data):
    from app.routes.training import train_gnn

    session_id = _store(session_store, valid_graph_data)
    config = {"model_name": "GCN", "hidden_channels": 16, "dropout": 0.2, "lr": 0.01, "epochs": 2}
    request = TrainRequest(session_id=session_id, configuration=config)

    async def mock_training_stream():
        yield json.dumps({"status": "started", "message": "Training GCN model"}) + "\n"
        yield json.dumps({"epoch": 1, "train_loss": 0.5, "val_loss": 0.4}) + "\n"
        yield json.dumps({"epoch": 2, "train_loss": 0.3, "val_loss": 0.2}) + "\n"
        yield json.dumps({"status": "completed", "test_accuracy": 0.85}) + "\n"

    mock_data = MagicMock()
    mock_data.y = torch.tensor([0, 1])
    mock_data.num_node_features = 4

    with patch("app.routes.training.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("app.routes.training.training_service.create_model") as MockCreate, \
         patch("app.routes.training.split_data", return_value=mock_data), \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("torch.optim.Adam"), \
         patch("torch.nn.CrossEntropyLoss"), \
         patch("torch.optim.lr_scheduler.ReduceLROnPlateau"), \
         patch("torch.cuda.is_available", return_value=False):

        MockBuilder.return_value.build_data.return_value = mock_data
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        MockCreate.return_value = mock_model

        with patch("app.routes.training.StreamingResponse", return_value=StreamingResponse(mock_training_stream())):
            response = await train_gnn(request, session_store=session_store)

            collected_data = []
            async for chunk in response.body_iterator:
                chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                collected_data.append(json.loads(chunk_str.strip()))

            assert len(collected_data) == 4
            assert collected_data[0]["status"] == "started"
            assert "epoch" in collected_data[1]
            assert collected_data[3]["status"] == "completed"
            assert "test_accuracy" in collected_data[3]
