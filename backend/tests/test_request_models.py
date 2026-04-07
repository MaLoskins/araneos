import pytest
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import ModelConfig, TrainRequest


class TestModelConfig:
    def test_valid_model_config(self):
        """Test that ModelConfig can be instantiated with valid parameters."""
        config = ModelConfig(
            model_name="GCN",
            hidden_channels=64,
            lr=0.01,
            epochs=200,
            dropout=0.5,
            extra_params={"num_layers": 2}
        )
        assert config.model_name == "GCN"
        assert config.hidden_channels == 64
        assert config.lr == 0.01
        assert config.epochs == 200
        assert config.dropout == 0.5
        assert config.extra_params == {"num_layers": 2}

    def test_valid_model_config_without_extra_params(self):
        """Test that ModelConfig can be instantiated without extra_params."""
        config = ModelConfig(
            model_name="GraphSAGE",
            hidden_channels=32,
            lr=0.001,
            epochs=100,
            dropout=0.2
        )
        assert config.model_name == "GraphSAGE"
        assert config.hidden_channels == 32
        assert config.lr == 0.001
        assert config.epochs == 100
        assert config.dropout == 0.2
        assert config.extra_params is None

    def test_model_config_validation_model_name(self):
        """Test that ModelConfig correctly validates model_name as str."""
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(
                model_name=123,
                hidden_channels=64,
                lr=0.01,
                epochs=200,
                dropout=0.5,
            )
        assert "Input should be a valid string" in str(excinfo.value)

    def test_model_config_validation_hidden_channels(self):
        """Test that ModelConfig correctly validates hidden_channels as int."""
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(
                model_name="GCN",
                hidden_channels="not-a-number",
                lr=0.01,
                epochs=200,
                dropout=0.5,
            )
        assert "Input should be a valid integer" in str(excinfo.value)

    def test_model_config_validation_lr(self):
        """Test that ModelConfig correctly validates lr as float."""
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(
                model_name="GCN",
                hidden_channels=64,
                lr="not-a-number",
                epochs=200,
                dropout=0.5,
            )
        assert "Input should be a valid number" in str(excinfo.value)

    def test_model_config_validation_epochs(self):
        """Test that ModelConfig correctly validates epochs as int."""
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(
                model_name="GCN",
                hidden_channels=64,
                lr=0.01,
                epochs=100.5,
                dropout=0.5,
            )
        assert "Input should be a valid integer" in str(excinfo.value)

    def test_model_config_validation_dropout(self):
        """Test that ModelConfig correctly validates dropout as float."""
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(
                model_name="GCN",
                hidden_channels=64,
                lr=0.01,
                epochs=200,
                dropout="not-a-number",
            )
        assert "Input should be a valid number" in str(excinfo.value)

    def test_model_config_validation_extra_params(self):
        """Test that ModelConfig correctly validates extra_params as dict."""
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(
                model_name="GCN",
                hidden_channels=64,
                lr=0.01,
                epochs=200,
                dropout=0.5,
                extra_params="not_a_dict"
            )
        assert "Input should be a valid dictionary" in str(excinfo.value)


class TestTrainRequest:
    def test_valid_train_request(self):
        """Test that TrainRequest can be instantiated with valid parameters."""
        config = ModelConfig(
            model_name="GCN",
            hidden_channels=64,
            lr=0.01,
            epochs=200,
            dropout=0.5
        )

        request = TrainRequest(
            session_id="abc12345",
            configuration=config
        )

        assert request.session_id == "abc12345"
        assert request.configuration == config
        assert request.configuration.model_name == "GCN"

    def test_train_request_with_dict_model_config(self):
        """Test that TrainRequest can be instantiated with a dict for configuration."""
        model_config_dict = {
            "model_name": "GraphSAGE",
            "hidden_channels": 32,
            "lr": 0.001,
            "epochs": 100,
            "dropout": 0.2,
            "extra_params": {"aggregation": "mean"}
        }

        request = TrainRequest(
            session_id="abc12345",
            configuration=model_config_dict
        )

        assert request.session_id == "abc12345"
        assert request.configuration.model_name == "GraphSAGE"
        assert request.configuration.extra_params == {"aggregation": "mean"}

    def test_train_request_missing_session_id(self):
        """Test that TrainRequest validates session_id is required."""
        model_config = {
            "model_name": "GCN",
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5
        }

        with pytest.raises(ValidationError):
            TrainRequest(configuration=model_config)

    def test_train_request_invalid_model_config(self):
        """Test that TrainRequest validates the configuration."""
        invalid_model_config = {
            "model_name": 123,
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5
        }

        with pytest.raises(ValidationError) as excinfo:
            TrainRequest(session_id="abc12345", configuration=invalid_model_config)

        assert "Input should be a valid string" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
