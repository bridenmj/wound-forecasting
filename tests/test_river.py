import pytest
import torch
from torch import nn

from wound_forecasting.river import load_river_weights


class TinyRiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = nn.Linear(2, 2, bias=False)
        self.vector_field_regressor = nn.Linear(2, 2, bias=False)


def test_load_river_only_weights_keeps_external_autoencoder():
    model = TinyRiver()
    autoencoder = model.ae.weight.detach().clone()
    package = {
        "model": {
            "vector_field_regressor.weight": torch.full((2, 2), 4.0),
        }
    }
    _, result = load_river_weights(model, package)
    assert torch.equal(model.ae.weight, autoencoder)
    assert torch.equal(model.vector_field_regressor.weight, torch.full((2, 2), 4.0))
    assert "ae.weight" in result.missing_keys


def test_load_river_weights_rejects_unknown_key():
    with pytest.raises(KeyError, match="unknown tensors"):
        load_river_weights(TinyRiver(), {"model": {"wrong": torch.ones(1)}})
