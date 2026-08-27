import pytest
import torch
from torch import nn

from wound_forecasting.river import RiverVectorFieldRegressor, load_river_weights


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


def test_final_river_vector_field_matches_released_checkpoint_structure():
    model = RiverVectorFieldRegressor(
        depth=2,
        mid_depth=3,
        state_size=64,
        state_res=(16, 16),
        inner_dim=384,
        out_norm="ln",
        dropout=0.075,
    )
    state = model.state_dict()
    assert len(state) == 102
    assert "position_encoding.row_embed.weight" in state
    assert state["project_in.1.weight"].shape == (384, 256)
    assert state["project_out.4.weight"].shape == (64, 384, 3, 3)
