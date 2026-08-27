import pytest
import torch
from torch import nn

from wound_forecasting.river import RiverVectorFieldRegressor, load_river_weights
from wound_forecasting.vqmuse_upstream import VQMuseAutoencoder


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


class TinyVQ(nn.Module):
    def encode(self, values):
        return values + 1, torch.tensor(0.0)

    def decode(self, values):
        return values - 1

    def quantize(self, values, return_loss=False):
        assert return_loss is False
        return values.round(), None, None


def test_public_vqmuse_wrapper_exposes_river_interface():
    wrapper = VQMuseAutoencoder(TinyVQ())
    values = torch.tensor([0.25, 1.75])
    assert torch.equal(wrapper.encode(values), values + 1)
    assert torch.equal(wrapper.decode(values), values - 1)
    assert torch.equal(wrapper.quantize(values), values.round())
