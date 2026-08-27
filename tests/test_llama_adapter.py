import pytest
import torch
from torch import nn

from wound_forecasting.llama_adapter import (
    ADAPTER_DELTA_FORMAT,
    load_adapter_delta,
)
from wound_forecasting.llama_text_adapter import TextEncoder


class TinyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.adapter = nn.Linear(2, 2, bias=False)


def adapter_package(state=None, base_model="Emma02/LVM_ckpts"):
    if state is None:
        state = {"adapter.weight": torch.full((2, 2), 3.0)}
    return {
        "format": ADAPTER_DELTA_FORMAT,
        "model": state,
        "config": {"base_model": base_model},
    }


def test_load_adapter_delta_overlays_only_adapter_parameters():
    model = TinyAdapterModel()
    original_base = model.base.weight.detach().clone()

    package, result = load_adapter_delta(
        model,
        adapter_package(),
        expected_base_model="Emma02/LVM_ckpts",
    )

    assert package["format"] == ADAPTER_DELTA_FORMAT
    assert torch.equal(model.base.weight, original_base)
    assert torch.equal(model.adapter.weight, torch.full((2, 2), 3.0))
    assert result.unexpected_keys == []
    assert "base.weight" in result.missing_keys


def test_load_adapter_delta_rejects_wrong_base():
    with pytest.raises(ValueError, match="expects base model"):
        load_adapter_delta(
            TinyAdapterModel(),
            adapter_package(),
            expected_base_model="another/base",
        )


def test_load_adapter_delta_rejects_unknown_tensor():
    package = adapter_package({"missing.weight": torch.ones(2, 2)})
    with pytest.raises(KeyError, match="absent from the model"):
        load_adapter_delta(TinyAdapterModel(), package)


def test_load_adapter_delta_rejects_unknown_format():
    package = adapter_package()
    package["format"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported adapter format"):
        load_adapter_delta(TinyAdapterModel(), package)


def test_public_text_encoder_produces_adapter_queries():
    encoder = TextEncoder(
        vocabulary_size=100,
        embedding_dimension=16,
        sequence_length=8,
        number_of_layers=1,
        number_of_heads=2,
        adapter_sequence_length=4,
        padding_index=99,
        dropout=0.0,
    )
    tokens = torch.tensor([[1, 2, 3, 99, 99, 99, 99, 99]])
    output = encoder(tokens)
    assert output.shape == (1, 4, 16)
    assert torch.isfinite(output).all()
    state_keys = set(encoder.state_dict())
    assert "encoder_blocks.0.attn.attn.in_proj_weight" in state_keys
    assert "encoder_blocks.0.mlp.fc1.weight" in state_keys
    assert "encoder_blocks.0.mlp.fc2.weight" in state_keys
    assert "learned_query" in state_keys
