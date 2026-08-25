import pytest
import torch
from torch import nn

from wound_forecasting.llama_adapter import (
    ADAPTER_DELTA_FORMAT,
    load_adapter_delta,
)


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
