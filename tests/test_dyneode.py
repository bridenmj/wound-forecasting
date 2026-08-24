import pytest
import torch

from wound_forecasting.dyneode import ContextConditionedODEFunc, collapse_broadcast_w


def test_collapse_broadcast_w():
    values = torch.randn(4, 512)
    broadcast = values[:, None, :].expand(4, 14, 512).clone()
    assert torch.equal(collapse_broadcast_w(broadcast), values)


def test_conditioned_field_supports_variable_context():
    field = ContextConditionedODEFunc(dim=8, hidden_dim=16, context_hidden_dim=4)
    context = torch.randn(5, 8)
    times = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0])
    encoded = field.encode_context(context, times)
    assert encoded.shape == (8,)
    assert field(torch.tensor(1.0), context[-1], encoded).shape == (8,)


def test_context_times_must_increase():
    field = ContextConditionedODEFunc(dim=8, hidden_dim=16, context_hidden_dim=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        field.encode_context(torch.randn(2, 8), torch.tensor([0.0, 0.0]))

