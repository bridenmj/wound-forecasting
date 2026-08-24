import math

import pytest
import torch

from wound_forecasting.metrics import compute_targetwise_psnr_ssim


def test_targetwise_metrics_count_unique_targets():
    real = {
        1: torch.zeros(2, 3, 16, 16),
        2: torch.zeros(1, 3, 16, 16),
    }
    fake = {
        1: torch.full((2, 3, 16, 16), 0.1),
        2: torch.full((1, 3, 16, 16), 0.2),
    }

    result = compute_targetwise_psnr_ssim(fake, real, device="cpu")

    assert result["per_horizon"][1]["n_targets"] == 2
    assert result["per_horizon"][2]["n_targets"] == 1
    assert result["overall"]["n_targets"] == 3
    expected_psnr = (20.0 + 20.0 + 20.0 * math.log10(5.0)) / 3.0
    assert result["overall"]["psnr"] == pytest.approx(expected_psnr, abs=1e-5)
    assert result["overall"]["aggregation"] == "mean_over_unique_targets"


def test_targetwise_metrics_reject_misaligned_pools():
    fake = {1: torch.zeros(2, 3, 16, 16)}
    real = {1: torch.zeros(1, 3, 16, 16)}

    with pytest.raises(ValueError, match="misaligned"):
        compute_targetwise_psnr_ssim(fake, real, device="cpu")

