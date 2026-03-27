import pytest
import torch

from boltz.model.loss.diffusion import weighted_rigid_align as weighted_rigid_align_v1
from boltz.model.loss.diffusionv2 import weighted_rigid_align as weighted_rigid_align_v2


@pytest.mark.parametrize(
    "align_fn",
    [weighted_rigid_align_v1, weighted_rigid_align_v2],
)
def test_weighted_rigid_align_falls_back_when_svd_fails(monkeypatch, align_fn):
    true_coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]]
    )
    pred_coords = true_coords + 3.0
    weights = torch.ones((1, 4))
    mask = torch.ones((1, 4))

    def failing_svd(*args, **kwargs):
        raise RuntimeError("svd failed to converge")

    monkeypatch.setattr(torch.linalg, "svd", failing_svd)

    aligned = align_fn(true_coords, pred_coords, weights, mask)

    assert aligned.shape == true_coords.shape
    assert torch.isfinite(aligned).all()
    assert torch.allclose(aligned.mean(dim=1), pred_coords.mean(dim=1))
