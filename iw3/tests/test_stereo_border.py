import pytest
import torch

from iw3.backward_warp import apply_divergence_nn_LR, apply_divergence_nn_symmetric


class DirectParallax(torch.nn.Module):
    name = "sbs.row_flow_v3"
    delta_output = True

    def __init__(self, symmetric: bool) -> None:
        super().__init__()
        self.symmetric = symmetric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = -(x[:, :1] * x[:, 1:2] + x[:, 2:3]) * 32
        return torch.cat([delta, torch.zeros_like(delta)], dim=1)


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("synthetic_view", ["both", "left", "right"])
def test_preserve_screen_border_keeps_outer_pixels(symmetric: bool, synthetic_view: str) -> None:
    # RGB[1,3,8,128] + depth[1,1,8,128] → 特征[1,3,8,128] → delta[1,2,8,128] → 同尺寸双眼。
    rgb = torch.linspace(0, 1, 128).view(1, 1, 1, 128).expand(1, 3, 8, 128)
    depth = torch.zeros(1, 1, 8, 128)
    model = DirectParallax(symmetric)
    displaced = apply_divergence_nn_LR(
        model, rgb, depth, 5.0, 1.0, None, synthetic_view=synthetic_view, enable_amp=False
    )
    preserved = apply_divergence_nn_LR(
        model,
        rgb,
        depth,
        5.0,
        1.0,
        None,
        synthetic_view=synthetic_view,
        preserve_screen_border=True,
        enable_amp=False,
    )
    for eye in preserved:
        assert eye.shape == rgb.shape and torch.isfinite(eye).all()
        torch.testing.assert_close(eye[..., 0], torch.zeros(1, 3, 8))
        torch.testing.assert_close(eye[..., -1], torch.ones(1, 3, 8))
    if synthetic_view in {"both", "left"}:
        assert float(displaced[0][..., 0].min()) > 0.01
    if synthetic_view in {"both", "right"}:
        assert float(displaced[1][..., -1].max()) < 0.99


def test_symmetric_old_positional_call_keeps_border_displacement_default() -> None:
    rgb = torch.linspace(0, 1, 128).view(1, 1, 1, 128).expand(1, 3, 8, 128)
    depth = torch.zeros(1, 1, 8, 128)
    left, right = apply_divergence_nn_symmetric(DirectParallax(True), rgb, depth, 5.0, 1.0, "both", False)
    assert float(left[..., 0].min()) > 0.01
    assert float(right[..., -1].max()) < 0.99
