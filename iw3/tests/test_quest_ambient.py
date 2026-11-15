import importlib
import math
import struct
import zlib
from typing import Any

import pytest
import torch


def background() -> Any:
    return importlib.import_module("iw3.desktop.quest_ambient").AmbientBackground()


def pixels(payload: bytes | None) -> torch.Tensor:
    assert payload is not None
    length = payload[0]
    assert payload[1 : 1 + length] == b"iw3_ambient"
    assert struct.unpack("!HHB", payload[1 + length : 6 + length]) == (160, 90, 1)
    raw = bytearray(zlib.decompress(payload[6 + length :]))
    return torch.frombuffer(raw, dtype=torch.uint8).reshape(90, 160, 4)


def solid(red: float, green: float, blue: float) -> torch.Tensor:
    return torch.tensor([red, green, blue], dtype=torch.float32).view(3, 1, 1).expand(3, 180, 320)


def test_rgba_keeps_each_edge_on_its_source_side() -> None:
    frame = torch.zeros(3, 180, 320)
    frame[0, :30] = 1
    frame[1, -30:] = 1
    frame[:, :, :30] = torch.tensor([0, 0, 1]).view(3, 1, 1)
    frame[:, :, -30:] = torch.tensor([1, 1, 0]).view(3, 1, 1)
    rgb = pixels(background().update(frame, 1, tick=0))[:, :, :3].float()
    top, bottom, left, right = rgb[12, 80], rgb[77, 80], rgb[45, 20], rgb[45, 140]
    assert top[0] > top[1] + 40 and top[0] > top[2] + 40
    assert bottom[1] > bottom[0] + 40 and bottom[1] > bottom[2] + 40
    assert left[2] > left[0] + 40 and left[2] > left[1] + 40
    assert right[0] > right[2] + 40 and right[1] > right[2] + 40


def test_alpha_fades_to_zero_at_every_outer_edge() -> None:
    rgba = pixels(background().update(torch.ones(3, 90, 160), 1, tick=0))
    alpha = rgba[:, :, 3]
    assert torch.count_nonzero(alpha[0]) == torch.count_nonzero(alpha[-1]) == 0
    assert torch.count_nonzero(alpha[:, 0]) == torch.count_nonzero(alpha[:, -1]) == 0
    assert 0 < alpha[45, 10] < alpha[45, 30] <= alpha[45, 80] == 255
    assert torch.isfinite(rgba.float()).all()


@pytest.mark.parametrize("brightness,expected", [(0, 0), (0.25, 64), (1, 255)])
def test_brightness_scales_rgb_without_changing_alpha(brightness: float, expected: int) -> None:
    rgba = pixels(background().update(torch.ones(3, 90, 160), brightness, tick=0))
    assert torch.all(rgba[:, :, :3] == expected)
    assert rgba[45, 80, 3] == 255


def test_ten_hz_limit_does_not_feed_skipped_frames_into_smoothing() -> None:
    ambient = background()
    pixels(ambient.update(solid(1, 0, 0), 1, tick=0))
    assert ambient.update(solid(0, 1, 0), 1, tick=0.099) is None
    rgba = pixels(ambient.update(solid(0, 0, 1), 1, tick=0.1))
    assert rgba[45, 80, 1] == 0
    assert abs(int(rgba[45, 80, 0]) - 171) <= 1
    assert abs(int(rgba[45, 80, 2]) - 84) <= 1
    assert ambient.update(solid(0, 0, 1), 1, tick=0.199) is None
    assert ambient.update(solid(0, 0, 1), 1, tick=0.2) is not None


def test_smoothing_uses_elapsed_time_and_reset_forgets_old_color() -> None:
    ambient = background()
    pixels(ambient.update(solid(1, 0, 0), 1, tick=10))
    rgba = pixels(ambient.update(solid(0, 0, 1), 1, tick=10.25))
    assert abs(int(rgba[45, 80, 0]) - 94) <= 1
    assert abs(int(rgba[45, 80, 2]) - 161) <= 1
    ambient.reset()
    rgba = pixels(ambient.update(solid(0, 0, 1), 1, tick=10.251))
    assert tuple(rgba[45, 80, :3].tolist()) == (0, 0, 255)


def test_ten_hz_schedule_is_not_rejected_by_float_rounding() -> None:
    ambient = background()
    for tick in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        assert ambient.update(solid(1, 0, 0), 1, tick=tick) is not None


def test_brightness_change_applies_without_color_history_delay() -> None:
    ambient = background()
    pixels(ambient.update(solid(1, 1, 1), 1, tick=0))
    rgba = pixels(ambient.update(solid(1, 1, 1), 0, tick=0.1))
    assert torch.count_nonzero(rgba[:, :, :3]) == 0


def test_capture_padding_is_excluded_only_when_content_rect_is_given() -> None:
    frame = torch.zeros(3, 180, 320)
    frame[:, 40:140, 60:260] = torch.tensor([0.2, 0.6, 1]).view(3, 1, 1)
    with_rect = background().update(frame, 1, (60, 40, 260, 140), tick=0)
    cropped = background().update(frame[:, 40:140, 60:260], 1, tick=0)
    assert with_rect == cropped
    assert pixels(with_rect)[45, 80, 2] == 255
    assert pixels(background().update(frame, 1, tick=0))[45, 80, 2] == 0


@pytest.mark.parametrize(
    "frame",
    [
        torch.zeros(1, 90, 160),
        torch.zeros(3, 0, 160),
        torch.zeros(1, 3, 90, 160),
        torch.zeros(3, 90, 160, dtype=torch.uint8),
        torch.full((3, 90, 160), math.nan),
        torch.full((3, 90, 160), math.inf),
        torch.full((3, 90, 160), -0.01),
        torch.full((3, 90, 160), 1.01),
    ],
)
def test_invalid_frame_is_rejected(frame: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        background().update(frame, 0.25, tick=0)


@pytest.mark.parametrize("brightness", [True, -0.01, 1.01, math.nan, math.inf, "0.25"])
def test_invalid_brightness_is_rejected(brightness: Any) -> None:
    with pytest.raises(ValueError):
        background().update(torch.zeros(3, 90, 160), brightness, tick=0)


@pytest.mark.parametrize(
    "rect",
    [
        (-1, 0, 100, 90),
        (0, 0, 161, 90),
        (0, 0, 160, 91),
        (20, 0, 20, 90),
        (0, 10, 160, 9),
        (0, 0, 160),
        (False, 0, 160, 90),
    ],
)
def test_invalid_content_rect_is_rejected(rect: Any) -> None:
    with pytest.raises(ValueError):
        background().update(torch.zeros(3, 90, 160), 0.25, rect, tick=0)


@pytest.mark.parametrize("tick", [math.nan, math.inf, True, "0"])
def test_invalid_tick_is_rejected(tick: Any) -> None:
    with pytest.raises(ValueError):
        background().update(torch.zeros(3, 90, 160), 0.25, tick=tick)


def test_update_detaches_input_without_changing_its_pixels() -> None:
    frame = solid(0.2, 0.4, 0.6).clone().requires_grad_()
    before = frame.detach().clone()
    pixels(background().update(frame, 0.25, tick=0))
    assert torch.equal(frame, before) and frame.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_frame_matches_cpu_texture() -> None:
    frame = solid(0.2, 0.6, 1).clone()
    frame[:, :40] = torch.tensor([1, 0, 0]).view(3, 1, 1)
    cpu = pixels(background().update(frame, 0.25, tick=0))
    cuda = pixels(background().update(frame.cuda(), 0.25, tick=0))
    assert torch.max(torch.abs(cpu.to(torch.int16) - cuda.to(torch.int16))) <= 1
