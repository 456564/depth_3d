import hashlib

import pytest
import torch

from iw3.desktop.quest_ambient import AmbientBackground

from .test_quest_ambient import pixels, solid
from .test_quest_feather import alpha_at, frame, settled

STYLES = ("edge", "blur", "halo", "cinema", "aurora", "mist", "stars")


def test_paused_style_change_preserves_settled_embedded_black_borders():
    image = frame((132, 0, 252, 216))
    stereo = torch.cat((image, image), 2)
    ambient = AmbientBackground()
    for tick in (0, 0.1, 0.2, 0.3):
        initial = pixels(ambient.update(image, 0.67, tick=tick, sbs=stereo))
    assert alpha_at(initial, 0.1, 0.5) == 255
    for tick, style in enumerate(("blur", "halo", "cinema", "edge"), start=1):
        changed = pixels(ambient.update(image, 0.67, tick=0.3 + tick * 0.001, sbs=stereo, style=style))
        assert alpha_at(changed, 0.1, 0.5) == 255
        assert changed[90, 160, 3] == 0


def gradient(height=90, width=160):
    x = torch.linspace(0, 1, width)[None, :].expand(height, width)
    y = torch.linspace(0, 1, height)[:, None].expand(height, width)
    return torch.stack((0.2 + 0.7 * x, 0.15 + 0.6 * y, 0.75 - 0.3 * x))


def test_default_and_explicit_edge_preserve_accepted_landscape_pixels():
    image = gradient()
    stereo = torch.cat((image, image), 2)
    implicit = pixels(AmbientBackground().update(image, 0.67, tick=0, sbs=stereo))
    explicit = pixels(AmbientBackground().update(image, 0.67, tick=0, sbs=stereo, style="edge"))
    assert torch.equal(implicit, explicit)
    assert hashlib.sha256(implicit.numpy().tobytes()).hexdigest() == (
        "48360a1c96dac8983c0ce21f6e9047c84f2f8c97ede79e74cc1b46ea6dbec2bc"
    )


def test_default_edge_preserves_accepted_portrait_pixels():
    image = torch.tensor([0.2, 0.6, 1.0])[:, None, None].expand(3, 64, 36)
    rgba = pixels(
        AmbientBackground().update(
            image,
            0.67,
            tick=0,
            sbs=torch.cat((image, image), 2),
            video_rect=(0.341796875, 0, 0.658203125, 1),
            scale=(1.7, 4),
        )
    )
    assert hashlib.sha256(rgba.numpy().tobytes()).hexdigest() == (
        "59ba852926f7184a3534495b696b98ffafc04a73c3bd6ed0dc5d349f4fd6ea76"
    )


@pytest.mark.parametrize("style", ["", "unknown", "EDGE", None, 1, True, [], {}])
def test_invalid_style_is_rejected_at_the_renderer_boundary(style):
    ambient = AmbientBackground()
    with pytest.raises(ValueError, match="style"):
        ambient.update(gradient(), 0.5, tick=0, style=style)
    assert ambient._last_tick is None and ambient._colors is None


@pytest.mark.parametrize("style", STYLES)
def test_styles_preserve_clear_subject_and_fill_portrait_reference_gaps(style):
    image = gradient(64, 36)
    before = image.clone()
    rgba = pixels(
        AmbientBackground().update(
            image,
            0.67,
            tick=0,
            sbs=torch.cat((image, image), 2),
            video_rect=(0.341796875, 0, 0.658203125, 1),
            style=style,
        )
    )
    assert rgba.shape == (180, 320, 4) and rgba.dtype == torch.uint8
    assert rgba[90, 160, 3] == 0
    assert rgba[90, 96, 3] == rgba[90, 223, 3] == 255
    assert torch.count_nonzero(rgba[0, :, 3]) == torch.count_nonzero(rgba[-1, :, 3]) == 0
    assert torch.count_nonzero(rgba[:, 0, 3]) == torch.count_nonzero(rgba[:, -1, 3]) == 0
    assert torch.equal(image, before) and image.grad is None


@pytest.mark.parametrize("style", STYLES)
def test_styles_keep_final_sbs_subtitles_protected(style):
    display = frame()
    display[:, 190:198, 18:30] = 1
    display[:, 190:198, 354:366] = 1
    display[:, 107, 24] = 1
    rgba = settled(AmbientBackground(), frame(), sbs=torch.cat((display, display), 2), style=style)
    assert alpha_at(rgba, 24 / 384, 194 / 216) == alpha_at(rgba, 24 / 384, 107 / 216) == 0


def test_blur_uses_interior_content_rather_than_only_four_edges():
    plain = torch.full((3, 90, 160), 0.2)
    interior = plain.clone()
    interior[0, 18:72, 32:128] = 1
    edge = pixels(AmbientBackground().update(interior, 1, tick=0, fill_borders=False, style="edge"))
    blur = pixels(AmbientBackground().update(interior, 1, tick=0, fill_borders=False, style="blur"))
    assert blur[90, 60, 0] > edge[90, 60, 0] + 15
    assert blur[90, 60, 3] > 0
    assert torch.equal(blur[:, :, 3], edge[:, :, 3])


def test_halo_retains_direction_but_fades_faster_only_outside_reference_screen():
    image = gradient()
    edge = pixels(AmbientBackground().update(image, 1, tick=0, style="edge"))
    halo = pixels(AmbientBackground().update(image, 1, tick=0, style="halo"))
    assert torch.equal(edge[:, :, :3], halo[:, :, :3])
    assert 0 < halo[90, 40, 3] < edge[90, 40, 3]
    assert halo[90, 40, 2] > halo[90, 40, 0]
    assert halo[90, 280, 0] > halo[90, 280, 2]


def test_cinema_is_low_brightness_average_color_without_directional_gradient():
    image = torch.zeros(3, 90, 160)
    image[0, :, :80] = 1
    image[2, :, 80:] = 1
    rgba = pixels(AmbientBackground().update(image, 1, tick=0, fill_borders=False, style="cinema"))
    assert rgba[90, 40, :3].tolist() == pytest.approx((45, 0, 45), abs=1)
    assert torch.equal(rgba[:, :, :3], rgba[90, 40, :3].expand(180, 320, 3))
    assert rgba[90, 40, 3] > 0 and rgba[90, 160, 3] == 0


@pytest.mark.parametrize("style", STYLES)
def test_each_style_keeps_ten_hz_and_color_smoothing(style):
    ambient = AmbientBackground()
    ambient.update(solid(1, 0, 0), 1, tick=0, style=style)
    assert ambient.update(solid(0, 1, 0), 1, tick=0.099, style=style) is None
    rgba = pixels(ambient.update(solid(0, 0, 1), 1, tick=0.1, style=style))
    assert rgba[90, 160, 0] > rgba[90, 160, 2] > 0
    assert rgba[90, 160, 1] == 0


@pytest.mark.parametrize("style", ("blur", "halo", "cinema"))
def test_style_switch_bypasses_previous_rate_gate_and_forgets_old_colors(style):
    ambient = AmbientBackground()
    ambient.update(solid(1, 0, 0), 1, tick=0, style="edge")
    changed = ambient.update(solid(0, 0, 1), 1, tick=0.01, style=style)
    fresh = AmbientBackground().update(solid(0, 0, 1), 1, tick=0.01, style=style)
    assert changed == fresh and changed is not None
    assert ambient.update(solid(0, 1, 0), 1, tick=0.05, style=style) is None
