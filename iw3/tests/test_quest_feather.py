import importlib
import math
import struct
import zlib

import pytest
import torch


def background():
    return importlib.import_module("iw3.desktop.quest_ambient").AmbientBackground()


def pixels(payload):
    assert payload is not None
    length = payload[0]
    assert struct.unpack("!HHB", payload[1 + length : 6 + length]) == (320, 180, 1)
    return torch.frombuffer(bytearray(zlib.decompress(payload[6 + length :])), dtype=torch.uint8).reshape(180, 320, 4)


def frame(rect=(132, 0, 252, 216), noise=False):
    image = torch.zeros(3, 216, 384)
    if noise:
        image.uniform_(0, 7 / 255, generator=torch.Generator().manual_seed(12))
    left, top, right, bottom = rect
    image[:, top:bottom, left:right] = torch.tensor([0.2, 0.6, 1.0])[:, None, None]
    return image


def alpha_at(rgba, u, v=0.5, scale=(2.0, 2.0)):
    x = round((0.5 + (u - 0.5) / scale[0]) * 319)
    y = round((0.5 + (v - 0.5) / scale[1]) * 179)
    return int(rgba[y, x, 3])


def settled(ambient, image, **kwargs):
    result = None
    for tick in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        result = ambient.update(image, 1.0, tick=tick, **kwargs)
    return pixels(result)


def gl_sample(rgba, u, v, scale=(2.0, 2.0)):
    # 接收端RGBA纹素中心为(i+.5)/size；按实际GL双线性坐标合成。
    x = (0.5 + (u - 0.5) / scale[0]) * rgba.shape[1] - 0.5
    y = (0.5 + (v - 0.5) / scale[1]) * rgba.shape[0] - 0.5
    left, top = math.floor(x), math.floor(y)
    tx, ty = x - left, y - top
    return (rgba[top, left].float() * (1 - tx) + rgba[top, left + 1].float() * tx) * (1 - ty) + (
        rgba[top + 1, left].float() * (1 - tx) + rgba[top + 1, left + 1].float() * tx
    ) * ty


@pytest.mark.parametrize("border", [1, 3, 6])
@pytest.mark.parametrize("feather", [0.0, 0.03])
@pytest.mark.parametrize("scale", [(2.0, 2.0), (4.0, 4.0)])
def test_thin_paired_black_borders_have_no_dark_seam_at_real_gl_samples(border, feather, scale):
    image = frame((border, border, 384 - border, 216 - border))
    rgba = settled(background(), image, sbs=torch.cat((image, image), 2), feather=feather, scale=scale)
    color = torch.tensor([51.0, 153.0, 255.0])
    for u, v in (
        (border / 2 / 384, 0.5),
        ((384 - border / 2) / 384, 0.5),
        (0.5, border / 2 / 216),
        (0.5, (216 - border / 2) / 216),
        (0.5, (216 - border + 0.5) / 216),
    ):
        sample = gl_sample(rgba, u, v, scale)
        assert sample[3] == pytest.approx(255.0, abs=0.01)
        # 黑视频叠加后应与外侧柔光同色，不留低alpha或混黑提色造成的暗圈。
        assert sample[:3] * (sample[3] / 255) == pytest.approx(color, abs=3)
    assert gl_sample(rgba, 0.5, 0.5, scale)[3] == 0


@pytest.mark.parametrize("border", [1, 3])
def test_thin_single_bottom_border_is_filled_without_changing_portrait_content(border):
    image = frame((132, 0, 252, 216 - border))
    before = image.clone()
    rgba = settled(background(), image, sbs=torch.cat((image, image), 2), feather=0.03)
    sample = gl_sample(rgba, 0.5, (216 - border + 0.5) / 216)
    assert sample[3] == pytest.approx(255.0, abs=0.01)
    assert sample[:3] == pytest.approx((51, 153, 255), abs=3)
    assert torch.equal(image, before)


def test_fine_edge_color_crop_keeps_capture_roi_offset_and_fractional_pixel_bounds():
    source = torch.zeros(3, 279, 475)
    source[0] = 1
    content = torch.zeros(3, 215, 383)
    content[:, 1:-3, 1:-3] = torch.tensor([0.2, 0.6, 1.0])[:, None, None]
    source[:, 31:246, 47:430] = content
    ambient = background()
    rgba = settled(ambient, source, content_rect=(47, 31, 430, 246), sbs=torch.cat((content, content), 2), feather=0.03)
    assert gl_sample(rgba, 0.5, (215 - 2) / 215)[:3] == pytest.approx((51, 153, 255), abs=3)
    assert ambient._source_window.active == pytest.approx((1 / 383, 1 / 215, 380 / 383, 212 / 215))


def test_fine_edges_do_not_shrink_the_protected_window_in_a_low_confidence_dark_scene():
    image = torch.full((3, 216, 384), 7 / 255)
    image[:, -3:] = 0
    ambient = background()
    settled(ambient, image, sbs=torch.cat((image, image), 2), feather=0.0)
    assert ambient._source_window.active == ambient._display_window.active == (0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("rect", [(132, 0, 252, 216), (0, 24, 384, 192)])
@pytest.mark.parametrize("noise", [False, True])
def test_content_colors_replace_black_edges_without_covering_center(rect, noise):
    rgba = settled(background(), frame(rect, noise))
    assert tuple(rgba[90, 160, :3].tolist()) == pytest.approx((51, 153, 255), abs=5)
    assert alpha_at(rgba, 0.5) == 0
    if rect[0]:
        assert alpha_at(rgba, 0.1) == alpha_at(rgba, 0.9) == 255
    else:
        assert alpha_at(rgba, 0.5, 0.03) == alpha_at(rgba, 0.5, 0.97) == 255


def test_feather_is_a_narrow_monotone_ring_and_outer_edges_are_transparent():
    rgba = pixels(background().update(torch.ones(3, 216, 384), 1.0, tick=0.0, feather=0.1))
    values = [alpha_at(rgba, u) for u in (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.5)]
    assert values[0] >= 250 and values[-1] == 0
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert any(0 < value < 255 for value in values)
    alpha = rgba[:, :, 3]
    assert not torch.count_nonzero(alpha[0]) and not torch.count_nonzero(alpha[-1])
    assert not torch.count_nonzero(alpha[:, 0]) and not torch.count_nonzero(alpha[:, -1])


def test_zero_feather_and_disabled_detection_leave_the_whole_main_window_clear():
    rgba = settled(background(), frame(), fill_borders=False, feather=0.0)
    assert all(alpha_at(rgba, u) == 0 for u in (0.03, 0.1, 0.5, 0.9, 0.97))
    assert rgba[90, 30, 3] > 0


def test_final_sbs_uses_union_of_both_eyes():
    left, right = frame((144, 0, 240, 216)), frame((120, 0, 264, 216))
    rgba = settled(background(), frame(), sbs=torch.cat((left, right), dim=2), feather=0.0)
    assert alpha_at(rgba, 0.32) == alpha_at(rgba, 0.68) == 0
    assert alpha_at(rgba, 0.1) == 255


def test_display_subtitle_and_tiny_bright_marks_are_not_hidden_by_border_fill():
    display = frame()
    display[:, 190:198, 18:30] = 1.0
    display[:, 190:198, 354:366] = 1.0
    display[:, 107, 24] = 1.0
    rgba = settled(background(), frame(), sbs=torch.cat((display, display), dim=2))
    assert alpha_at(rgba, 24 / 384, 194 / 216) == 0
    assert alpha_at(rgba, 24 / 384, 107 / 216) == 0


@pytest.mark.parametrize(
    "mark", [(75, 100, 78, 110), (75, 100, 78, 124), (75, 100, 78, 130), (75, 100, 78, 150), (1, 1, 4, 4)]
)
@pytest.mark.parametrize("feather", [0.0, 0.03, 0.1])
@pytest.mark.parametrize("scale", [(2.0, 2.0), (4.0, 4.0)])
def test_feather_does_not_mix_over_bright_marks_outside_the_main_video(mark, feather, scale):
    display = frame()
    left, top, right, bottom = mark
    display[:, top:bottom, left:right] = 1.0
    display[:, top:bottom, 384 - right : 384 - left] = 1.0
    rgba = settled(background(), frame(), sbs=torch.cat((display, display), dim=2), feather=feather, scale=scale)
    u, v = (left + right) / (2 * 384), (top + bottom) / (2 * 216)
    assert alpha_at(rgba, u, v, scale) == alpha_at(rgba, 1 - u, v, scale) == 0
    # 四个邻近纹素均为0，接收端双线性采样不会重新把柔光混入亮字。
    for at_u in (u, 1 - u):
        x = (0.5 + (at_u - 0.5) / scale[0]) * 319
        y = (0.5 + (v - 0.5) / scale[1]) * 179
        assert not torch.count_nonzero(rgba[math.floor(y) : math.ceil(y) + 1, math.floor(x) : math.ceil(x) + 1, 3])


def test_mid_side_subtitles_leave_normal_vertical_feather_unchanged():
    display = frame()
    display[:, 100:110, 75:78] = 1.0
    display[:, 100:110, 306:309] = 1.0
    plain = settled(background(), frame())
    protected = settled(background(), frame(), sbs=torch.cat((display, display), dim=2))
    assert torch.equal(plain[:, 160, 3], protected[:, 160, 3])


def test_sparse_subtitles_do_not_turn_source_edge_colors_black():
    image = frame(noise=True)
    image[:, 195:201, 18:105] = 1.0
    image[:, 195:201, 279:366] = 1.0
    rgba = settled(background(), image)
    assert tuple(rgba[90, 160, :3].tolist()) == pytest.approx((51, 153, 255), abs=5)


def test_small_bright_object_in_a_dark_scene_does_not_create_a_content_crop():
    image = torch.full((3, 216, 384), 7 / 255)
    image[:, 90:126, 165:219] = 0.8
    rgba = settled(background(), image, feather=0.0)
    assert alpha_at(rgba, 0.1) == alpha_at(rgba, 0.9) == 0


def test_unpaired_dark_edge_is_kept_as_part_of_the_main_picture():
    image = frame((60, 0, 384, 216))
    rgba = settled(background(), image, feather=0.0)
    assert alpha_at(rgba, 0.03) == 0


@pytest.mark.parametrize("dark", [0.0, 7 / 255, 20 / 255])
def test_dark_or_unknown_frames_immediately_release_the_previous_narrow_window(dark):
    ambient = background()
    assert alpha_at(settled(ambient, frame(), feather=0.0), 0.1) == 255
    image = torch.full((3, 216, 384), dark)
    rgba = pixels(ambient.update(image, 1.0, tick=1.1, feather=0.0))
    assert alpha_at(rgba, 0.1) == alpha_at(rgba, 0.5) == 0


def test_expansion_is_immediate_but_narrowing_requires_stable_samples():
    ambient = background()
    image = frame()
    first = pixels(ambient.update(image, 1.0, tick=0.0, feather=0.0))
    assert alpha_at(first, 0.1) == 0
    pixels(ambient.update(image, 1.0, tick=0.1, feather=0.0))
    assert alpha_at(pixels(ambient.update(image, 1.0, tick=0.2, feather=0.0)), 0.1) == 255
    assert alpha_at(pixels(ambient.update(torch.ones_like(image), 1.0, tick=0.3, feather=0.0)), 0.1) == 0


def test_final_sbs_shape_change_clears_pending_narrowing():
    ambient = background()
    image = frame()
    ambient.update(image, 1.0, tick=0.0, sbs=torch.cat((image, image), 2), feather=0.0)
    ambient.update(image, 1.0, tick=0.1, sbs=torch.cat((image, image), 2), feather=0.0)
    small = torch.nn.functional.interpolate(image[None], size=(108, 192), mode="area")[0]
    rgba = pixels(ambient.update(image, 1.0, tick=0.2, sbs=torch.cat((small, small), 2), feather=0.0))
    assert alpha_at(rgba, 0.1) == 0


def test_rotating_content_between_portrait_and_letterbox_releases_old_bars_first():
    ambient = background()
    settled(ambient, frame(), feather=0.0)
    wide = frame((0, 24, 384, 192))
    rgba = pixels(ambient.update(wide, 1.0, tick=1.1, feather=0.0))
    assert alpha_at(rgba, 0.1) == 0
    assert alpha_at(rgba, 0.5, 0.03) == 0
    ambient.update(wide, 1.0, tick=1.2, feather=0.0)
    rgba = pixels(ambient.update(wide, 1.0, tick=1.3, feather=0.0))
    assert alpha_at(rgba, 0.5, 0.03) == 255


def test_reset_forgets_the_confirmed_display_window():
    ambient = background()
    settled(ambient, frame(), feather=0.0)
    ambient.reset()
    rgba = pixels(ambient.update(frame(), 1.0, tick=1.1, feather=0.0))
    assert alpha_at(rgba, 0.1) == 0


def test_capture_roi_changes_clear_color_and_tracker_history():
    ambient = background()
    image = frame()
    settled(ambient, image)
    image[:, :, :120] = torch.tensor([1.0, 0.0, 0.0])[:, None, None]
    rgba = pixels(ambient.update(image, 1.0, (0, 0, 120, 216), tick=1.1, fill_borders=False))
    assert tuple(rgba[90, 160, :3].tolist()) == (255, 0, 0)


def test_anisotropic_scale_maps_the_clear_window_to_the_actual_main_surface():
    rgba = pixels(background().update(torch.ones(3, 216, 384), 1.0, tick=0.0, scale=(1.4, 3.0), feather=0.0))
    assert alpha_at(rgba, 0.5, scale=(1.4, 3.0)) == 0
    assert alpha_at(rgba, 0.02, scale=(1.4, 3.0)) == 0
    assert rgba[90, 20, 3] > 0 and rgba[20, 160, 3] > 0


@pytest.mark.parametrize(
    "options",
    [
        {"fill_borders": 1},
        {"feather": True},
        {"feather": -0.01},
        {"feather": 0.11},
        {"feather": math.nan},
        {"scale": (0.0, 2.0)},
        {"scale": (2.0,)},
        {"scale": (math.inf, 2.0)},
        {"sbs": torch.zeros(3, 40, 81)},
        {"sbs": torch.zeros(4, 40, 80)},
        {"sbs": torch.full((3, 40, 80), math.nan)},
    ],
)
def test_invalid_overlay_inputs_are_rejected(options):
    with pytest.raises(ValueError):
        background().update(torch.ones(3, 216, 384), 1.0, tick=0.0, **options)


def test_rate_limited_calls_do_not_read_back_or_detect_again(monkeypatch):
    ambient = background()
    image = frame()
    ambient.update(image, 1.0, tick=0.0)
    monkeypatch.setattr(torch.nn.functional, "interpolate", lambda *args, **kwargs: pytest.fail("unexpected readback"))
    assert ambient.update(image, 1.0, tick=0.099) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_matches_cpu_with_final_stereo_protection_and_no_input_mutation():
    image = frame(noise=True)
    sbs = torch.cat((image, image), 2)
    cpu = settled(background(), image, sbs=sbs)
    cuda_frame = image.cuda().requires_grad_()
    before = cuda_frame.detach().clone()
    cuda = settled(background(), cuda_frame, sbs=sbs.cuda())
    assert torch.max(torch.abs(cpu.to(torch.int16) - cuda.to(torch.int16))) <= 1
    assert torch.equal(cuda_frame, before) and cuda_frame.grad is None
