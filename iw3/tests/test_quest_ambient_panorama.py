import importlib
import math
import struct
import zlib

import pytest
import torch

from iw3.desktop.quest_ambient import AmbientBackground
from iw3.desktop.quest_input import _rotate
from iw3.desktop.quest_server import QuestServer

STYLES = ("edge", "blur", "halo", "cinema", "aurora", "mist", "stars")


def rgba(payload: bytes | None) -> torch.Tensor:
    assert payload is not None
    size = payload[0]
    width, height, mode = struct.unpack("!HHB", payload[size + 1 : size + 6])
    assert mode == 1
    return torch.frombuffer(bytearray(zlib.decompress(payload[size + 6 :])), dtype=torch.uint8).reshape(
        height, width, 4
    )


@pytest.mark.parametrize("style", STYLES)
def test_full_background_keeps_portrait_clear_and_fills_every_face(style: str) -> None:
    frame = torch.tensor([0.2, 0.6, 1.0])[:, None, None].expand(3, 64, 36)
    image = rgba(
        AmbientBackground().update(
            frame, 0.8, tick=0, video_rect=(0.34, 0, 0.66, 1), style=style, panoramic=True, motion=False
        )
    )
    assert image.shape == (316, 320, 4)
    assert image[90, 160, 3] == 0
    assert image[90, 96, 3] == image[90, 223, 3] == 255
    for x, y in ((2, 182), (70, 182), (138, 182), (2, 250), (70, 250), (138, 250)):
        face = image[y : y + 64, x : x + 64]
        assert torch.all(face[..., 3] == 255)
        assert face[..., :3].float().mean() > 0
    assert torch.isfinite(image.float()).all()


@pytest.mark.parametrize("style", ("aurora", "mist", "stars"))
def test_new_styles_are_visible_without_panorama_and_animate_slowly(style: str) -> None:
    frame = torch.tensor([0.2, 0.6, 1.0])[:, None, None].expand(3, 64, 36)
    ambient = AmbientBackground()
    first = rgba(ambient.update(frame, 0.8, tick=0, style=style, motion=True))
    assert first.shape == (180, 320, 4)
    assert first[..., :3].float().mean() > 0 and first[90, 160, 3] == 0
    second = rgba(ambient.update(frame, 0.8, tick=2, style=style, motion=True))
    assert not torch.equal(first[..., :3], second[..., :3])
    assert torch.equal(first[..., 3], second[..., 3])
    assert (first[..., :3].float() - second[..., :3].float()).abs().mean() < 30
    frozen = AmbientBackground()
    first = frozen.update(frame, 0.8, tick=0, style=style, motion=False)
    second = frozen.update(frame, 0.8, tick=2, style=style, motion=False)
    assert first == second


def test_motion_refresh_uses_cached_cpu_colors_and_respects_ten_hz(monkeypatch) -> None:
    ambient = AmbientBackground()
    initial = ambient.update(torch.ones(3, 40, 30), 0.6, tick=0, style="aurora", panoramic=True)
    monkeypatch.setattr(ambient, "_validate_frame", lambda *_: pytest.fail("paused animation read the source tensor"))
    assert ambient.animate(tick=0.099) is None
    moving = ambient.animate(tick=0.1)
    assert moving is not None
    assert ambient.animate(tick=0.199) is None
    assert ambient.animate(tick=2) != initial
    ambient.reset()
    assert ambient.animate(tick=3) is None


@pytest.mark.parametrize("key", ("panoramic", "motion"))
@pytest.mark.parametrize("value", (0, 1, None, "true"))
def test_ambient_rejects_non_boolean_panorama_controls(key, value) -> None:
    with pytest.raises(ValueError, match="ambient"):
        AmbientBackground().update(torch.ones(3, 40, 30), 0.6, **{key: value})
    with pytest.raises(ValueError, match="ambient"):
        QuestServer("127.0.0.1", 128, 64, 30, 35, **{"ambient_360" if key == "panoramic" else "ambient_motion": value})


def test_panorama_switch_applies_immediately_without_overwriting_overlay_dimensions() -> None:
    ambient = AmbientBackground()
    frame = torch.ones(3, 40, 30)
    assert rgba(ambient.update(frame, 0.6, tick=0)).shape == (180, 320, 4)
    assert rgba(ambient.update(frame, 0.6, tick=0.01, panoramic=True)).shape == (316, 320, 4)
    assert (ambient.WIDTH, ambient.HEIGHT) == (320, 180)


def test_six_native_layers_are_closed_inward_and_precede_unmodified_video() -> None:
    server = QuestServer("127.0.0.1", 72, 64, 30, 35, ambient_360=True, reference_aspect=16 / 9)
    before = server._layers()["camadas"]
    server._ambient_visible = True
    layers = server._layers()["camadas"]
    faces, video, overlay = layers[:6], layers[6:8], layers[8]
    assert len(layers) == 9 and video == before
    assert video[0]["tam"] == pytest.approx([1.1390625, 2.025])
    assert overlay["alpha"] is True and overlay["rect"] == [0, 0, 320, 180]
    for face in faces:
        assert face["tipo"] == "quad" and face["olho"] == "ambos" and face["alpha"] is False
        assert face["tex"] == overlay["tex"]
        normal = _rotate(face["rot"], [0, 0, 1])
        assert sum(-a * b for a, b in zip(face["pos"], normal)) > 0
    for direction in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1]):
        hits = 0
        for face in faces:
            normal = _rotate(face["rot"], [0, 0, 1])
            denominator = sum(a * b for a, b in zip(direction, normal))
            if denominator >= -1e-9:
                continue
            t = sum(a * b for a, b in zip(face["pos"], normal)) / denominator
            delta = [t * a - b for a, b in zip(direction, face["pos"])]
            inverse = [-v for v in face["rot"][:3]] + [face["rot"][3]]
            local = _rotate(inverse, delta)
            hits += t > 0 and abs(local[0]) <= face["tam"][0] / 2 + 1e-6 and abs(local[1]) <= face["tam"][1] / 2 + 1e-6
        assert hits >= 1


@pytest.mark.parametrize("style", STYLES)
def test_shared_edge_colors_match_before_texture_sampling(style: str) -> None:
    module = importlib.import_module("iw3.desktop.quest_ambient_panorama")
    panorama = module.AmbientPanorama()
    palette = torch.tensor([[0.1, 0.2, 0.9], [0.9, 0.2, 0.4], [0.2, 0.7, 0.4], [0.8, 0.4, 0.1]])
    faces = panorama.colors(palette, style, phase=0.4)
    assert faces.shape == (6, 64, 64, 3)
    assert torch.isfinite(faces).all() and 0 <= faces.min() <= faces.max() <= 1
    assert torch.allclose(faces[0, :, -1], faces[2, :, 0], atol=1e-5)
    assert torch.allclose(faces[0, 0], faces[4, -1], atol=1e-5)
    assert math.isfinite(float(faces.mean()))


def test_styles_are_distinct_and_follow_video_palette_instead_of_fixed_artwork() -> None:
    frame = torch.zeros(3, 64, 64)
    frame[0, :32] = 1
    frame[1, 32:] = 1
    frame[2, :, :32] = 1
    images = [
        AmbientBackground().update(frame, 0.6, tick=0, style=style, panoramic=True, motion=False) for style in STYLES
    ]
    assert len(set(images)) == 7
    for style in STYLES:
        red = rgba(
            AmbientBackground().update(
                torch.tensor([1.0, 0.0, 0.0])[:, None, None].expand(3, 32, 32), 1, tick=0, style=style, panoramic=True
            )
        )
        assert red[180:, :, 0].sum() > 0
        assert not torch.count_nonzero(red[180:, :, 1:3])


def test_six_face_geometry_tracks_window_pose_without_using_head_orientation() -> None:
    server = QuestServer("127.0.0.1", 128, 64, 30, 35, ambient_360=True)
    server._ambient_visible = True
    rotation = [0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    position = [2.0, 1.0, -4.0]
    faces = server._layers((position, rotation), [5, 5])["camadas"][:6]
    assert faces[0]["pos"] == pytest.approx([-48, 1, -4])
    assert faces[1]["pos"] == pytest.approx([52, 1, -4])
    for face in faces:
        normal = _rotate(face["rot"], [0, 0, 1])
        delta = [a - b for a, b in zip(position, face["pos"])]
        assert sum(a * b for a, b in zip(delta, normal)) == pytest.approx(50)
