import importlib
import struct
import zlib

import numpy as np


def image_pixels() -> np.ndarray:
    night = importlib.import_module("iw3.desktop.quest_night")
    payload = night.texture()
    length = payload[0]
    assert payload[1 : 1 + length] == b"iw3_night"
    assert struct.unpack("!HHB", payload[1 + length : 6 + length]) == (1548, 1032, 1)
    raw = zlib.decompress(payload[6 + length :])
    assert len(raw) == 1548 * 1032 * 4
    return np.frombuffer(raw, dtype=np.uint8).reshape(1032, 1548, 4)


def faces() -> list[np.ndarray]:
    pixels = image_pixels()
    return [pixels[y + 2 : y + 514, x + 2 : x + 514] for y in (0, 516) for x in (0, 516, 1032)]


def test_night_texture_is_opaque_dim_and_has_real_detail() -> None:
    pixels = image_pixels()
    assert np.all(pixels[..., 3] == 255)
    rgb = pixels[..., :3]
    assert 0 < np.median(rgb) < 48
    assert np.quantile(rgb, 0.99) < 110
    assert rgb.max() <= 128
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 1000
    assert all(face[..., :3].std() > 2 for face in faces())


def test_texture_reuses_the_same_encoded_bytes() -> None:
    night = importlib.import_module("iw3.desktop.quest_night")
    assert night.texture() is night.texture()


def test_each_texture_tile_has_two_pixel_replicated_gutters() -> None:
    pixels = image_pixels()
    for y in (0, 516):
        for x in (0, 516, 1032):
            tile = pixels[y : y + 516, x : x + 516]
            assert np.array_equal(tile[:2], np.repeat(tile[2:3], 2, axis=0))
            assert np.array_equal(tile[-2:], np.repeat(tile[-3:-2], 2, axis=0))
            assert np.array_equal(tile[:, :2], np.repeat(tile[:, 2:3], 2, axis=1))
            assert np.array_equal(tile[:, -2:], np.repeat(tile[:, -3:-2], 2, axis=1))


def test_adjacent_world_directions_have_matching_edge_pixels() -> None:
    front, back, right, left, top, bottom = faces()
    pairs = (
        (front[:, 0], left[:, -1]),
        (front[:, -1], right[:, 0]),
        (back[:, 0], right[:, -1]),
        (back[:, -1], left[:, 0]),
        (front[0], top[-1]),
        (front[-1], bottom[0]),
        (back[0], top[0, ::-1]),
        (back[-1], bottom[-1, ::-1]),
        (right[0], top[::-1, -1]),
        (right[-1], bottom[:, -1]),
        (left[0], top[:, 0]),
        (left[-1], bottom[::-1, 0]),
    )
    assert all(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))) <= 1 for a, b in pairs)


def test_sky_is_above_the_ground_in_the_bundled_panorama() -> None:
    _, _, _, _, top, bottom = faces()
    assert top[..., :3].mean() > bottom[..., :3].mean() + 3
