from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from nunif.utils.video import pyav_init_cuda_primary_context


@pytest.mark.parametrize(
    "width,height,fps,bitrate",
    [(0, 64, 30, 35), (126, 64, 30, 35), (128, 63, 30, 35), (128, 64, 0, 35), (128, 64, 30, 4), (128, 64, 30, 101)],
)
def test_reject_invalid_encoder_configuration(width: int, height: int, fps: int, bitrate: int) -> None:
    from iw3.desktop.quest_encoder import QuestEncoder

    with pytest.raises(ValueError):
        QuestEncoder(width, height, fps, bitrate)


def test_reject_invalid_frame_and_timestamps() -> None:
    from iw3.desktop.quest_encoder import QuestEncoder

    encoder = QuestEncoder(128, 64, 30, 35)
    for frame in (torch.zeros(3, 64, 64), torch.zeros(1, 3, 64, 128), torch.zeros(3, 64, 128, dtype=torch.uint8)):
        with pytest.raises(ValueError):
            encoder.encode(frame, 0)
    with pytest.raises(ValueError):
        encoder.encode(torch.zeros(3, 64, 128), -1)


@pytest.mark.hwaccel
@pytest.mark.skipif(
    not torch.cuda.is_available() or "hevc_nvenc" not in av.codecs_available, reason="requires NVENC GPU"
)
def test_nvenc_preserves_each_eye_color_pts_and_reconnect_keyframe() -> None:
    from iw3.desktop.quest_encoder import QuestEncoder

    pyav_init_cuda_primary_context()
    width, height = 640, 180
    encoder = QuestEncoder(width, height, 30, 35)
    packets = []
    timestamps = [0, 33_333, 66_666, 100_000, 200_000, 233_333, 266_666, 300_000, 333_333, 366_666, 400_000, 433_333]
    colors = [(1.0, 0.0, 0.25), (0.25, 0.0, 1.0), (0.75, 0.0, 0.5)]
    with torch.inference_mode():
        for i, pts in enumerate(timestamps):
            rgb = torch.zeros(3, height, width, device="cuda")
            rgb[:, :, : width // 2] = torch.tensor(colors[i % 3], device="cuda")[:, None, None]
            rgb[1, :, width // 2 :] = 1.0
            encoded = encoder.encode(rgb, pts, force_idr=i == 5)
            assert len(encoded) == 1
            packets.extend(encoded)
    assert [packet.pts for packet in packets] == timestamps
    assert all(packet.time_base == Fraction(1, 1_000_000) and packet.dts == packet.pts for packet in packets)
    assert packets[0].is_keyframe and packets[5].is_keyframe
    assert bytes(packets[0]).startswith(b"\x00\x00\x00\x01")

    decoder = av.CodecContext.create("hevc", "r")
    decoded = []
    for packet in packets:
        decoded.extend(decoder.decode(packet))
    decoded.extend(decoder.decode(None))
    assert len(decoded) == len(timestamps)
    for i, frame in enumerate(decoded):
        assert frame.width == width and frame.height == height
        pixels = frame.to_ndarray(format="rgb24")
        np.testing.assert_allclose(pixels[height // 2, width // 4], np.array(colors[i % 3]) * 255, atol=10)
        np.testing.assert_allclose(pixels[height // 2, width * 3 // 4], [0, 255, 0], atol=10)

    fresh_decoder = av.CodecContext.create("hevc", "r")
    first_frame = fresh_decoder.decode(packets[5]) + fresh_decoder.decode(None)
    assert len(first_frame) == 1
    with pytest.raises(ValueError):
        encoder.encode(torch.zeros(3, height, width), timestamps[-1])
