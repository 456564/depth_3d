from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from nunif.utils.video import VideoOutputConfig, pyav_init_cuda_primary_context
from nunif.utils.video.color_transform import configure_video_codec, setup_color_transform

pytestmark = [
    pytest.mark.hwaccel,
    pytest.mark.skipif(
        not torch.cuda.is_available() or "hevc_nvenc" not in av.codecs_available, reason="requires CUDA NVENC"
    ),
]


def decode_all(packets: list[av.Packet]) -> list[av.VideoFrame]:
    decoder = av.CodecContext.create("hevc", "r")
    frames = []
    for packet in packets:
        frames.extend(decoder.decode(packet))
    return frames + decoder.decode(None)


@pytest.mark.parametrize(
    "width,height,dtype,pix_fmt",
    [
        (640, 180, torch.float32, "yuv420p"),
        (636, 182, torch.float16, "yuv420p"),
        (636, 182, torch.float32, "yuv420p10le"),
    ],
)
def test_cuda_nvenc_transform_preserves_every_frame_and_both_eyes(width, height, dtype, pix_fmt) -> None:
    from iw3.desktop.quest_encoder import QuestEncoder

    pyav_init_cuda_primary_context(max_devices=1)
    config = VideoOutputConfig(video_codec="hevc_nvenc", pix_fmt=pix_fmt, colorspace="bt709-tv")
    configure_video_codec(config)
    _, transform = setup_color_transform(None, config, device=torch.device("cuda:0"))
    encoder = QuestEncoder(width, height, 30, 35).encoder
    encoder.pix_fmt = "cuda"
    stream = torch.cuda.Stream()
    packets = []
    expected = []
    with torch.inference_mode(), torch.cuda.stream(stream):
        for index in range(65):
            rgb = torch.empty(3, height, width, dtype=dtype, device="cuda")
            left, right = ((index % 5 + 1) / 5, 0.1, 0.25), (0.15, ((index + 2) % 5 + 1) / 5, 0.8)
            rgb[:, :, : width // 2] = torch.tensor(left, device="cuda")[:, None, None]
            rgb[:, :, width // 2 :] = torch.tensor(right, device="cuda")[:, None, None]
            frame = transform(rgb)
            assert frame.format.name == "cuda" and frame.sw_format.name == config.pix_fmt
            frame.pts, frame.time_base = index * 33_333, Fraction(1, 1_000_000)
            if index == 23:
                frame.pict_type = av.video.frame.PictureType.I
            encoded = encoder.encode(frame)
            assert len(encoded) == 1
            packets.extend(encoded)
            expected.append((left, right))
            # 不保留旧RGB/AVFrame；DLPack必须持有实际编码storage。
            del rgb, frame
    assert not encoder.encode(None)
    frames = decode_all(packets)
    assert len(frames) == 65
    assert [p.pts for p in packets] == [index * 33_333 for index in range(65)]
    for index, frame in enumerate(frames):
        image = frame.to_ndarray(format="rgb24")
        for eye, x in enumerate((width // 4, width * 3 // 4)):
            for y in (2, height // 2, height - 3):
                np.testing.assert_allclose(image[y, x], np.array(expected[index][eye]) * 255, atol=10)
    assert packets[23].is_keyframe
    assert len(decode_all([packets[23]])) == 1


def test_quest_cuda_encoding_stays_on_gpu_and_accepts_later_cpu_frame() -> None:
    from iw3.desktop.quest_encoder import QuestEncoder

    pyav_init_cuda_primary_context(max_devices=1)
    encoder = QuestEncoder(640, 180, 30, 35)
    with torch.inference_mode():
        rgb = torch.zeros(3, 180, 640, device="cuda")
        rgb[0, :, :320], rgb[1, :, 320:] = 1, 1
        packets = encoder.encode(rgb, 0)
        assert encoder.encoder.pix_fmt == "cuda", "CUDA input must not download the encoded frame to CPU"
        packets += encoder.encode(rgb.cpu(), 33_333)
    assert len(decode_all(packets)) == 2


def test_cuda_initialization_failure_uses_compatible_nvenc_input(monkeypatch) -> None:
    from iw3.desktop import quest_encoder as module

    pyav_init_cuda_primary_context(max_devices=1)
    original = module.setup_color_transform
    attempted = []

    def unavailable(*args, **kwargs):
        if kwargs.get("device") is not None:
            attempted.append(True)
            raise RuntimeError("synthetic unavailable CUDA frame interop")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "setup_color_transform", unavailable)
    encoder = module.QuestEncoder(640, 180, 30, 35)
    packet = encoder.encode(torch.zeros(3, 180, 640, device="cuda"), 0)
    assert attempted and encoder.encoder.pix_fmt == "nv12"
    assert len(decode_all(packet)) == 1


def test_first_cuda_encode_failure_recreates_codec_and_sends_decodable_idr(monkeypatch) -> None:
    from iw3.desktop.quest_encoder import QuestEncoder

    pyav_init_cuda_primary_context(max_devices=1)
    original = QuestEncoder._create_encoder

    class BrokenCudaContext:
        def encode(self, frame):
            raise RuntimeError("synthetic unsupported CUDA input at codec open")

    def unavailable(self, pix_fmt):
        return BrokenCudaContext() if pix_fmt == "cuda" else original(self, pix_fmt)

    monkeypatch.setattr(QuestEncoder, "_create_encoder", unavailable)
    encoder = QuestEncoder(640, 180, 30, 35)
    packets = encoder.encode(torch.zeros(3, 180, 640, device="cuda"), 123_456)
    assert encoder.last_pts == 123_456 and encoder.encoder.pix_fmt == "nv12"
    assert len(packets) == 1 and packets[0].is_keyframe
    assert len(decode_all(packets)) == 1


def test_cpu_first_codec_keeps_compatible_input_when_later_frame_is_cuda() -> None:
    from iw3.desktop.quest_encoder import QuestEncoder

    encoder = QuestEncoder(640, 180, 30, 35)
    packets = encoder.encode(torch.zeros(3, 180, 640), 0)
    packets += encoder.encode(torch.zeros(3, 180, 640, device="cuda"), 33_333)
    assert encoder.encoder.pix_fmt == "nv12"
    assert len(decode_all(packets)) == 2
