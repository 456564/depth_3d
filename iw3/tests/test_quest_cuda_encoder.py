from concurrent.futures import ThreadPoolExecutor
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
            assert (
                frame.format.name == "cuda" and frame.sw_format is not None and frame.sw_format.name == config.pix_fmt
            )
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


def test_prepared_cuda_frames_encode_from_another_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from iw3.desktop.quest_encoder import PreparedCudaFrame, QuestEncoder

    pyav_init_cuda_primary_context(max_devices=1)
    width, height = 7680, 2160
    encoder = QuestEncoder(width, height, 30, 35)
    expected = [
        ((0.9, 0.1, 0.2), (0.2, 0.9, 0.1)),
        ((0.2, 0.1, 0.8), (0.8, 0.8, 0.2)),
        ((0.1, 0.8, 0.8), (0.8, 0.2, 0.7)),
        ((0.6, 0.2, 0.1), (0.4, 0.8, 0.3)),
    ]

    def make_rgb(colors: tuple[tuple[float, float, float], tuple[float, float, float]]) -> torch.Tensor:
        rgb = torch.empty(3, height, width, dtype=torch.float16, device="cuda")
        for eye, color in enumerate(colors):
            rgb[:, :, eye * width // 2 : (eye + 1) * width // 2] = torch.tensor(color, device="cuda")[:, None, None]
        return rgb

    producer_stream = torch.cuda.Stream()
    warmup = make_rgb(expected[0])
    assert encoder.cuda_preparer() is None
    packets = encoder.encode(warmup, 0)
    del warmup
    prepare_cuda_frame = encoder.cuda_preparer()
    assert prepare_cuda_frame is not None

    def prepare(index: int) -> tuple[torch.Tensor, PreparedCudaFrame | None]:
        with torch.cuda.stream(producer_stream):
            rgb = make_rgb(expected[index])
            return rgb, prepare_cuda_frame(rgb)

    def unexpected_conversion(frame: torch.Tensor) -> av.VideoFrame:
        raise AssertionError("a matching prepared frame must skip the synchronous conversion")

    with ThreadPoolExecutor(max_workers=1) as producer, monkeypatch.context() as patch:
        patch.setattr(encoder.transform, "from_cuda_tensor", unexpected_conversion)
        for index in (1, 2):
            frame, prepared = producer.submit(prepare, index).result(timeout=10)
            assert prepared is not None
            packets += encoder.encode(frame, index * 33_333, force_idr=index == 1, prepared=prepared)

    # 旧的转色结果即使来自同一编码器，也不能用于不同的RGB张量。
    packets += encoder.encode(make_rgb(expected[3]), 99_999, prepared=prepared)
    assert [p.pts for p in packets] == [0, 33_333, 66_666, 99_999]
    assert packets[1].is_keyframe and len(decode_all([packets[1]])) == 1
    decoded = decode_all(packets)
    assert len(decoded) == len(expected)
    for decoded_frame, colors in zip(decoded, expected, strict=True):
        image = decoded_frame.to_ndarray(format="rgb24")
        for eye, x in enumerate((width // 4, width * 3 // 4)):
            for y in (2, height // 2, height - 3):
                np.testing.assert_allclose(image[y, x], np.array(colors[eye]) * 255, atol=10)


def test_4k_bt709_nv12_downsamples_both_chroma_planes_together(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.nn.functional as F

    from nunif.utils.video.color_transform import ColorTransform

    pyav_init_cuda_primary_context(max_devices=1)
    width, height = 7680, 2160
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(709)
    rgb = torch.rand(3, height, width, device=device, generator=generator)
    rgb[:, :8, : width // 2] = 0
    rgb[:, :8, width // 2 :] = 1
    config = VideoOutputConfig(video_codec="hevc_nvenc", pix_fmt="yuv420p", colorspace="bt709-tv")
    configure_video_codec(config)
    _, transform = setup_color_transform(None, config, device=device)
    assert transform.is_cuda_dlpack_supported(rgb)
    expected_y, expected_uv = ColorTransform.to_yuv_planes(
        *ColorTransform.rgb_to_yuv(rgb.unsqueeze(0), colorspace=1, color_range=1, out_format="nv12")
    )

    interpolate = F.interpolate
    resized = []

    def observe(x, *args, **kwargs):
        resized.append(tuple(x.shape))
        return interpolate(x, *args, **kwargs)

    monkeypatch.setattr(F, "interpolate", observe)
    y, uv, fmt = transform.prepare_yuv_planes(rgb)
    # 原始 RGB [3,2160,7680] → Y [2160,7680] + UV [1080,3840,2]，同一 CUDA allocation。
    assert fmt == "nv12" and y.shape == expected_y.shape and uv.shape == expected_uv.shape
    assert y.dtype == uv.dtype == torch.uint8 and y.device == uv.device == device
    assert uv.data_ptr() - y.data_ptr() == width * height
    assert y.untyped_storage().data_ptr() == uv.untyped_storage().data_ptr()
    assert torch.equal(y, expected_y)
    delta = (uv.to(torch.int16) - expected_uv.to(torch.int16)).abs()
    assert delta.max().item() <= 1
    assert (delta > 0).sum().item() < uv.numel() / 1000
    assert resized == [(1, 2, height, width)]


@pytest.mark.parametrize("width,height", [(7680, 2160), (4320, 3840)])
def test_quest_preparer_fuses_4k_bt709_nv12_without_changing_bytes(
    monkeypatch: pytest.MonkeyPatch, width: int, height: int
) -> None:
    import sys

    from iw3.desktop.quest_encoder import QuestEncoder

    if sys.platform != "win32" or torch.__version__ != "2.14.1+cu132":
        pytest.skip("requires the validated Windows CUDA runtime")
    pyav_init_cuda_primary_context(max_devices=1)
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (8, 9) or torch.backends.cudnn.version() != 92400:
        pytest.skip("requires the validated Ada/cuDNN runtime")
    encoder = QuestEncoder(width, height, 60, 45)
    producer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        rng = torch.Generator(device=device).manual_seed(823)
        frame = torch.rand((3, height, width), device=device, generator=rng)
        frame[:, :32] = (torch.arange(width, device=device) % 219 + 0.5) / 219
    producer.synchronize()
    first_packets = encoder.encode(frame, 0, force_idr=True)
    assert encoder.encoder.pix_fmt == "cuda"
    prepare = encoder.cuda_preparer()
    assert prepare is not None
    with torch.cuda.stream(producer):
        old_y, old_uv, old_fmt = encoder.transform.prepare_yuv_planes(frame)
    producer.synchronize()

    def slow_path(*args, **kwargs):
        pytest.fail("eligible 4K frame fell back to the old color conversion")

    monkeypatch.setattr(encoder.transform, "prepare_yuv_planes", slow_path)
    with torch.cuda.stream(producer):
        prepared = prepare(frame)
    assert prepared is not None
    packets = encoder.encode(frame, 16667, force_idr=True, prepared=prepared)
    y, uv, fmt = prepared.planes
    assert old_fmt == fmt == "nv12" and torch.equal(old_y, y) and torch.equal(old_uv, uv)
    assert y.untyped_storage().data_ptr() == uv.untyped_storage().data_ptr()
    assert uv.data_ptr() - y.data_ptr() == height * width
    assert [(p.pts, p.is_keyframe) for p in packets] == [(16667, True)]
    assert [(f.width, f.height) for f in decode_all(first_packets + packets)] == [(width, height)] * 2
