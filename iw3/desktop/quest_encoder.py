from fractions import Fraction
from typing import cast

import av
import torch
from av.video.codeccontext import VideoCodecContext

from nunif.utils.video import VideoOutputConfig
from nunif.utils.video.color_transform import configure_video_codec, setup_color_transform


class QuestEncoder:
    def __init__(self, width: int, height: int, fps: int, bitrate: int) -> None:
        if not all(type(value) is int for value in (width, height, fps, bitrate)):
            raise ValueError("Quest encoder dimensions, FPS and bitrate must be integers")
        if width <= 0 or height <= 0 or width % 4 or height % 2:
            raise ValueError("Full SBS width must be divisible by 4 and height must be positive and even")
        if fps <= 0 or not 5 <= bitrate <= 100:
            raise ValueError("Quest FPS must be positive and bitrate must be 5-100 Mbps")
        self.width = width
        self.height = height
        self.last_pts = -1
        config = VideoOutputConfig(video_codec="hevc_nvenc", pix_fmt="yuv420p", colorspace="bt709-tv")
        configure_video_codec(config)
        # shortcut: CUDA DLPack 连续帧颜色校验失败，修复并通过逐帧校验后才能启用直传。
        _, self.transform = setup_color_transform(None, config)
        self.encoder = cast(VideoCodecContext, av.CodecContext.create("hevc_nvenc", "w"))
        self.encoder.width = width
        self.encoder.height = height
        self.encoder.pix_fmt = config.pix_fmt
        self.encoder.time_base = Fraction(1, 1_000_000)
        self.encoder.framerate = Fraction(fps, 1)
        self.encoder.bit_rate = bitrate * 1_000_000
        self.encoder.gop_size = fps * 2
        self.encoder.max_b_frames = 0
        self.encoder.colorspace = self.transform.dst_colorspace
        self.encoder.color_primaries = self.transform.dst_color_primaries
        self.encoder.color_trc = self.transform.dst_color_trc
        self.encoder.color_range = self.transform.dst_color_range
        self.encoder.options = {
            "preset": "p5",
            "tune": "ull",
            "rc": "cbr",
            "zerolatency": "1",
            "rc-lookahead": "0",
            "forced-idr": "1",
            "delay": "0",
        }

    @torch.inference_mode()
    def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False) -> list[av.Packet]:
        if frame.shape != (3, self.height, self.width):
            raise ValueError(f"Expected RGB CHW {(3, self.height, self.width)}, got {tuple(frame.shape)}")
        if frame.dtype not in (torch.float16, torch.float32) or frame.device.type not in {"cpu", "cuda"}:
            raise ValueError("Quest frames must be float16/float32 RGB tensors on CPU or CUDA")
        if type(pts) is not int or pts <= self.last_pts:
            raise ValueError("Quest frame PTS must be nonnegative and strictly increasing integer microseconds")
        # 实测 RGB CHW float CUDA [3,1080,3840] → CPU NV12 → HEVC Annex-B；双眼逐帧颜色校验。
        video_frame = self.transform(frame)
        video_frame.pts = pts
        video_frame.time_base = Fraction(1, 1_000_000)
        if force_idr:
            video_frame.pict_type = av.video.frame.PictureType.I
        packets = self.encoder.encode(video_frame)
        self.last_pts = pts
        return packets
