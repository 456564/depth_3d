import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import av
import torch
from av.video.codeccontext import VideoCodecContext

from nunif.utils.video import VideoOutputConfig
from nunif.utils.video.color_transform import OutputTransform, configure_video_codec, setup_color_transform

from .quest_nv12 import try_pack_nv12


@dataclass(frozen=True, eq=False)
class PreparedCudaFrame:
    frame: torch.Tensor
    transform: OutputTransform
    planes: tuple[torch.Tensor, torch.Tensor, str]
    ready: torch.cuda.Event


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
        self._config = config
        self._cuda_device: torch.device | None = None
        _, self.transform = setup_color_transform(None, config)
        self.fps, self.bitrate = fps, bitrate
        self.encoder = self._create_encoder(config.pix_fmt)

    def _create_encoder(self, pix_fmt: str) -> VideoCodecContext:
        encoder = cast(VideoCodecContext, av.CodecContext.create("hevc_nvenc", "w"))
        encoder.width = self.width
        encoder.height = self.height
        encoder.pix_fmt = pix_fmt
        encoder.time_base = Fraction(1, 1_000_000)
        encoder.framerate = Fraction(self.fps, 1)
        encoder.bit_rate = self.bitrate * 1_000_000
        encoder.gop_size = self.fps * 2
        encoder.max_b_frames = 0
        encoder.colorspace = self.transform.dst_colorspace
        encoder.color_primaries = self.transform.dst_color_primaries
        encoder.color_trc = self.transform.dst_color_trc
        encoder.color_range = self.transform.dst_color_range
        preset = "p1" if os.getenv("IW3_4K60_EXPERIMENT") == "1" and (self.width, self.height) == (7680, 2160) else "p5"
        encoder.options = {
            "preset": preset,
            "tune": "ull",
            "rc": "cbr",
            "zerolatency": "1",
            "rc-lookahead": "0",
            "forced-idr": "1",
            "delay": "0",
        }
        return encoder

    def cuda_preparer(self) -> Callable[[torch.Tensor], PreparedCudaFrame | None] | None:
        """Snapshot the CUDA transform for the producer without sharing the codec."""
        device = self._cuda_device
        if self.last_pts < 0 or device is None:
            return None
        transform = self.transform

        @torch.inference_mode()
        def prepare(frame: torch.Tensor) -> PreparedCudaFrame | None:
            if (
                frame.device != device
                or not transform.is_dlpack_supported(frame)
                or not transform.is_cuda_dlpack_supported(frame)
            ):
                return None
            stream = torch.cuda.current_stream(device)
            frame.record_stream(stream)
            planes = try_pack_nv12(frame, transform, stream)
            if planes is None:
                planes = transform.prepare_yuv_planes(frame)
            return PreparedCudaFrame(frame, transform, planes, stream.record_event())

        return prepare

    @torch.inference_mode()
    def encode(
        self,
        frame: torch.Tensor,
        pts: int,
        force_idr: bool = False,
        *,
        prepared: PreparedCudaFrame | None = None,
    ) -> list[av.Packet]:
        if frame.shape != (3, self.height, self.width):
            raise ValueError(f"Expected RGB CHW {(3, self.height, self.width)}, got {tuple(frame.shape)}")
        if frame.dtype not in (torch.float16, torch.float32) or frame.device.type not in {"cpu", "cuda"}:
            raise ValueError("Quest frames must be float16/float32 RGB tensors on CPU or CUDA")
        if type(pts) is not int or pts <= self.last_pts:
            raise ValueError("Quest frame PTS must be nonnegative and strictly increasing integer microseconds")
        if self.last_pts < 0 and frame.is_cuda:
            try:
                # 首帧在真实编码线程/设备初始化；打开后不切换NVENC输入格式。
                with torch.cuda.device(frame.device):
                    _, self.transform = setup_color_transform(None, self._config, device=frame.device)
                    self._cuda_device = frame.device
                    self.encoder = self._create_encoder("cuda")
                    return self._encode(frame, pts, force_idr)
            except (av.FFmpegError, RuntimeError, ValueError, NotImplementedError) as error:
                logging.getLogger(__name__).warning("CUDA encoder input unavailable, using compatible NVENC: %s", error)
                self._cuda_device = None
                _, self.transform = setup_color_transform(None, self._config)
                self.encoder = self._create_encoder(self._config.pix_fmt)
        if self._cuda_device is not None:
            frame = frame.to(self._cuda_device)
        return self._encode(frame, pts, force_idr, prepared)

    def _encode(
        self, frame: torch.Tensor, pts: int, force_idr: bool, prepared: PreparedCudaFrame | None = None
    ) -> list[av.Packet]:
        # RGB CUDA [3,1080,3840] → 共享storage的CUDA NV12 → HEVC；DLPack持有帧存储。
        if prepared is not None and prepared.frame is frame and prepared.transform is self.transform:
            prepared.ready.synchronize()
            video_frame = self.transform.from_yuv_planes(*prepared.planes)
        else:
            video_frame = self.transform(frame)
        video_frame.pts = pts
        video_frame.time_base = Fraction(1, 1_000_000)
        if force_idr:
            video_frame.pict_type = av.video.frame.PictureType.I
        packets = self.encoder.encode(video_frame)
        self.last_pts = pts
        return packets
