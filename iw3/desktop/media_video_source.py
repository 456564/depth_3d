"""浏览器压缩视频的 NVDEC 单槽源；播放时钟由网页提供。"""

import io
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import av
import torch

from nunif.utils.video.color_transform import ColorTransform
from nunif.utils.video.hwaccel import create_hwaccel


@dataclass(frozen=True)
class MediaSnapshot:
    generation: str
    data: bytes
    revision: int
    media_time: float
    paused: bool
    playback_rate: float
    received_at: float
    duration: float | None = None

    def position_at(self, now: float) -> float | None:
        elapsed = now - self.received_at
        if (
            not self.generation
            or not self.data
            or not all(math.isfinite(value) for value in (now, elapsed, self.media_time, self.playback_rate))
            or not 0 <= elapsed <= 1.0
            or self.media_time < 0
            or not 0 < self.playback_rate <= 16
        ):
            return None
        position = self.media_time + (0 if self.paused else elapsed * self.playback_rate)
        if self.duration is not None and math.isfinite(self.duration) and self.duration > 0:
            position = min(position, self.duration)
        return position


class _NVDECReader:
    def __init__(self, data: bytes, device: torch.device) -> None:
        self.device = device
        hwaccel = create_hwaccel("cuda", device_id=device.index or 0, disable_software_fallback=True)
        self.container = av.open(io.BytesIO(data), mode="r", hwaccel=hwaccel)
        try:
            self.stream = self.container.streams.video[0]
            self.width, self.height = self.stream.codec_context.width, self.stream.codec_context.height
            if not 0 < self.width <= 8192 or not 0 < self.height <= 8192:
                raise ValueError("Unsupported source video dimensions")
            rate = self.stream.average_rate or self.stream.guessed_rate
            self.fps = float(rate) if rate else 30.0
            if not math.isfinite(self.fps) or self.fps <= 0:
                self.fps = 30.0
        except Exception:
            self.container.close()
            raise
        self._frames: Iterator[av.VideoFrame] | None = None
        self._selected: av.VideoFrame | None = None
        self._next: av.VideoFrame | None = None
        self._rgb: torch.Tensor | None = None

    @staticmethod
    def _timestamp(frame: av.VideoFrame) -> float:
        if frame.pts is None or frame.time_base is None:
            raise ValueError("Video frame has no presentation timestamp")
        return float(frame.pts * frame.time_base)

    def frame_at(self, target: float) -> tuple[torch.Tensor, float] | None:
        selected_time = self._timestamp(self._selected) if self._selected is not None else None
        if self._frames is None or (
            selected_time is not None and (target < selected_time or target > selected_time + 0.5)
        ):
            self.container.seek(int(target / self.stream.time_base), stream=self.stream, backward=True)
            self._frames = iter(self.container.decode(self.stream))
            self._selected = self._next = self._rgb = None
        old_selected = self._selected
        while True:
            if self._next is None:
                self._next = next(self._frames, None)
            if self._next is None or self._timestamp(self._next) > target:
                break
            self._selected, self._next = self._next, None
        if self._selected is None:
            return None
        timestamp = self._timestamp(self._selected)
        if target - timestamp > max(0.1, 2 / self.fps):
            return None
        if self._rgb is None or self._selected is not old_selected:
            frame = self._selected
            if not self.stream.codec_context.is_hwaccel or frame.format.name != "cuda":
                raise RuntimeError("NVDEC hardware frames are required")
            if frame.is_corrupt or frame.sw_format is None or frame.sw_format.name not in {"nv12", "p010le"}:
                raise RuntimeError("Unsupported or corrupt NVDEC video frame")
            if tuple(frame.planes[0].__dlpack_device__()) != (2, self.device.index or 0):
                raise RuntimeError("NVDEC and AI must use the same CUDA device")
            # 本机实测：压缩 H.264 → AVFrame cuda/NV12 → float32 RGB[3,1920,1080]，范围[0,1]。
            self._rgb = ColorTransform.to_tensor(frame, device=self.device).squeeze(0)
        return self._rgb, timestamp

    def close(self) -> None:
        self._selected = self._next = self._rgb = None
        self.container.close()


class MediaVideoSource(threading.Thread):
    def __init__(
        self,
        snapshot: Callable[[], MediaSnapshot | None],
        device: torch.device,
        fps: float = 30,
    ) -> None:
        super().__init__(daemon=True)
        if device.type != "cuda":
            raise ValueError("Browser video source requires a CUDA device and NVDEC")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Source frame rate must be positive")
        self.snapshot = snapshot
        self.device = device
        self.fps = fps
        self.frame_set_event = threading.Event()
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame: torch.Tensor | None = None
        self._ready: torch.cuda.Event | None = None
        self._pending_generation: str | None = None
        self._frame_generation: str | None = None
        self._valid = False
        self._content_rect: tuple[int, int, int, int] | None = None
        self._reader: _NVDECReader | None = None
        self._generation: str | None = None
        self._revision = -1
        self._failed: tuple[str, int] | None = None
        self._failed_target: float | None = None
        self._last_sent: tuple[str, float] | None = None
        self._frame_times: deque[float] = deque(maxlen=120)
        self._status: dict[str, Any] = {
            "status": "waiting",
            "message": "等待浏览器提供当前视频的压缩数据。",
            "decoder": "NVDEC",
            "hardware_active": False,
            "source_size": None,
        }

    def _clear_slot(self, wake: bool = False) -> None:
        with self._lock:
            was_valid = self._valid
            self._frame = self._ready = self._pending_generation = self._content_rect = None
            self._frame_generation = self._last_sent = None
            self._valid = False
            if wake and was_valid:
                self.frame_set_event.set()
            elif not wake:
                self.frame_set_event.clear()

    def _close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _set_status(self, status: str, message: str, **values: Any) -> None:
        with self._lock:
            self._status.update(status=status, message=message, hardware_active=status == "running", **values)
            if status != "running":
                self._status["source_size"] = None

    @torch.inference_mode()
    def _update(self, current: MediaSnapshot | None, now: float) -> None:
        target = current.position_at(now) if current is not None else None
        if current is None or target is None:
            self._clear_slot(wake=True)
            self._set_status("waiting", "等待当前网页的视频数据与播放状态。")
            return
        if self._generation != current.generation:
            self._clear_slot()
            self._close_reader()
            self._generation = current.generation
            self._last_sent = self._failed = None
            self._failed_target = None
            with self._lock:
                self._frame_times.clear()
        key = (current.generation, current.revision)
        if key == self._failed and self._failed_target is not None and abs(target - self._failed_target) <= 0.5:
            return
        try:
            if self._reader is None:
                self._reader = _NVDECReader(current.data, self.device)
                self._revision = current.revision
            result = self._reader.frame_at(target)
            if result is None and current.revision != self._revision:
                self._close_reader()
                self._reader = _NVDECReader(current.data, self.device)
                self._revision = current.revision
                result = self._reader.frame_at(target)
            if result is None:
                self._clear_slot(wake=True)
                self._set_status("waiting", "等待当前播放位置的视频分片。")
                return
            self._failed = self._failed_target = None
            frame, timestamp = result
            self._set_status(
                "running", "网页视频源，NVDEC 硬解。", source_size=(self._reader.width, self._reader.height)
            )
            with self._lock:
                self._valid = True
            if (current.generation, timestamp) == self._last_sent:
                return
            ready = None
            if frame.is_cuda:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(frame.device))
            with self._lock:
                if self.stop_event.is_set():
                    return
                self._frame = frame
                self._ready = ready
                self._pending_generation = current.generation
                self._frame_times.append(now)
                self.frame_set_event.set()
            self._last_sent = (current.generation, timestamp)
        except Exception as exc:
            self._clear_slot(wake=True)
            self._close_reader()
            self._failed = key
            self._failed_target = target
            self._set_status("error", f"网页视频 NVDEC 解码失败（{type(exc).__name__}）；等待新的视频数据。")

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                self._update(self.snapshot(), started)
                self.stop_event.wait(max(0, 1 / self.fps - (time.monotonic() - started)))
        finally:
            self._close_reader()
            self._clear_slot()
            self.frame_set_event.set()

    def get_frame(self) -> torch.Tensor | None:
        current = self.snapshot()
        now = time.monotonic()
        with self._lock:
            frame = self._frame
            self._frame = None
            self.frame_set_event.clear()
            if (
                self.stop_event.is_set()
                or current is None
                or current.position_at(now) is None
                or current.generation != self._pending_generation
            ):
                self._content_rect = None
                self._ready = self._pending_generation = None
                self._frame_generation = self._last_sent = None
                self._valid = False
                return None
            if frame is not None:
                if frame.is_cuda:
                    consumer = torch.cuda.current_stream(frame.device)
                    if self._ready is not None:
                        consumer.wait_event(self._ready)
                    frame.record_stream(consumer)
                self._content_rect = (0, 0, frame.shape[-1], frame.shape[-2])
                self._frame_generation = self._pending_generation
            self._ready = self._pending_generation = None
            return frame

    def get_content_rect(self) -> tuple[int, int, int, int] | None:
        with self._lock:
            return self._content_rect

    def get_frame_generation(self) -> str | None:
        with self._lock:
            return self._frame_generation

    def is_current_generation(self, generation: str | None) -> bool:
        current = self.snapshot()
        with self._lock:
            valid = self._valid
        return (
            not self.stop_event.is_set()
            and valid
            and generation is not None
            and current is not None
            and current.generation == generation
            and current.position_at(time.monotonic()) is not None
        )

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def get_fps(self) -> float:
        with self._lock:
            if len(self._frame_times) < 2 or time.monotonic() - self._frame_times[-1] > 1.0:
                return 0.0
            elapsed = self._frame_times[-1] - self._frame_times[0]
            return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    def stop(self) -> None:
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=4)
            if self.is_alive():
                raise RuntimeError("NVDEC video source did not stop within 4 seconds")
        else:
            self._close_reader()
            self._clear_slot()
        self.frame_set_event.set()
