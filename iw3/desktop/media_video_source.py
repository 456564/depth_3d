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

from .media_http_source import QUALITY_OPTIONS, MediaHTTPError, MediaHTTPIO, select_renditions, validate_source

SOURCE_GRACE_SECONDS = 3.0  # 允许初始分片准备；持续不可读三秒后才请求显式录屏。


@dataclass(frozen=True)
class MediaSnapshot:
    generation: str
    data: bytes = b""
    revision: int = 0
    media_time: float = 0.0
    paused: bool = False
    playback_rate: float = 1.0
    received_at: float = 0.0
    duration: float | None = None
    source: dict[str, Any] | None = None

    def position_at(self, now: float) -> float | None:
        elapsed = now - self.received_at
        if (
            not self.generation
            or not self.data
            and self.source is None
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
    @staticmethod
    def _deny_side_input(url: str, flags: int, options: dict[str, str]) -> io.RawIOBase:
        raise MediaHTTPError("媒体禁止引用额外资源")

    def __init__(self, data: bytes | io.RawIOBase, device: torch.device, *, live: bool = False) -> None:
        self.device = device
        self.live = live
        hwaccel = create_hwaccel("cuda", device_id=device.index or 0, disable_software_fallback=True)
        self.container = av.open(
            io.BytesIO(data) if isinstance(data, bytes) else data,
            mode="r",
            format="flv" if live else None if isinstance(data, bytes) else "mov",
            hwaccel=hwaccel,
            timeout=(2.0, 2.0),
            io_open=self._deny_side_input,
        )
        try:
            self.stream = self.container.streams.video[0]
            self.width, self.height = self.stream.codec_context.width, self.stream.codec_context.height
            if not 0 < self.width <= 8192 or not 0 < self.height <= 8192:
                raise ValueError("Unsupported source video dimensions")
            self.codec = self.stream.codec_context.name
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
        # 网页0秒可能早于MP4首帧；只钳制起点，不平移后续播放时钟。
        if self.stream.start_time is not None:
            target = max(target, float(self.stream.start_time * self.stream.time_base))
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
            self._rgb = self._to_rgb(self._selected)
        return self._rgb, timestamp

    def _to_rgb(self, frame: av.VideoFrame) -> torch.Tensor:
        if not self.stream.codec_context.is_hwaccel or frame.format.name != "cuda":
            raise RuntimeError("NVDEC hardware frames are required")
        if frame.is_corrupt or frame.sw_format is None or frame.sw_format.name not in {"nv12", "p010le"}:
            raise RuntimeError("Unsupported or corrupt NVDEC video frame")
        if tuple(frame.planes[0].__dlpack_device__()) != (2, self.device.index or 0):
            raise RuntimeError("NVDEC and AI must use the same CUDA device")
        if not 0 < frame.width <= 8192 or not 0 < frame.height <= 8192:
            raise RuntimeError("Unsupported NVDEC frame dimensions")
        # 实测：HEVC FLV → AVFrame cuda/NV12 → float32 RGB[3,1920,1088]，范围[0,1]。
        return ColorTransform.to_tensor(frame, device=self.device).squeeze(0)

    def next_frame(self, *, convert: bool = True) -> tuple[torch.Tensor | None, float]:
        if self._frames is None:
            self._frames = iter(self.container.decode(self.stream))
        frame = next(self._frames)
        if not self.stream.codec_context.is_hwaccel or frame.format.name != "cuda":
            raise RuntimeError("NVDEC hardware frames are required")
        return (self._to_rgb(frame) if convert else None), self._timestamp(frame)

    def close(self) -> None:
        self._selected = self._next = self._rgb = None
        self.container.close()


class MediaVideoSource(threading.Thread):
    def __init__(
        self,
        snapshot: Callable[[], MediaSnapshot | None],
        device: torch.device,
        fps: float = 30,
        quality: str = "auto",
        state: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        if device.type != "cuda":
            raise ValueError("Browser video source requires a CUDA device and NVDEC")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Source frame rate must be positive")
        if quality not in QUALITY_OPTIONS:
            raise ValueError("Invalid browser quality")
        self.snapshot = snapshot
        self.device = device
        self.fps = fps
        self.quality = quality
        self.state = state
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
        self._http: MediaHTTPIO | None = None
        self._choices: list[tuple[dict[str, Any], str]] = []
        self._choice = 0
        self._source_key: tuple[str, str] | None = None
        self._decode_epoch = 0
        self._last_live_publish = -math.inf
        self._live_reconnects = 0
        self._waiting_since: float | None = None
        self._waiting_generation: str | None = None
        self._unreadable_since: float | None = None
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
            "selected_quality": None,
            "available_quality": [],
            "fallback_reason": "",
            "fallback_required": False,
            "failure_generation": None,
            "codec": None,
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
        reader, self._reader = self._reader, None
        handle, self._http = self._http, None
        try:
            if reader is not None:
                reader.close()
        finally:
            if handle is not None:
                handle.close()

    def _set_status(self, status: str, message: str, **values: Any) -> None:
        with self._lock:
            self._status.update(status=status, message=message, hardware_active=status == "running", **values)
            if status != "running":
                self._status["source_size"] = None
            if self._status.get("fallback_required") is not True:
                self._status["failure_generation"] = None

    def _mark_unreadable(self, current: MediaSnapshot, now: float) -> None:
        if self._unreadable_since is None:
            self._unreadable_since = now
        expired = now - self._unreadable_since >= SOURCE_GRACE_SECONDS
        status = self.get_status()
        self._set_status(
            status["status"],
            status["message"],
            fallback_required=expired,
            fallback_reason="网页分片持续不可解码或缺少播放位置" if expired else "",
            failure_generation=current.generation if expired else None,
        )

    @torch.inference_mode()
    def _update(self, current: MediaSnapshot | None, now: float) -> None:
        target = current.position_at(now) if current is not None else None
        if current is None or target is None:
            self._clear_slot(wake=True)
            self._close_reader()
            if self._source_key is not None:
                self._unreadable_since = None
            operation = self.state() if self.state is not None else None
            waiting_generation = (
                operation.get("generation")
                if operation is not None and operation.get("fresh") is True
                else None
                if operation is not None
                else current.generation
                if current is not None
                else self._generation
            )
            if self._waiting_since is None or waiting_generation != self._waiting_generation:
                self._waiting_since = now
                self._waiting_generation = waiting_generation
            expired = now - self._waiting_since >= SOURCE_GRACE_SECONDS and (
                operation is None or waiting_generation is not None
            )
            self._set_status(
                "waiting",
                "等待当前网页的视频数据与播放状态。",
                fallback_required=expired,
                fallback_reason="网页未提供可读媒体来源" if expired else "",
                failure_generation=waiting_generation if expired else None,
            )
            return
        self._waiting_since = None
        if self._generation != current.generation:
            self._clear_slot()
            self._close_reader()
            self._generation = current.generation
            self._last_sent = self._failed = None
            self._failed_target = None
            self._source_key = None
            self._unreadable_since = None
            self._set_status(
                "waiting",
                "正在切换网页媒体。",
                fallback_required=False,
                selected_quality=None,
                available_quality=[],
                fallback_reason="",
                codec=None,
                input_source="网页压缩视频（不录屏）",
                audio_path="Windows 网页进程回环；视频跟随网页播放时钟",
            )
            with self._lock:
                self._frame_times.clear()
        if current.source is not None:
            self._update_remote(current, now, target)
            return
        key = (current.generation, current.revision)
        if key == self._failed and self._failed_target is not None and abs(target - self._failed_target) <= 0.5:
            self._mark_unreadable(current, now)
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
                self._mark_unreadable(current, now)
                return
            self._failed = self._failed_target = None
            self._unreadable_since = None
            frame, timestamp = result
            self._set_status(
                "running",
                "网页视频源，NVDEC 硬解。",
                source_size=(self._reader.width, self._reader.height),
                fallback_required=False,
                fallback_reason="",
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
            self._mark_unreadable(current, now)

    def _remote_cancelled(self, generation: str) -> bool:
        if self.stop_event.is_set():
            return True
        current = self.snapshot()
        return current is None or current.generation != generation or current.position_at(time.monotonic()) is None

    def _generation_token(self, current: MediaSnapshot) -> str:
        # 同一网页操作内的降档/重连也要隔离已在 AI 中的旧帧。
        return current.generation if current.source is None else f"{current.generation}:nvdec:{self._decode_epoch}"

    @torch.inference_mode()
    def _update_remote(self, current: MediaSnapshot, now: float, target: float) -> None:
        description = current.source
        assert description is not None
        source_key = (current.generation, description["id"])
        if self._source_key != source_key:
            self._close_reader()
            self._clear_slot()
            self._source_key = source_key
            self._choice = self._live_reconnects = 0
            self._last_live_publish = -math.inf
            self._unreadable_since = None
            try:
                description = validate_source(description)
                renditions = select_renditions(description, self.quality)
                self._choices = [(rendition, url) for rendition in renditions for url in rendition["urls"]]
            except ValueError:
                self._choices = []
            available = sorted(
                {
                    min(item["width"], item["height"])
                    for item in description["renditions"]
                    if item["width"] and item["height"]
                },
                reverse=True,
            )
            self._set_status(
                "waiting",
                "正在读取网页原媒体。",
                fallback_required=False,
                fallback_reason="",
                selected_quality=None,
                available_quality=available,
                codec=None,
            )
        live = description["kind"] == "live"
        reason = self.get_status()["fallback_reason"]
        while self._choice < len(self._choices) and not self._remote_cancelled(current.generation):
            rendition, url = self._choices[self._choice]
            stage = "打开媒体"
            try:
                if self._reader is None:
                    self._http = MediaHTTPIO(url, lambda: self._remote_cancelled(current.generation), live=live)
                    self._reader = _NVDECReader(self._http, self.device, live=live)
                    with self._lock:
                        self._decode_epoch += 1
                    if self.quality != "auto" and min(self._reader.width, self._reader.height) > int(self.quality):
                        raise MediaHTTPError("实际源尺寸超过所选画质上限")
                latest = self.snapshot()
                if latest is not None:
                    fresh_target = latest.position_at(time.monotonic())
                    if fresh_target is not None:
                        target = fresh_target
                stage = "解码帧"
                if live:
                    publish = (not current.paused or not self._valid) and now - self._last_live_publish >= 1 / self.fps
                    frame, timestamp = self._reader.next_frame(convert=publish)
                    if not publish or frame is None:
                        return
                else:
                    result = self._reader.frame_at(target)
                    if result is None:
                        checked_at = time.monotonic()
                        if self._unreadable_since is None:
                            self._unreadable_since = checked_at
                        if checked_at - self._unreadable_since >= SOURCE_GRACE_SECONDS:
                            raise MediaHTTPError("原视频持续缺少当前播放位置")
                        self._clear_slot(wake=True)
                        self._set_status(
                            "waiting",
                            "等待原视频播放位置就绪。",
                            fallback_required=False,
                            fallback_reason=reason,
                        )
                        return
                    frame, timestamp = result
                if self._remote_cancelled(current.generation):
                    self._clear_slot(wake=True)
                    self._close_reader()
                    return
                quality = min(self._reader.width, self._reader.height)
                self._unreadable_since = None
                self._set_status(
                    "running",
                    "直播原流，NVDEC 硬解。" if live else "网页原视频，NVDEC 硬解。",
                    source_size=(self._reader.width, self._reader.height),
                    selected_quality=f"{quality}p",
                    fallback_required=False,
                    fallback_reason=reason,
                    codec=self._reader.codec,
                    input_source="直播原流（不录屏）" if live else "短视频原流（不录屏）",
                    audio_path="Windows 网页进程回环；直播视频独立读取，音画延迟待校准"
                    if live
                    else "Windows 网页进程回环；视频跟随网页播放时钟",
                )
                with self._lock:
                    self._valid = True
                frame_generation = self._generation_token(current)
                if (frame_generation, timestamp) == self._last_sent:
                    return
                ready = None
                if frame.is_cuda:
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(frame.device))
                with self._lock:
                    if self.stop_event.is_set():
                        return
                    self._frame, self._ready = frame, ready
                    self._pending_generation = frame_generation
                    self._frame_times.append(now)
                    self.frame_set_event.set()
                self._last_sent = (frame_generation, timestamp)
                self._last_live_publish = now
                return
            except Exception as exc:
                had_frame = self._last_sent is not None
                self._clear_slot(wake=True)
                self._close_reader()
                self._unreadable_since = None
                if self._remote_cancelled(current.generation):
                    return
                code = getattr(exc, "errno", None)
                detail = f"，错误码{code}" if type(code) is int else ""
                reason = (
                    str(exc)
                    if isinstance(exc, MediaHTTPError)
                    else f"NVDEC {stage}失败（{type(exc).__name__}{detail}）"
                )
                # 已成功的直播只重连当前 URL 两次；初次失败仍按同档备用、下一档顺序。
                if live and had_frame and self._live_reconnects < 2:
                    self._live_reconnects += 1
                    self.stop_event.wait(0.25 * self._live_reconnects)
                else:
                    self._choice += 1
        if not self._remote_cancelled(current.generation):
            self._set_status(
                "error",
                "网页原媒体不可用，需要屏幕捕获。",
                fallback_required=True,
                failure_generation=current.generation,
                fallback_reason=reason or "没有符合画质上限的可读来源",
                selected_quality=None,
                codec=None,
            )

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                current = self.snapshot()
                self._update(current, started)
                # 持续直播必须读完参考帧；只限制 RGB 转换/发布，不能按目标 FPS 节流网络。
                draining = current is not None and current.source is not None and current.source["kind"] == "live"
                if not draining or self._reader is None:
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
                or self._generation_token(current) != self._pending_generation
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
            and self._generation_token(current) == generation
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
        # 先打断网络读取，再等待 owner 在线程内部关闭 FFmpeg；不跨线程 close container。
        handle = self._http
        if handle is not None:
            handle.close()
        if self.is_alive():
            self.join(timeout=4)
            if self.is_alive():
                raise RuntimeError("NVDEC video source did not stop within 4 seconds")
        else:
            self._close_reader()
            self._clear_slot()
        self.frame_set_event.set()
