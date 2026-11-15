import threading
import time
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F

from . import screenshot_process as capture_source


def resize_frame(frame, size):
    # CHW uint8 - > CHW float
    frame = frame.unsqueeze(0).float().div_(255.0)
    return F.interpolate(
        frame,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


class ScreenshotThreadWCCUDA(threading.Thread):
    def __init__(
        self,
        fps,
        frame_width,
        frame_height,
        monitor_index,
        window_name,
        device,
        crop_top=0,
        crop_left=0,
        crop_right=0,
        crop_bottom=0,
        window_handle=None,
        fit_window=False,
        monitor_target=None,
        **_ignore_unsupported_kwargs,
    ):
        super().__init__(daemon=True)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.monitor_index = monitor_index
        self.window_name = window_name
        self.window_handle = window_handle
        self.fit_window = fit_window
        self.monitor_target = dict(monitor_target) if monitor_target is not None else None
        if fit_window and window_handle is None and monitor_target is None:
            raise ValueError("Fitted capture requires a fixed window or monitor")
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.device = device
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.frame = None
        self.frame_mapping: dict[str, Any] | None = None
        self._frame_geometry: Any = None
        self._pending_mapping: dict[str, Any] | None = None
        self._pending_geometry: Any = None
        self._content_rect: tuple[int, int, int, int] | None = None
        self._pending_content_rect: tuple[int, int, int, int] | None = None
        self._ready_event: torch.cuda.Event | None = None
        self._pinned_pid = None
        self.frame_set_event = threading.Event()
        self.stop_event = threading.Event()
        self.fps_counter = deque(maxlen=120)
        self.tick = 0
        self._capture: Any = None

    def run(self):
        from wc_cuda import WindowsCapture

        if self.window_name or self.window_handle is not None:
            # ignore
            monitor_index = None
        else:
            # 1 origin
            monitor_index = self.monitor_index + 1

        capture = WindowsCapture(
            cursor_capture=None,
            draw_border=None,
            monitor_index=monitor_index,
            # native 优先匹配标题；固定 HWND 时必须清空标题。
            window_name=None if self.window_handle is not None else self.window_name,
            window_hwnd=self.window_handle,
            device_id=self.device.index,
        )
        self._capture = capture
        if self.fit_window and self.window_handle is not None:
            geometry = self._get_geometry()
            if geometry is not None:
                self._pinned_pid = geometry[0]

        @capture.event
        def on_frame_arrived(frame, capture_control):
            if self.stop_event.is_set():
                capture_control.stop()
                return
            geometry = self._get_geometry() if self.fit_window else None
            with self.frame_lock:
                if geometry != self._frame_geometry:
                    self.frame_mapping = None
            mapping = None
            content_rect = (0, 0, self.frame_width, self.frame_height)
            with torch.inference_mode():
                # 原生输入 uint8[H,W,4] BGRA，输出 float32[3,canvas_H,canvas_W] RGB，范围 [0,1]。
                source_frame = frame.frame_buffer[..., [2, 1, 0]].permute(2, 0, 1)
                window_capture = self.window_name or self.window_handle is not None
                crops = (
                    (self.crop_top, self.crop_left, self.crop_right, self.crop_bottom)
                    if window_capture
                    else (0, 0, 0, 0)
                )
                if window_capture and (
                    self.crop_top > 0 or self.crop_left > 0 or self.crop_right > 0 or self.crop_bottom > 0
                ):
                    h, w = source_frame.shape[-2:]
                    top = self.crop_top
                    bottom = h - self.crop_bottom if self.crop_bottom > 0 else h
                    left = self.crop_left
                    right = w - self.crop_right if self.crop_right > 0 else w
                    source_frame = source_frame[:, top:bottom, left:right]

                source_height, source_width = source_frame.shape[-2:]
                if window_capture or self.fit_window:
                    dest_frame = source_frame.new_zeros((3, self.frame_height, self.frame_width), dtype=torch.float32)
                    if source_width > 0 and source_height > 0:
                        scale = min(self.frame_width / source_width, self.frame_height / source_height)
                        width = max(1, min(self.frame_width, round(source_width * scale)))
                        height = max(1, min(self.frame_height, round(source_height * scale)))
                        left, top = (self.frame_width - width) // 2, (self.frame_height - height) // 2
                        resized = resize_frame(source_frame, size=(height, width))
                        dest_frame[:, top : top + height, left : left + width] = resized
                        content_rect = (left, top, left + width, top + height)
                        if self.fit_window and geometry is not None and geometry == self._get_geometry():
                            if self.window_handle is not None:
                                pid, window_rect, capture_rect = geometry
                                if pid == self._pinned_pid:
                                    mapping = {
                                        "hwnd": self.window_handle,
                                        "pid": pid,
                                        "window_rect": window_rect,
                                        "capture_rect": capture_rect,
                                    }
                            elif geometry == self.monitor_target:
                                mapping = {"target": "monitor", **geometry, "capture_rect": geometry["monitor_rect"]}
                            if mapping is not None:
                                cl, ct, cr, cb = mapping["capture_rect"]
                                if frame.frame_buffer.shape[:2] != (cb - ct, cr - cl):
                                    mapping = None
                                else:
                                    mapping.update(
                                        backend="wc_cuda",
                                        crops=crops,
                                        content_rect=(left, top, left + width, top + height),
                                        buffer_size=(self.frame_width, self.frame_height),
                                    )
                elif source_frame.shape != (3, self.frame_height, self.frame_width):
                    dest_frame = resize_frame(source_frame, size=(self.frame_height, self.frame_width))
                else:
                    dest_frame = source_frame / 255.0
                ready_event = None
                if dest_frame.is_cuda:
                    producer_stream = torch.cuda.current_stream(dest_frame.device)
                    frame.frame_buffer.record_stream(producer_stream)
                    ready_event = torch.cuda.Event()
                    ready_event.record(producer_stream)

            with self.frame_lock:
                if self.stop_event.is_set():
                    return
                if mapping is None or geometry != self._frame_geometry:
                    self.frame_mapping = None
                self.frame = dest_frame
                self._pending_mapping = mapping
                self._pending_geometry = geometry
                self._pending_content_rect = content_rect
                self._ready_event = ready_event
                self.frame_set_event.set()

            now = time.perf_counter()
            if self.tick > 0:
                process_time = now - self.tick
                with self.fps_lock:
                    self.fps_counter.append(process_time)
            self.tick = now

            if self.stop_event.is_set():
                capture_control.stop()

        @capture.event
        def on_closed():
            with self.frame_lock:
                self.frame_mapping = None

        try:
            # event loop
            if not self.stop_event.is_set():
                capture.start()
        finally:
            try:
                capture.stop()
            finally:
                self._capture = None
                self.stop_event.set()
                with self.frame_lock:
                    self.frame = None
                    self.frame_mapping = self._pending_mapping = None
                    self._frame_geometry = self._pending_geometry = None
                    self._content_rect = self._pending_content_rect = None
                    self._ready_event = None
                self.frame_set_event.set()
                time.sleep(0.1)

    def get_frame(self):
        while not self.frame_set_event.wait(1):
            if not self.is_alive():
                raise RuntimeError("thread is already dead")
        with self.frame_lock:
            if self.stop_event.is_set():
                raise RuntimeError("thread is dead")
            frame = self.frame
            if frame is not None and frame.is_cuda:
                consumer_stream = torch.cuda.current_stream(frame.device)
                if self._ready_event is not None:
                    consumer_stream.wait_event(self._ready_event)
                frame.record_stream(consumer_stream)
            self.frame_mapping = self._pending_mapping
            self._frame_geometry = self._pending_geometry
            self._content_rect = self._pending_content_rect
            self.frame = None
            self._pending_mapping = self._pending_geometry = self._ready_event = None
            self._pending_content_rect = None
            self.frame_set_event.clear()

        if frame is None:
            raise RuntimeError("thread is dead")

        return frame

    def _get_geometry(self):
        if self.window_handle is not None:
            return capture_source._window_geometry(self.window_handle, "windows_capture")
        if self.monitor_target is not None:
            return capture_source._monitor_geometry(self.monitor_index)
        return None

    def get_frame_mapping(self) -> dict[str, Any] | None:
        with self.frame_lock:
            if self.stop_event.is_set() or self.frame_mapping is None:
                return None
            if self._get_geometry() != self._frame_geometry:
                self.frame_mapping = None
                return None
            return dict(self.frame_mapping)

    def get_content_rect(self) -> tuple[int, int, int, int] | None:
        with self.frame_lock:
            return None if self.stop_event.is_set() else self._content_rect

    def get_fps(self):
        with self.fps_lock:
            if self.fps_counter:
                mean_processing_time = sum(self.fps_counter) / len(self.fps_counter)
                return 1 / mean_processing_time
            else:
                return 0

    def stop(self) -> None:
        self.stop_event.set()
        with self.frame_lock:
            self._content_rect = self._pending_content_rect = None
        deadline = time.monotonic() + 4
        while self.is_alive():
            capture = self._capture
            if capture is not None:
                # wc_cuda.start 初始化后会覆盖之前的 stop；退出前必须再次停止。
                capture.stop()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("CUDA capture thread did not stop within 4 seconds")
            self.join(timeout=min(0.1, remaining))
