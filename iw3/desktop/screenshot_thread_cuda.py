import threading
import time
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F


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
        **_ignore_unsupported_kwargs,
    ):
        super().__init__(daemon=True)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.monitor_index = monitor_index
        self.window_name = window_name
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.device = device
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.frame = None
        self.frame_set_event = threading.Event()
        self.stop_event = threading.Event()
        self.fps_counter = deque(maxlen=120)
        self.tick = 0
        self._capture: Any = None

    def run(self):
        from wc_cuda import WindowsCapture

        if self.window_name:
            # ignore
            monitor_index = None
        else:
            # 1 origin
            monitor_index = self.monitor_index + 1

        capture = WindowsCapture(
            cursor_capture=None,
            draw_border=None,
            monitor_index=monitor_index,
            window_name=self.window_name,
            device_id=self.device.index,
        )
        self._capture = capture

        @capture.event
        def on_frame_arrived(frame, capture_control):
            # NOTE: wc_cuda already ensures that `frame` uses dedicated memory and is not reused.
            with torch.inference_mode():
                # BGRA HWC -> RGB CHW
                source_frame = frame.frame_buffer[..., [2, 1, 0]].permute(2, 0, 1)
                if self.window_name and (
                    self.crop_top > 0 or self.crop_left > 0 or self.crop_right > 0 or self.crop_bottom > 0
                ):
                    h, w = source_frame.shape[-2:]
                    top = self.crop_top
                    bottom = h - self.crop_bottom if self.crop_bottom > 0 else h
                    left = self.crop_left
                    right = w - self.crop_right if self.crop_right > 0 else w
                    source_frame = source_frame[:, top:bottom, left:right]

                if source_frame.shape != (3, self.frame_height, self.frame_width):
                    if self.window_name is not None:
                        min_h = min(self.frame_height, source_frame.shape[1])
                        min_w = min(self.frame_width, source_frame.shape[2])
                        dest_frame = torch.zeros(
                            (3, self.frame_height, self.frame_width), dtype=torch.float32, device=self.device
                        )
                        dest_frame[:, 0:min_h, 0:min_w].copy_(source_frame[:, 0:min_h, 0:min_w]).div_(255.0)
                    else:
                        dest_frame = resize_frame(source_frame, size=(self.frame_height, self.frame_width))
                else:
                    dest_frame = source_frame / 255.0

            with self.frame_lock:
                self.frame = dest_frame
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
            pass

        try:
            # event loop
            if not self.stop_event.is_set():
                capture.start()
        finally:
            try:
                capture.stop()
            finally:
                self._capture = None
                self.frame_set_event.set()
                time.sleep(0.1)

    def get_frame(self):
        while not self.frame_set_event.wait(1):
            if not self.is_alive():
                raise RuntimeError("thread is already dead")
        with self.frame_lock:
            frame = self.frame
            self.frame = None
            self.frame_set_event.clear()

        if frame is None:
            raise RuntimeError("thread is dead")

        return frame

    def get_fps(self):
        with self.fps_lock:
            if self.fps_counter:
                mean_processing_time = sum(self.fps_counter) / len(self.fps_counter)
                return 1 / mean_processing_time
            else:
                return 0

    def stop(self) -> None:
        self.stop_event.set()
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
