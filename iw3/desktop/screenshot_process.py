import ctypes
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch
import wx
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .screenshot_thread_kwcapture import (
    enum_window_names as kwcapture_enum_window_names,
)
from .screenshot_thread_kwcapture import (
    get_monitor_size_list as kwcapture_get_monitor_size_list,
)
from .screenshot_thread_kwcapture import (
    get_window_rect_by_title as kwcapture_get_window_rect_by_title,
)
from .screenshot_thread_kwcapture import (  # noqa
    is_kwcapture_supported,
)

_x11_connection_pool: dict[int, Any] = {}
_mss_pool: dict[int, Any] = {}


def _window_geometry(hwnd, backend):
    import win32gui
    import win32process

    try:
        if not win32gui.IsWindow(hwnd):
            return None
        pid = win32process.GetWindowThreadProcessId(hwnd)[1]
        window_rect = win32gui.GetWindowRect(hwnd)
        capture_rect = window_rect
        if backend == "windows_capture":

            class RECT(ctypes.Structure):
                _fields_ = [(name, ctypes.c_int32) for name in ("left", "top", "right", "bottom")]

            rect = RECT()
            dwm = ctypes.WinDLL("dwmapi")
            dwm.DwmGetWindowAttribute.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
            dwm.DwmGetWindowAttribute.restype = ctypes.c_int32
            if dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
                return None
            capture_rect = (rect.left, rect.top, rect.right, rect.bottom)
        return pid, window_rect, capture_rect
    except (win32gui.error, OSError):
        return None


def _monitor_geometry(index):
    import win32api

    if type(index) is not int or index < 0:
        return None
    try:
        monitors = win32api.EnumDisplayMonitors()
        if index >= len(monitors):
            return None
        handle, _, rect = monitors[index]
        info = win32api.GetMonitorInfo(handle)
        if tuple(info["Monitor"]) != tuple(rect):
            return None
        return {
            "monitor_index": index,
            "monitor_handle": int(handle),
            "monitor_device": info["Device"],
            "monitor_rect": tuple(rect),
        }
    except Exception:
        # 热插拔时枚举与查询可能跨越设备销毁，拒绝沿用旧坐标。
        return None


def _decode_frame_mapping(values, monitor_target=None):
    if not values[0]:
        return None
    if values[3] in {3, 4}:
        if (
            monitor_target is None
            or monitor_target["monitor_handle"] != values[1]
            or monitor_target["monitor_index"] != values[2]
            or monitor_target["monitor_rect"] != tuple(values[4:8])
        ):
            return None
        return {
            "target": "monitor",
            **monitor_target,
            "backend": "mss" if values[3] == 3 else "windows_capture",
            "capture_rect": tuple(values[8:12]),
            "crops": tuple(values[12:16]),
            "content_rect": tuple(values[16:20]),
            "buffer_size": tuple(values[20:22]),
        }
    return {
        "hwnd": values[1],
        "pid": values[2],
        "backend": "mss" if values[3] == 1 else "windows_capture",
        "window_rect": tuple(values[4:8]),
        "capture_rect": tuple(values[8:12]),
        "crops": tuple(values[12:16]),
        "content_rect": tuple(values[16:20]),
        "buffer_size": tuple(values[20:22]),
    }


def is_linux_x11():
    return sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE", "").lower() == "x11"


def is_mss_supported():
    if sys.platform == "linux":
        return is_linux_x11()
    else:
        return True


def get_x11():
    key = os.getpid()
    display, root = _x11_connection_pool.get(key, (None, None))
    if display is None:
        from Xlib import display as Xdisplay

        display = Xdisplay.Display()
        root = display.screen().root
        _x11_connection_pool[key] = (display, root)

    return display, root


def get_x11root():
    display, root = get_x11()
    return root


def get_mss():
    key = os.getpid()
    sct = _mss_pool.get(key)
    if sct is None:
        import mss

        sct = mss.mss(with_cursor=True)
        _mss_pool[key] = sct

    return sct


class FrameMSS:
    def __init__(self, frame):
        self.frame_buffer = frame


class CaptureControlMSS:
    def __init__(self):
        self._stop = False
        self.frame_pos = [0, 0]
        self.frame_geometry = None

    def stop(self):
        self._stop = True


class WindowsCaptureMSS:
    def __init__(self, monitor_index=0, window_name=None, window_handle=None, monitor_target=None):
        self.window_name = window_name
        self.monitor_index = monitor_index
        self.window_handle = window_handle
        self.monitor_target = monitor_target

    def event(self, handler):
        if handler.__name__ == "on_frame_arrived":
            self.on_frame_arrived = handler
        elif handler.__name__ == "on_closed":
            self.on_closed = handler
        else:
            raise ValueError(handler.__name__)

    def start(self):
        import mss

        control = CaptureControlMSS()
        while True:
            with mss.mss(with_cursor=True) as sct:
                tick = time.perf_counter()
                if self.window_handle is not None:
                    geometry = _window_geometry(self.window_handle, "mss")
                    if geometry is None:
                        raise RuntimeError("Capture window is no longer available")
                    _, (left, top, right, bottom), _ = geometry
                    control.frame_geometry = geometry
                    position = {"left": left, "top": top, "width": right - left, "height": bottom - top}
                    control.frame_pos = [top, left]
                elif self.window_name is not None:
                    position = get_window_rect_by_title(self.window_name, sct=sct)
                    control.frame_pos = [position["top"], position["left"]]
                elif self.monitor_target is not None:
                    current = _monitor_geometry(self.monitor_index)
                    if current != self.monitor_target:
                        raise RuntimeError("显示器布局已变化，请重启串流")
                    left, top, right, bottom = current["monitor_rect"]
                    control.monitor_geometry = current
                    position = {"left": left, "top": top, "width": right - left, "height": bottom - top}
                    control.frame_pos = [top, left]
                else:
                    position = sct.monitors[self.monitor_index + 1]
                shot = sct.grab(position)
                self.on_frame_arrived(FrameMSS(np.asarray(shot)), control)
                if control._stop:
                    self.on_closed()
                    break

                process_time = time.perf_counter() - tick
                wait_time = max((1 / 60) - process_time, 0)
                time.sleep(wait_time)


def draw_cursor(x, pos, size=12, offset=[0, 0], mapping=None):
    C, H, W = x.shape
    left, top, right, bottom = 0, 0, W, H
    if mapping is not None:
        cl, ct, cr, cb = mapping["capture_rect"]
        crop_top, crop_left, crop_right, crop_bottom = mapping["crops"]
        source_width, source_height = cr - cl - crop_left - crop_right, cb - ct - crop_top - crop_bottom
        px, py = pos[0] - cl - crop_left, pos[1] - ct - crop_top
        if not (0 <= px < source_width and 0 <= py < source_height):
            return
        left, top, right, bottom = mapping["content_rect"]
        pos = (left + round(px * (right - left) / source_width), top + round(py * (bottom - top) / source_height))
        offset = (0, 0)
    r = min(size // 2, (right - left) // 2, (bottom - top) // 2)
    if r < 1:
        return
    rr = r // 2
    pos_x = min(max(pos[0] - offset[1], left + r), right - r)
    pos_y = min(max(pos[1] - offset[0], top + r), bottom - r)
    px = x[:, pos_y - rr : pos_y + rr, pos_x - rr : pos_x + rr].clone()
    color = torch.tensor((0x33 / 255.0, 0x80 / 255.0, 0x80 / 255.0), dtype=px.dtype, device=px.device).view(3, 1, 1)
    x[:, pos_y - r : pos_y + r, pos_x - r : pos_x + r] = color
    x[:, pos_y - rr : pos_y + rr, pos_x - rr : pos_x + rr] = px


def get_monitor_size_list():
    if is_kwcapture_supported():
        # On Wayland, mss sees the XWayland screen only (or nothing at all). kwcapture asks
        # KWin, so the sizes are the ones a capture will actually return (device pixels).
        size_list = kwcapture_get_monitor_size_list()
        if size_list:
            return size_list
    if sys.platform == "win32":
        import win32api

        monitors = win32api.EnumDisplayMonitors()
        size_list = []
        for monitor in monitors:
            sx, sy, width, height = monitor[2]
            width = width - sx
            height = height - sy
            size_list.append((width, height))
        return size_list
    else:  # This doesn't use any platform specific call (safe for all OS)
        monitors = get_mss().monitors[1:]
        size_list = []
        for monitor in monitors:
            size_list.append((monitor["width"], monitor["height"]))
        return size_list


def get_screen_size(monitor_index):
    size_list = get_monitor_size_list()
    return size_list[monitor_index]


DENY_WINDOW_NAMES = {"Microsoft Text Input Application", "Program Manager"}


def enum_window_names_x11():
    import Xlib
    from Xlib import X, Xatom

    d, root = get_x11()

    NET_CLIENT_LIST = d.intern_atom("_NET_CLIENT_LIST")
    NET_WM_NAME = d.intern_atom("_NET_WM_NAME")
    NET_WM_DESKTOP = d.intern_atom("_NET_WM_DESKTOP")
    NET_CURRENT_DESKTOP = d.intern_atom("_NET_CURRENT_DESKTOP")
    UTF8_STRING = d.intern_atom("UTF8_STRING")

    current_desktop_prop = root.get_full_property(NET_CURRENT_DESKTOP, X.AnyPropertyType)
    current_desktop = current_desktop_prop.value[0] if current_desktop_prop else None

    client_list_prop = root.get_full_property(NET_CLIENT_LIST, X.AnyPropertyType)
    if not client_list_prop:
        return []

    windows = []
    for win_id in client_list_prop.value:
        try:
            w = d.create_resource_object("window", win_id)
            desk_prop = w.get_full_property(NET_WM_DESKTOP, X.AnyPropertyType)
            if desk_prop:
                desktop = desk_prop.value[0]
                if current_desktop is not None and desktop != current_desktop:
                    continue

            name = None
            for atom in (NET_WM_NAME, Xatom.WM_NAME):
                prop = w.get_full_property(atom, UTF8_STRING)
                if prop:
                    name = prop.value
                    if isinstance(name, bytes):
                        name = name.decode(errors="ignore")
                    break
            if not name:
                continue

            geom = w.get_geometry()
            abs_pos = root.translate_coords(w, 0, 0)
            if geom.width < 128 or geom.height < 128:
                continue

            window_name = (
                str(name)
                + "|"
                + str(abs_pos.x)
                + ","
                + str(abs_pos.y)
                + "|"
                + str(geom.width)
                + ","
                + str(geom.height)
                + "|"
                + str(w.id)
            )
            windows.append(window_name)

        except Xlib.error.XError:
            continue

    return sorted(windows)


def enum_window_names():
    if sys.platform == "win32":
        import win32gui

        window_names = []

        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and title not in DENY_WINDOW_NAMES:
                    window_names.append(title)

        win32gui.EnumWindows(callback, None)
        return sorted(window_names)
    elif is_kwcapture_supported():
        # KWin keeps the window list; X11's _NET_CLIENT_LIST does not exist here
        return kwcapture_enum_window_names()
    elif is_linux_x11():
        return enum_window_names_x11()
    else:
        return []  # WARNING: mac is unimplemented!


def find_window_x11(address, x_display):
    import Xlib

    try:
        window = x_display.create_resource_object("window", int(address))
    except Xlib.error.XError:
        window = None

    return window


def get_window_rect_by_title(title, sct=None):
    if sys.platform == "win32":
        import win32gui

        hwnd = win32gui.FindWindow(None, title)
        if hwnd == 0:
            return None

        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        return {"left": left, "top": top, "width": width, "height": height}
    elif is_linux_x11():
        import Xlib

        x_display, x_root = get_x11()
        comp = title.rsplit("|", 4)
        if len(comp) < 4:
            return None

        ret = None
        window = find_window_x11(comp[-1], x_display)
        if window is not None:
            try:
                geom = window.get_geometry()
                abs_pos = x_root.translate_coords(window, 0, 0)
                ret = {"left": abs_pos.x, "top": abs_pos.y, "width": geom.width, "height": geom.height}
            except Xlib.error.XError:
                pass
        if ret is None:
            # Window not found falling back to initial window area
            ret = {}
            pos = comp[-3].split(",")
            ret["left"] = int(pos[0])
            ret["top"] = int(pos[1])
            size = comp[-2].split(",")
            ret["width"] = int(size[0])
            ret["height"] = int(size[1])

        # Ensure bounding box is stricly inside monitor area
        sct = sct or get_mss()

        if ret["left"] < 0:
            ret["width"] += ret["left"]
            ret["left"] = 0
        if ret["top"] < 0:
            ret["height"] += ret["top"]
            ret["top"] = 0
        if ret["width"] <= 0 or ret["height"] <= 0:
            print("window position is invalid or out of screen!", file=sys.stderr)
            # Return primary monitor area
            return dict(sct.monitors[1])

        box = sct.monitors[0]  # Combined monitor area
        if ret["left"] + ret["width"] >= box["width"]:
            ret["width"] = box["width"] - ret["left"] - 1
        if ret["top"] + ret["height"] >= box["height"]:
            ret["height"] = box["height"] - ret["top"] - 1
        if ret["width"] <= 0 or ret["height"] <= 0:
            print("window position is invalid or out of screen!", file=sys.stderr)
            # Return primary monitor area
            return dict(sct.monitors[1])

        return ret
    elif is_kwcapture_supported():
        # Device pixels, measured from a real frame: KWin's own geometry is in logical
        # pixels, which under fractional scaling is not what the capture thread returns.
        # None (window closed) is returned as-is, so the caller reports "not found"
        # instead of streaming a zero-sized area.
        return kwcapture_get_window_rect_by_title(title)
    else:
        # TODO: Not implemented
        return {"left": 0, "right": 0, "width": 0, "height": 0}


def estimate_fps(fps_counter):
    diff = []
    prev = None
    for t in fps_counter:
        if prev is not None:
            diff.append(t - prev)
        prev = t
    if diff:
        mean_time = sum(diff) / len(diff)
        return 1 / mean_time
    else:
        return 0


def capture_process(
    frame_size,
    monitor_index,
    window_name,
    frame_shm,
    frame_pos,
    frame_lock,
    frame_event,
    stop_event,
    backend="mss",
    crop_top=0,
    crop_left=0,
    crop_right=0,
    crop_bottom=0,
    window_handle=None,
    fit_window=False,
    frame_mapping=None,
    monitor_target=None,
):
    frame_buffer = np.ndarray(frame_size, dtype=np.uint8, buffer=frame_shm.buf)
    frame_count = 0
    pinned_pid = None
    if fit_window:
        if (window_handle is None and monitor_target is None) or frame_mapping is None:
            raise ValueError("Fitted window capture requires a fixed window and shared mapping")
        shcore = ctypes.WinDLL("shcore")
        shcore.SetProcessDpiAwareness.argtypes = (ctypes.c_int,)
        shcore.SetProcessDpiAwareness.restype = ctypes.c_int32
        shcore.SetProcessDpiAwareness(2)
        if monitor_target is not None:
            if window_handle is not None or window_name or any((crop_top, crop_left, crop_right, crop_bottom)):
                raise ValueError("桌面控制要求显示器目标且不裁剪")
            if _monitor_geometry(monitor_target["monitor_index"]) != monitor_target:
                raise RuntimeError("显示器布局已变化，请重启串流")
        else:
            geometry = _window_geometry(window_handle, backend)
            if geometry is None:
                raise RuntimeError("Capture window is no longer available")
            pinned_pid = geometry[0]

    if backend == "mss":
        options = {"window_handle": window_handle} if window_handle is not None else {}
        if monitor_target is not None:
            options["monitor_target"] = monitor_target
        capture = WindowsCaptureMSS(monitor_index=monitor_index, window_name=window_name, **options)
    elif backend == "windows_capture":
        try:
            from windows_capture import WindowsCapture
        except ImportError:
            frame_event.set()
            raise

        if window_name:
            # ignore
            monitor_index = None
        else:
            # 1 origin
            monitor_index = monitor_index + 1

        if window_handle is not None:
            import win32gui

            matches = []
            win32gui.EnumWindows(
                lambda hwnd, _: matches.append(hwnd) if win32gui.GetWindowText(hwnd) == window_name else None, None
            )
            if matches != [window_handle] or win32gui.FindWindow(None, window_name) != window_handle:
                raise RuntimeError("Capture title no longer identifies the fixed window")
        capture = WindowsCapture(
            cursor_capture=None,
            draw_border=None,
            monitor_index=monitor_index,
            window_name=window_name,
        )

    @capture.event
    def on_frame_arrived(frame, capture_control):
        nonlocal frame_shm, frame_event, frame_lock, stop_event, frame_buffer, frame_pos, window_name, frame_count, crop_top, crop_left, crop_right, crop_bottom  # noqa
        if not frame_event.is_set():
            with frame_lock:
                source_frame = frame.frame_buffer
                buffer_height, buffer_width = frame_buffer.shape[:2]
                content_rect = (0, 0, buffer_width, buffer_height)
                geometry = None
                if fit_window:
                    if monitor_target is not None:
                        current = _monitor_geometry(monitor_target["monitor_index"])
                        before = getattr(capture_control, "monitor_geometry", current)
                        if current == before == monitor_target:
                            rect = current["monitor_rect"]
                            geometry = (None, rect, rect)
                    else:
                        geometry = (
                            getattr(capture_control, "frame_geometry", None)
                            if backend == "mss"
                            else _window_geometry(window_handle, backend)
                        )
                if fit_window:
                    frame_mapping[0] = 0
                # Apply cropping for window capture
                crop_applied = (window_name or window_handle is not None) and (
                    crop_top > 0 or crop_left > 0 or crop_right > 0 or crop_bottom > 0
                )
                if crop_applied:
                    h, w = source_frame.shape[:2]
                    top = crop_top
                    bottom = h - crop_bottom if crop_bottom > 0 else h
                    left = crop_left
                    right = w - crop_right if crop_right > 0 else w
                    source_frame = source_frame[top:bottom, left:right, :]

                if fit_window:
                    frame_buffer[:] = 0
                    if geometry is not None and geometry[0] == pinned_pid:
                        _, window_rect, capture_rect = geometry
                        width, height = capture_rect[2] - capture_rect[0], capture_rect[3] - capture_rect[1]
                        source_height, source_width = source_frame.shape[:2]
                        if frame.frame_buffer.shape[:2] == (height, width) and source_width > 0 and source_height > 0:
                            buffer_height, buffer_width = frame_buffer.shape[:2]
                            scale = min(buffer_width / source_width, buffer_height / source_height)
                            fitted_width = max(1, min(buffer_width, round(source_width * scale)))
                            fitted_height = max(1, min(buffer_height, round(source_height * scale)))
                            left = (buffer_width - fitted_width) // 2
                            top = (buffer_height - fitted_height) // 2
                            # RGBX 按四字节原样缩放，避免 Pillow 对 BGRA 的 alpha 预乘。
                            resized = np.asarray(
                                Image.fromarray(source_frame, mode="RGBX").resize(
                                    (fitted_width, fitted_height), Image.Resampling.BOX
                                )
                            )
                            frame_buffer[top : top + fitted_height, left : left + fitted_width] = resized
                            content_rect = (left, top, left + fitted_width, top + fitted_height)
                            # 像素与映射同锁发布；尺寸变化中间帧不能沿用旧点击坐标。
                            unchanged = (
                                _monitor_geometry(monitor_target["monitor_index"]) == monitor_target
                                if monitor_target is not None
                                else _window_geometry(window_handle, backend) == geometry
                            )
                            if unchanged:
                                frame_mapping[:] = [
                                    1,
                                    monitor_target["monitor_handle"] if monitor_target is not None else window_handle,
                                    monitor_target["monitor_index"] if monitor_target is not None else pinned_pid,
                                    (3 if backend == "mss" else 4)
                                    if monitor_target is not None
                                    else (1 if backend == "mss" else 2),
                                    *window_rect,
                                    *capture_rect,
                                    crop_top,
                                    crop_left,
                                    crop_right,
                                    crop_bottom,
                                    left,
                                    top,
                                    left + fitted_width,
                                    top + fitted_height,
                                    buffer_width,
                                    buffer_height,
                                ]
                elif frame_buffer.shape != source_frame.shape:
                    if window_name is not None:
                        # NOTE: The size may differ due to resizing, Window effects, etc.
                        # I wanted to use replication padding,
                        # but since the edges of the Windows screen are black, it's meaningless
                        min_h = min(frame_buffer.shape[0], source_frame.shape[0])
                        min_w = min(frame_buffer.shape[1], source_frame.shape[1])
                        if frame_count % 30 == 0:
                            frame_buffer[:] = 0.0
                        frame_buffer[0:min_h, 0:min_w, :] = source_frame[0:min_h, 0:min_w, :]
                        content_rect = (0, 0, min_w, min_h)
                        frame_count += 1
                        if frame_count > 0xFFFF:
                            frame_count = 0
                    else:
                        raise RuntimeError(
                            f"Screen size missmatch. frame_buffer={frame_buffer.shape}, frame={source_frame.shape}"
                        )
                else:
                    frame_buffer[:] = source_frame

                if frame_mapping is not None:
                    frame_mapping[16:22] = [*content_rect, buffer_width, buffer_height]

                if hasattr(capture_control, "frame_pos"):
                    frame_pos[0] = capture_control.frame_pos[0] + (crop_top if crop_applied else 0)
                    frame_pos[1] = capture_control.frame_pos[1] + (crop_left if crop_applied else 0)

                frame_event.set()

        if stop_event.is_set():
            capture_control.stop()

    @capture.event
    def on_closed():
        if frame_mapping is not None:
            with frame_lock:
                frame_mapping[0] = 0

    try:
        # event loop
        capture.start()
    finally:
        if frame_mapping is not None:
            with frame_lock:
                frame_mapping[0] = 0
        frame_event.set()
        time.sleep(0.1)


def to_tensor(bgra, device):
    x = torch.from_numpy(bgra)
    x = x[:, :, :3].permute(2, 0, 1).contiguous().to(device)
    x = x / 255.0
    return x


class ScreenshotProcess(threading.Thread):
    def __init__(
        self,
        fps,
        frame_width,
        frame_height,
        monitor_index,
        window_name,
        device,
        backend="mss",
        crop_top=0,
        crop_left=0,
        crop_right=0,
        crop_bottom=0,
        draw_cursor_enabled=True,
        window_handle: int | None = None,
        fit_window: bool = False,
        monitor_target: dict[str, Any] | None = None,
    ):
        super().__init__(daemon=True)
        self.backend = backend
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.monitor_index = monitor_index
        self.window_name = window_name
        self.window_handle = window_handle
        self.fit_window = fit_window
        self.monitor_target = dict(monitor_target) if monitor_target is not None else None
        self.device = device
        self.crop_top = crop_top
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.draw_cursor_enabled = draw_cursor_enabled
        self.frame = None
        self.frame_mapping: dict[str, Any] | None = None
        self._content_rect: tuple[int, int, int, int] | None = None
        self._pending_content_rect: tuple[int, int, int, int] | None = None
        self.process: Any = None
        self.process_frame_buffer: Any = None
        self.process_stop_event: Any = None
        self.frame_set_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.fps_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.fps_counter: deque[float] = deque(maxlen=120)

        self.cuda_stream: torch.cuda.Stream | None = None
        if device.type == "cuda":
            self.cuda_stream = torch.cuda.Stream(device=device)

    def start_process(self):
        if self.monitor_target is not None:
            if _monitor_geometry(self.monitor_index) != self.monitor_target:
                raise RuntimeError("显示器布局已变化，请重启串流")
            left, top, right, bottom = self.monitor_target["monitor_rect"]
            screen_size = right - left, bottom - top
        elif self.window_handle is not None:
            geometry = _window_geometry(self.window_handle, self.backend)
            if geometry is None:
                raise RuntimeError("Capture window is no longer available")
            _, (left, top, right, bottom), _ = geometry
            screen_size = (right - left, bottom - top)
        elif self.window_name:
            rect = get_window_rect_by_title(self.window_name)
            if rect is None:
                raise RuntimeError(f"{self.window_name} not found")
            screen_size = (rect["width"], rect["height"])
        else:
            screen_size = get_screen_size(self.monitor_index)
        self.screen_width = screen_size[0]
        self.screen_height = screen_size[1]
        template = np.zeros((self.screen_height, self.screen_width, 4), dtype=np.uint8)
        self.process_frame_buffer = shared_memory.SharedMemory(create=True, size=template.nbytes)
        self.process_frame_pos = mp.Array("i", [0, 0], lock=False)
        self.process_frame_mapping = mp.Array("q", [0] * 22, lock=False)
        self.process_stop_event = mp.Event()
        self.process_frame_event = mp.Event()
        self.process_frame_lock = mp.Lock()
        self.process = mp.Process(
            target=capture_process,
            args=(
                tuple(template.shape),
                self.monitor_index,
                self.window_name,
                self.process_frame_buffer,
                self.process_frame_pos,
                self.process_frame_lock,
                self.process_frame_event,
                self.process_stop_event,
                self.backend,
                self.crop_top,
                self.crop_left,
                self.crop_right,
                self.crop_bottom,
                self.window_handle,
                self.fit_window,
                self.process_frame_mapping,
                self.monitor_target,
            ),
        )
        self.process.start()

    def run(self):
        with self.frame_lock:
            self.frame_mapping = None
        frame_buffer = None
        try:
            self.start_process()
            while True:
                tick = time.perf_counter()
                self.process_frame_event.clear()
                while not self.process_frame_event.wait(1):
                    if not self.process.is_alive():
                        raise RuntimeError("thread is already dead")

                with self.process_frame_lock:
                    frame = np.ndarray(
                        (self.screen_height, self.screen_width, 4), dtype=np.uint8, buffer=self.process_frame_buffer.buf
                    )
                    frame = frame.copy()
                    mapping = _decode_frame_mapping(self.process_frame_mapping, self.monitor_target)
                    left, top, right, bottom, buffer_width, buffer_height = self.process_frame_mapping[16:22]

                frame = torch.from_numpy(frame)
                if frame_buffer is None:
                    frame_buffer = frame.clone()
                    if torch.cuda.is_available():
                        frame_buffer = frame_buffer.pin_memory()
                else:
                    frame_buffer.copy_(frame)

                if self.cuda_stream is not None:
                    with torch.cuda.stream(self.cuda_stream):
                        frame = frame_buffer.to(self.device, non_blocking=True)
                        frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                        if (
                            self.backend == "mss"
                            and sys.platform != "linux"
                            and self.draw_cursor_enabled
                            and (not self.fit_window or mapping is not None)
                        ):
                            draw_cursor(
                                frame,
                                wx.GetMousePosition(),
                                offset=self.process_frame_pos,
                                mapping=mapping if self.fit_window else None,
                            )
                        if frame.shape[1:] != (self.frame_height, self.frame_width):
                            frame = TF.resize(
                                frame,
                                size=(self.frame_height, self.frame_width),
                                interpolation=InterpolationMode.BILINEAR,
                                antialias=True,
                            )
                    frame.record_stream(self.cuda_stream)
                else:
                    frame = frame_buffer.to(self.device)
                    frame = frame[:, :, 0:3][:, :, (2, 1, 0)].permute(2, 0, 1).contiguous() / 255.0
                    if (
                        self.backend == "mss"
                        and sys.platform != "linux"
                        and self.draw_cursor_enabled
                        and (not self.fit_window or mapping is not None)
                    ):
                        draw_cursor(
                            frame,
                            wx.GetMousePosition(),
                            offset=self.process_frame_pos,
                            mapping=mapping if self.fit_window else None,
                        )
                    if frame.shape[1:] != (self.frame_height, self.frame_width):
                        frame = TF.resize(
                            frame,
                            size=(self.frame_height, self.frame_width),
                            interpolation=InterpolationMode.BILINEAR,
                            antialias=True,
                        )

                content_rect = None
                if 0 <= left < right <= buffer_width and 0 <= top < bottom <= buffer_height:
                    # 原始 BGRA[1440,2560,4] → RGB[3,1080,1920]；ROI 必须同步缩放到最终帧。
                    content_rect = (
                        round(left * frame.shape[2] / buffer_width),
                        round(top * frame.shape[1] / buffer_height),
                        round(right * frame.shape[2] / buffer_width),
                        round(bottom * frame.shape[1] / buffer_height),
                    )
                    if content_rect[0] >= content_rect[2] or content_rect[1] >= content_rect[3]:
                        content_rect = None
                with self.frame_lock:
                    self.frame = frame
                    self.frame_mapping = mapping
                    self._pending_content_rect = content_rect
                    self.frame_set_event.set()

                process_time = time.perf_counter() - tick
                with self.fps_lock:
                    self.fps_counter.append(process_time)

                if self.stop_event.is_set():
                    break
        finally:
            with self.frame_lock:
                self.frame_mapping = None
                self._content_rect = self._pending_content_rect = None
            self.stop_event.set()
            self.frame_set_event.set()
            try:
                if self.process_stop_event is not None:
                    self.process_stop_event.set()
                if self.process is not None:
                    if self.process.pid is not None:
                        self.process.join(timeout=4)
                        if self.process.is_alive():
                            self.process.terminate()
                            self.process.join()
                    self.process = None
            finally:
                if self.process_frame_buffer is not None:
                    try:
                        self.process_frame_buffer.close()
                    finally:
                        self.process_frame_buffer.unlink()
                        self.process_frame_buffer = None

    def get_frame(self):
        while not self.frame_set_event.wait(1):
            if self.stop_event.is_set():
                raise RuntimeError("thread is already dead")
        if self.stop_event.is_set():
            raise RuntimeError("thread is already dead")
        with self.frame_lock:
            if self.cuda_stream is not None:
                torch.cuda.current_stream().wait_stream(self.cuda_stream)
            frame = self.frame
            self._content_rect = self._pending_content_rect
            self._pending_content_rect = None
            self.frame_set_event.clear()

        return frame

    def get_fps(self):
        with self.fps_lock:
            if self.fps_counter:
                mean_processing_time = sum(self.fps_counter) / len(self.fps_counter)
                return 1 / mean_processing_time
            else:
                return 0

    def get_frame_mapping(self) -> dict[str, Any] | None:
        with self.frame_lock:
            if self.monitor_target is not None and _monitor_geometry(self.monitor_index) != self.monitor_target:
                return None
            if self.stop_event.is_set() or (self.process is not None and not self.process.is_alive()):
                return None
            if self.process is not None:
                with self.process_frame_lock:
                    if _decode_frame_mapping(self.process_frame_mapping, self.monitor_target) != self.frame_mapping:
                        return None
            return dict(self.frame_mapping) if self.frame_mapping is not None else None

    def get_content_rect(self) -> tuple[int, int, int, int] | None:
        with self.frame_lock:
            return None if self.stop_event.is_set() else self._content_rect

    def stop(self):
        self.stop_event.set()
        with self.frame_lock:
            self._content_rect = self._pending_content_rect = None
        if self.ident is not None:
            self.join(timeout=4)
