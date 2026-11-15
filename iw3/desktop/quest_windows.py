import ctypes
import math
import sys
from collections.abc import Callable
from typing import Any


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", ctypes.c_uint32), ("data", _INPUTUNION)]


class _RECT(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in ("left", "top", "right", "bottom")]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32)]


class WindowsInput:
    def __init__(
        self,
        window_name: str | None,
        screenshot: str,
        crops: tuple[int, int, int, int],
        frame_mapping: Callable[[], dict[str, Any] | None] | None = None,
        *,
        monitor_index: int = 0,
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Quest PC input requires Windows")
        if screenshot not in {"wc_mp", "mss", "wc_cuda"}:
            raise ValueError("Quest PC input requires wc_mp, mss or wc_cuda capture")
        if len(crops) != 4 or any(type(value) is not int or value < 0 for value in crops):
            raise ValueError("Crop values must be four nonnegative pixel counts")
        window_name = window_name if window_name and window_name.strip() else None
        if (window_name is None or screenshot == "wc_cuda") and frame_mapping is None:
            raise ValueError("Monitor and CUDA control require captured-frame mapping")
        if window_name is None and any(crops):
            raise ValueError("Monitor control requires zero crop values")

        import win32gui
        import win32process

        self._gui = win32gui
        self._process = win32process
        self._screenshot = screenshot
        self._top, self._left, self._right, self._bottom = crops
        self._point: tuple[int, int] | None = None
        self._frame_mapping = frame_mapping
        self._point_mapping: dict[str, Any] | None = None
        self._left_down = False
        self._pending_key: int | None = None
        self._hwnd: int | None = None
        self._pid: int | None = None
        self._monitor_target: dict[str, Any] | None = None
        if window_name is None:
            from .screenshot_process import _monitor_geometry

            self._get_monitor_geometry = _monitor_geometry
            monitor = _monitor_geometry(monitor_index)
            if monitor is None:
                raise ValueError("Capture monitor was not found")
            self._monitor_target = dict(monitor)
        else:
            self._hwnd = win32gui.FindWindow(None, window_name)
            if not self._hwnd or not win32gui.IsWindow(self._hwnd):
                raise ValueError("Capture window was not found")
            matches: list[int] = []

            def collect(hwnd: int, _: object) -> None:
                if win32gui.GetWindowText(hwnd) == window_name:
                    matches.append(hwnd)

            # 捕获也按标题查找；重名会使两次查找绑定到不同窗口。
            win32gui.EnumWindows(collect, None)
            if matches != [self._hwnd]:
                raise ValueError("Capture window title must identify exactly one window")
            self._pid = win32process.GetWindowThreadProcessId(self._hwnd)[1]

        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int)
        self._user32.SendInput.restype = ctypes.c_uint
        self._user32.SetPhysicalCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
        self._user32.SetPhysicalCursorPos.restype = ctypes.c_int
        self._user32.GetPhysicalCursorPos.argtypes = (ctypes.POINTER(_POINT),)
        self._user32.GetPhysicalCursorPos.restype = ctypes.c_int
        if self._hwnd is not None and screenshot in {"wc_mp", "wc_cuda"}:
            self._dwm = ctypes.WinDLL("dwmapi", use_last_error=True)
            self._dwm.DwmGetWindowAttribute.argtypes = (
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_uint32,
            )
            self._dwm.DwmGetWindowAttribute.restype = ctypes.c_int32

        self._window_rect, self._capture_rect = self._geometry()
        wl, wt, wr, wb = self._window_rect
        cl, ct, cr, cb = self._capture_rect
        # 实测 WGC 原点是 DWM 左上角，wc_mp 共享缓冲区仍按完整 GetWindowRect 分配。
        self._buffer_width, self._buffer_height = wr - wl, wb - wt
        self._content_width = cr - cl - self._left - self._right
        self._content_height = cb - ct - self._top - self._bottom
        if min(self._buffer_width, self._buffer_height, self._content_width, self._content_height) <= 0:
            raise ValueError("Capture crop leaves no usable pixels")

    @property
    def hwnd(self) -> int | None:
        return self._hwnd

    @property
    def monitor_target(self) -> dict[str, Any] | None:
        return dict(self._monitor_target) if self._monitor_target is not None else None

    def _read_mapping(self) -> dict[str, Any] | None:
        mapping = self._frame_mapping() if self._frame_mapping is not None else None
        if not isinstance(mapping, dict):
            return None
        backend = {"wc_mp": "windows_capture", "mss": "mss", "wc_cuda": "wc_cuda"}[self._screenshot]
        if mapping.get("backend") != backend:
            return None
        result: dict[str, Any] = {"backend": backend}
        if self._monitor_target is not None:
            if mapping.get("target") != "monitor" or any(
                mapping.get(key) != value for key, value in self._monitor_target.items()
            ):
                return None
            result.update(target="monitor", **self._monitor_target)
        else:
            if mapping.get("hwnd") != self._hwnd or mapping.get("pid") != self._pid:
                return None
            result.update(hwnd=self._hwnd, pid=self._pid)
        fields = [
            ("capture_rect", 4),
            ("crops", 4),
            ("content_rect", 4),
            ("buffer_size", 2),
        ]
        if self._monitor_target is None:
            fields.append(("window_rect", 4))
        for field, size in fields:
            value = mapping.get(field)
            if not isinstance(value, (list, tuple)) or len(value) != size or any(type(v) is not int for v in value):
                return None
            result[field] = tuple(value)
        if result["crops"] != (self._top, self._left, self._right, self._bottom):
            return None
        if self._monitor_target is not None:
            if result["capture_rect"] != self._monitor_target["monitor_rect"]:
                return None
        else:
            wl, wt, wr, wb = result["window_rect"]
            if wr <= wl or wb <= wt:
                return None
        cl, ct, cr, cb = result["capture_rect"]
        left, top, right, bottom = result["content_rect"]
        width, height = result["buffer_size"]
        if (
            cr - cl <= self._left + self._right
            or cb - ct <= self._top + self._bottom
            or not 0 <= left < right <= width
            or not 0 <= top < bottom <= height
        ):
            return None
        return result

    def _geometry(self) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        if self._monitor_target is not None:
            rect = self._monitor_target["monitor_rect"]
            return rect, rect
        window_rect = self._gui.GetWindowRect(self._hwnd)
        if self._screenshot == "mss":
            return window_rect, window_rect
        rect = _RECT()
        if self._dwm.DwmGetWindowAttribute(self._hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
            raise OSError("Could not read capture window frame bounds")
        return window_rect, (rect.left, rect.top, rect.right, rect.bottom)

    def _target_valid(self, point: tuple[int, int] | None) -> bool:
        if point is None:
            return False
        try:
            if self._monitor_target is not None:
                left, top, right, bottom = self._monitor_target["monitor_rect"]
                return bool(
                    self._get_monitor_geometry(self._monitor_target["monitor_index"]) == self._monitor_target
                    and self._point_mapping is not None
                    and self._read_mapping() == self._point_mapping
                    and left <= point[0] < right
                    and top <= point[1] < bottom
                )
            geometry = (self._window_rect, self._capture_rect)
            if self._frame_mapping is not None:
                if self._point_mapping is None or self._read_mapping() != self._point_mapping:
                    return False
                geometry = self._point_mapping["window_rect"], self._point_mapping["capture_rect"]
            return bool(
                self._gui.IsWindow(self._hwnd)
                and self._process.GetWindowThreadProcessId(self._hwnd)[1] == self._pid
                and self._gui.IsWindowVisible(self._hwnd)
                and not self._gui.IsIconic(self._hwnd)
                and self._geometry() == geometry
                and self._gui.GetForegroundWindow() == self._hwnd
                and self._gui.GetAncestor(self._gui.WindowFromPoint(point), 2) == self._hwnd
            )
        except (self._gui.error, OSError):
            return False

    def _last_target_valid(self) -> bool:
        cursor = _POINT()
        if (
            self._target_valid(self._point)
            and self._user32.GetPhysicalCursorPos(ctypes.byref(cursor))
            and (cursor.x, cursor.y) == self._point
        ):
            return True
        self._point = None
        self._point_mapping = None
        return False

    def move(self, u: float, v: float) -> bool:
        self._point = None
        self._point_mapping = None
        if not (
            type(u) in (int, float)
            and type(v) in (int, float)
            and 0 <= u <= 1
            and 0 <= v <= 1
            and math.isfinite(u)
            and math.isfinite(v)
        ):
            return False
        if self._frame_mapping is not None:
            mapping = self._read_mapping()
            if mapping is None:
                return False
            width, height = mapping["buffer_size"]
            left, top, right, bottom = mapping["content_rect"]
            bx, by = u * width, v * height
            if not ((left <= bx < right or bx == right == width) and (top <= by < bottom or by == bottom == height)):
                return False
            cl, ct, cr, cb = mapping["capture_rect"]
            source_width, source_height = cr - cl - self._left - self._right, cb - ct - self._top - self._bottom
            # 逆变换同一帧的 letterbox，保留连续 UV，避免缩小时丢失源图最右/下像素。
            px = min(math.floor((bx - left) * source_width / (right - left)), source_width - 1)
            py = min(math.floor((by - top) * source_height / (bottom - top)), source_height - 1)
            point = cl + self._left + px, ct + self._top + py
            self._point_mapping = mapping
        else:
            px = min(math.floor(u * self._buffer_width), self._buffer_width - 1)
            py = min(math.floor(v * self._buffer_height), self._buffer_height - 1)
            if px >= self._content_width or py >= self._content_height:
                return False
            point = (self._capture_rect[0] + self._left + px, self._capture_rect[1] + self._top + py)
        if not self._target_valid(point) or not self._user32.SetPhysicalCursorPos(*point):
            return False
        self._point = point
        return True

    def _send_mouse(self, flags: int) -> None:
        events = (INPUT * 1)()
        events[0].mi.dwFlags = flags
        if self._user32.SendInput(1, events, ctypes.sizeof(INPUT)) != 1:
            raise OSError("Windows rejected the Quest mouse event")

    def mouse_down(self) -> bool:
        if not self._last_target_valid():
            return False
        if not self._left_down:
            self._send_mouse(2)
            self._left_down = True
        return True

    def mouse_up(self) -> None:
        if self._left_down:
            # 失焦后仍释放本会话按下的左键，发送失败时保留状态以便再次清理。
            self._send_mouse(4)
            self._left_down = False

    def tap_key(self, direction: str) -> bool:
        if direction not in {"up", "down"}:
            raise ValueError("Only up/down video navigation is allowed")
        self.release_keys()
        if not self._last_target_valid():
            return False
        if self._monitor_target is not None:
            try:
                pointed = self._gui.GetAncestor(self._gui.WindowFromPoint(self._point), 2)
                if not pointed or pointed != self._gui.GetForegroundWindow():
                    return False
            except (self._gui.error, OSError):
                return False
        events = (INPUT * 2)()
        for event in events:
            event.type = 1
            event.ki.wVk = 0x26 if direction == "up" else 0x28
        events[1].ki.dwFlags = 2
        sent = self._user32.SendInput(2, events, ctypes.sizeof(INPUT))
        if sent != 2:
            if sent == 1:
                self._pending_key = events[0].ki.wVk
                self.release_keys()
            raise OSError("Windows rejected the Quest key tap")
        return True

    def release_keys(self) -> None:
        if self._pending_key is not None:
            # 只释放本会话已注入的方向键；失败保留归属，供断线或停止时重试。
            release = (INPUT * 1)()
            release[0].type = 1
            release[0].ki.wVk = self._pending_key
            release[0].ki.dwFlags = 2
            if self._user32.SendInput(1, release, ctypes.sizeof(INPUT)) != 1:
                raise OSError("Windows rejected Quest key release after a partial tap")
            self._pending_key = None
