import ctypes
import os
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any


def _load_library() -> Any:
    if sys.platform != "win32" or sys.getwindowsversion().build < 20348:
        raise RuntimeError("独立采音需要Windows build 20348或更新版本；可选择扬声器回环兼容模式")
    filename = Path(__file__).with_name("native") / "quest_process_audio.dll"
    if not filename.is_file():
        raise RuntimeError("独立采音组件缺失；请先运行iw3/desktop/native/build.ps1，或选择扬声器回环兼容模式")
    library = ctypes.CDLL(str(filename))
    signatures = {
        "pa_create": ([ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int32),
        "pa_start": ([ctypes.c_void_p], ctypes.c_int32),
        "pa_read": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)],
            ctypes.c_int32,
        ),
        "pa_stop": ([ctypes.c_void_p], ctypes.c_int32),
        "pa_free": ([ctypes.c_void_p], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
    return library


def _check(result: int) -> None:
    if result < 0:
        raise RuntimeError(f"Windows独立采音失败 HRESULT=0x{result & 0xFFFFFFFF:08X}")


class ProcessAudio:
    sample_rate = 48000

    def __init__(self) -> None:
        self._library: Any = None
        self._handle = ctypes.c_void_p()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._chunks: deque[bytes] = deque(maxlen=10)
        self._error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._library = _load_library()
        self._error = None
        self._stop.clear()
        try:
            # 排除本进程，采集其余系统进程；不绑定扬声器端点，不修改Windows音量。
            _check(self._library.pa_create(os.getpid(), 0, ctypes.byref(self._handle)))
            _check(self._library.pa_start(self._handle))
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
        except Exception:
            self.stop()
            raise

    def _read(self) -> None:
        buffer = ctypes.create_string_buffer(1920)
        try:
            while not self._stop.is_set():
                written = ctypes.c_uint32()
                _check(self._library.pa_read(self._handle, buffer, len(buffer), 100, ctypes.byref(written)))
                if written.value > len(buffer) or written.value % 4:
                    raise ValueError("Windows独立采音PCM长度错误")
                if written.value:
                    # 固定48kHz双声道s16le，每10ms最多1920B；队列最多100ms。
                    self._chunks.append(buffer.raw[: written.value])
        except Exception as error:
            self._error = error

    def get_chunks(self) -> list[bytes]:
        chunks = []
        while True:
            try:
                chunks.append(self._chunks.popleft())
            except IndexError:
                return chunks

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Quest独立音频采集失败") from self._error

    def stop(self) -> None:
        self._stop.set()
        result = self._library.pa_stop(self._handle) if self._handle.value else 0
        if self._thread is not None:
            if self._thread.ident is not None:
                self._thread.join(2)
            if self._thread.is_alive():
                raise RuntimeError("Quest独立音频线程未能及时停止")
            self._thread = None
        if self._handle.value:
            self._library.pa_free(self._handle)
            self._handle = ctypes.c_void_p()
        self._chunks.clear()
        _check(result)
