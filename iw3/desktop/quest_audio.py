from collections import deque
from typing import Any

import numpy as np


class QuestAudio:
    def __init__(self, gain: float = 1.0, source: str = "speaker") -> None:
        if source not in {"speaker", "process"}:
            raise ValueError("Unknown Quest audio source")
        self._source = source
        self._process: Any = None
        self.set_gain(gain)
        self._chunks: deque[bytes] = deque(maxlen=10)
        self._manager: Any = None
        self._stream: Any = None
        self._sample_rate = 48000
        self._error: Exception | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def set_gain(self, gain: float) -> None:
        if type(gain) not in (int, float) or not 0 <= gain <= 4:
            raise ValueError("Quest音量增益必须是0到4之间的有限数字")
        self._gain = float(gain)

    def start(self) -> None:
        if self._process is not None:
            return
        if self._source == "process":
            from .quest_process_audio import ProcessAudio

            process = ProcessAudio()
            process.start()
            self._process = process
            self._sample_rate = process.sample_rate
            return
        if self._stream is not None:
            return
        try:
            import pyaudiowpatch as pa  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError(
                "Quest电脑声音需要PyAudioWPatch；在nunif-windows运行 "
                ".\\nunif-cmd.bat python -m pip install -r iw3/desktop/requirements-quest.txt"
            ) from error

        self._error = None
        try:
            self._manager = pa.PyAudio()
            device = self._manager.get_default_wasapi_loopback()
            if not device["isLoopbackDevice"]:
                raise ValueError("默认扬声器没有回环设备")
            self._sample_rate = int(device["defaultSampleRate"])
            if not self._manager.is_format_supported(
                self._sample_rate, input_device=device["index"], input_channels=2, input_format=pa.paInt16
            ):
                raise ValueError("默认扬声器不支持双声道s16音频")

            def callback(
                in_data: bytes, frame_count: int, time_info: dict[str, float], status_flags: int
            ) -> tuple[None, int]:
                try:
                    # 实测48000Hz：bytes(1920) → 480帧×2声道×int16；回调只入队，不执行网络操作。
                    if status_flags & ~pa.paInputOverflow:
                        raise RuntimeError(f"扬声器回环音频状态异常：{status_flags}")
                    if len(in_data) != frame_count * 4:
                        raise ValueError("扬声器回环音频块长度错误")
                    self._chunks.append(in_data)
                    return None, pa.paContinue
                except Exception as error:
                    self._error = error
                    return None, pa.paAbort

            self._stream = self._manager.open(
                format=pa.paInt16,
                channels=2,
                rate=self._sample_rate,
                frames_per_buffer=self._sample_rate // 100,
                input=True,
                input_device_index=device["index"],
                stream_callback=callback,
                start=False,
            )
            self._stream.start_stream()
        except Exception as error:
            try:
                self.stop()
            finally:
                raise RuntimeError("无法采集默认扬声器声音；请检查默认输出设备是否支持双声道音频") from error

    def get_chunks(self) -> list[bytes]:
        gain = self._gain
        chunks = self._process.get_chunks() if self._process is not None else []
        if self._process is None:
            while True:
                try:
                    chunks.append(self._chunks.popleft())
                except IndexError:
                    break
        if gain == 1:
            return chunks
        # PCM bytes(4N) → float32(2N) → s16le bytes(4N)；先扩类型，避免 int16 乘法回绕。
        return [
            np.clip(np.frombuffer(chunk, dtype="<i2").astype(np.float32) * gain, -32768, 32767).astype("<i2").tobytes()
            for chunk in chunks
        ]

    def check_error(self) -> None:
        if self._process is not None:
            self._process.check_error()
        if self._error is not None:
            raise RuntimeError("Quest电脑音频采集失败") from self._error
        if self._stream is not None and not self._stream.is_active():
            raise RuntimeError("Quest电脑音频采集意外停止，请检查默认扬声器设备")

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        manager, self._manager = self._manager, None
        try:
            if self._process is not None:
                self._process.stop()
                self._process = None
        finally:
            try:
                if stream is not None:
                    try:
                        stream.stop_stream()
                    finally:
                        stream.close()
            finally:
                try:
                    if manager is not None:
                        manager.terminate()
                finally:
                    self._chunks.clear()
