"""桌面与网页 NVDEC 共用一个消费者；这里只切来源，不做 AI 或编码。"""

import threading
from collections.abc import Callable
from typing import Any


class SourceRouter:
    def __init__(
        self,
        bridge: Any,
        desktop_factory: Callable[[], Any],
        media_factory: Callable[[], Any],
        on_switch: Callable[[str, int], None],
        *,
        fixed_browser: bool = False,
    ) -> None:
        self.bridge = bridge
        self._factories = {"desktop": desktop_factory, "browser": media_factory}
        self._on_switch = on_switch
        self._fixed_browser = fixed_browser
        self.source_kind = "browser" if fixed_browser else "desktop"
        # 工厂可能已初始化 OBS portal；即使尚未 start，stop 也负责释放。
        self._source = self._factories[self.source_kind]()
        self._epoch = 0
        self._frame_token: tuple[int, str | None] | None = None
        self._idle_event = threading.Event()
        self._started = False
        self._stopped = False
        self._fallback_key: tuple[Any] | None = None
        self._fallback_reason = ""

    @property
    def frame_set_event(self) -> threading.Event:
        return self._source.frame_set_event if self._source is not None else self._idle_event

    def _desired_kind(self) -> str:
        state = self.bridge.status()
        if state.get("fresh") is True and type(state.get("fullscreen")) is bool:
            if (state["fullscreen"] or self._fixed_browser) and self._fallback_key == (state.get("generation"),):
                return "desktop"
            if self._fixed_browser:
                return "browser"
            return "browser" if state["fullscreen"] else "desktop"
        # 超时不是退出全屏；网页源失效必须明确等待，不偷偷切回录屏。
        return self.source_kind

    def start(self) -> None:
        if self._started or self._stopped:
            raise RuntimeError("Source router cannot be started twice")
        self._started = True
        self._epoch += 1
        self._on_switch(self.source_kind, self._epoch)
        self._source.start()
        self.poll_route()

    def poll_route(self) -> None:
        if not self._started or self._stopped:
            return
        state = self.bridge.status()
        if state.get("fresh") is True:
            if (not self._fixed_browser and not state.get("fullscreen")) or self._fallback_key != (
                state.get("generation"),
            ):
                self._fallback_key, self._fallback_reason = None, ""
            if self.source_kind == "browser" and self._source is not None:
                status = self._source.get_status()
                if status.get("fallback_required") is True and status.get("failure_generation") == state.get(
                    "generation"
                ):
                    self._fallback_key = (state.get("generation"),)
                    self._fallback_reason = status.get("fallback_reason") or status.get("message", "原视频无法读取")
        desired = self._desired_kind()
        if desired == self.source_kind:
            return
        old = self._source
        self.source_kind = desired
        self._epoch += 1
        self._frame_token = None
        try:
            # 先释放旧输入、阻挡旧编码/环境光，才等待旧捕获器停止。
            self._on_switch(desired, self._epoch)
        finally:
            if old is not None:
                old.stop()
        self._source = None
        self._source = self._factories[desired]()
        self._source.start()

    def is_alive(self) -> bool:
        return bool(self._started and not self._stopped and self._source is not None and self._source.is_alive())

    def get_frame(self) -> Any:
        self.poll_route()
        source, epoch = self._source, self._epoch
        if source is None or not source.frame_set_event.is_set():
            return None
        if self.source_kind == "browser" and self.bridge.status().get("fresh") is not True:
            source.frame_set_event.clear()
            return None
        frame = source.get_frame()
        if frame is None:
            return None
        token = (epoch, source.get_frame_generation() if self.source_kind == "browser" else None)
        if not self.is_current_generation(token):
            return None
        # 实际两个入口都交付 float32 RGB[3,H,W]、[0,1]；保留原 CUDA event/stream 同步。
        self._frame_token = token
        return frame

    def get_frame_generation(self) -> tuple[int, str | None] | None:
        return self._frame_token

    def is_current_generation(self, token: tuple[int, str | None] | None) -> bool:
        source, kind = self._source, self.source_kind
        if token is None or self._stopped or token[0] != self._epoch or source is None or self._desired_kind() != kind:
            return False
        # A texture can finish while the producer is replacing the source.
        if kind == "desktop":
            valid = self._started and source.is_alive()
        else:
            validate = getattr(source, "is_current_generation", None)
            valid = validate is not None and self.bridge.status().get("fresh") is True and validate(token[1])
        return bool(
            valid
            and not self._stopped
            and token[0] == self._epoch
            and source is self._source
            and kind == self.source_kind
        )

    def get_content_rect(self) -> tuple[int, int, int, int] | None:
        if self._source is not None and hasattr(self._source, "get_content_rect"):
            return self._source.get_content_rect()
        return None

    def get_frame_mapping(self) -> dict[str, Any] | None:
        if self.source_kind == "desktop" and self._source is not None and hasattr(self._source, "get_frame_mapping"):
            return self._source.get_frame_mapping()
        return None

    def get_fps(self) -> float:
        if self._source is None or (self.source_kind == "browser" and self.get_status()["status"] != "running"):
            return 0.0
        return self._source.get_fps()

    def get_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "source_kind": self.source_kind,
            "route_epoch": self._epoch,
            "input_source": "桌面屏幕捕获" if self.source_kind == "desktop" else "网页压缩视频（不录屏）",
            "decoder": "屏幕捕获" if self.source_kind == "desktop" else "NVDEC",
            "hardware_active": False,
            "source_size": None,
            "audio_path": "Windows 回环采音；网页视频跟随播放时钟",
            "status": "running",
            "message": "桌面屏幕捕获；抖音全屏时自动切换原视频。",
        }
        if self.source_kind == "browser":
            if self._source is not None:
                status.update(self._source.get_status())
            if self.bridge.status().get("fresh") is not True:
                status.update(
                    status="waiting",
                    message="浏览器状态超时；等待网页恢复，不回退屏幕捕获。",
                    hardware_active=False,
                    source_size=None,
                )
        elif self._fallback_key is not None:
            status.update(
                message=f"取源失败，正在录屏：{self._fallback_reason}",
                fallback_reason=self._fallback_reason,
                input_source="屏幕捕获（取源失败回退）",
            )
        return status

    def stop(self) -> None:
        self._stopped = True
        self._frame_token = None
        source, self._source = self._source, None
        if source is not None:
            source.stop()
        self._idle_event.set()
