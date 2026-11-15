"""Quest 管理连接独立于媒体初始化；网络线程只提交有限设置请求。"""

import copy
import importlib.util
import sys
import threading
from argparse import Namespace
from typing import Any

from .quest_settings import COMMANDS, FIELD_MAP, validate_patch


class SessionInterrupted(RuntimeError):
    pass


class QuestSession:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.stop_event = args.state.get("stop_event") or threading.Event()
        self.run_stop = threading.Event()
        self._lock = threading.Condition()
        self._values = self.values(args)
        self._stats: dict[str, Any] = {}
        self._last_good = copy.deepcopy(self._values)
        self._requested_mapper = getattr(args, "mapper", None)
        self._last_good_mapper = self._requested_mapper
        self._choices: dict[str, list[Any]] = {}
        self._live: dict[str, Any] = {}
        self._request: tuple[str, dict[str, Any]] | None = None
        self._controls: set[str] = set()
        self._status = "loading"
        self._message = "按右 B 或左 Y 打开设置；媒体加载失败后仍可修改。"
        self.server: Any = None
        self.capture: Any = None
        self._fps_event = args.state.get("fps_event")
        self._args_lock = args.state.get("args_lock") or threading.Lock()
        self._committed = False
        self._recovery_message = ""
        self._operation: str | None = None
        self._stream_restart_count = 0

    @staticmethod
    def values(args: Namespace) -> dict[str, Any]:
        return {key: copy.deepcopy(getattr(args, key)) for key in FIELD_MAP if hasattr(args, key)}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(
                {
                    "values": {**self._values, **self._stats},
                    "choices": self._choices,
                    "status": self._status,
                    "message": self._message,
                }
            )

    def set_stats(self, **values: Any) -> None:
        with self._lock:
            self._stats.update(copy.deepcopy(values))

    def status(self, status: str, message: str) -> None:
        with self._lock:
            self._status, self._message = status, message
            self._lock.notify_all()

    def submit(self, command: str, patch: dict[str, Any]) -> None:
        if command not in COMMANDS:
            raise ValueError("Unknown Quest setting command")
        with self._lock:
            clean = validate_patch(patch, self._values, self._choices)
            if command == "set":
                if any(not FIELD_MAP[key].immediate for key in clean):
                    raise ValueError("This setting requires applying a new configuration")
                self._live.update(clean)
            elif command in {"refresh_windows", "recenter"}:
                self._controls.add(command)
            else:
                self._request = (command, clean)
                self.run_stop.set()
            self._lock.notify_all()

    def take_live_patch(self) -> dict[str, Any]:
        with self._lock:
            patch, self._live = self._live, {}
            return patch

    def refresh_choices(self) -> None:
        import torch

        from ..base_depth_model import BaseDepthModel
        from .utils import (
            enum_window_names,
            get_monitor_size_list,
            is_kwcapture_supported,
            is_mss_supported,
            is_obs_supported,
        )

        backends = ["pil"]
        if is_mss_supported():
            backends.append("mss")
        if sys.platform == "win32" and importlib.util.find_spec("windows_capture"):
            backends.append("wc_mp")
        if (
            sys.platform == "win32"
            and importlib.util.find_spec("wc_cuda")
            and torch.cuda.is_available()
            and not getattr(torch.version, "hip", None)
        ):
            backends.append("wc_cuda")
        if is_kwcapture_supported():
            backends.append("kwcapture")
        if is_obs_supported():
            backends.append("obs")
        choices: dict[str, list[Any]] = {
            "depth_model": BaseDepthModel.get_available_models(desktop=True, small_only=False),
            "depth_model_small": BaseDepthModel.get_available_models(desktop=True, small_only=True),
            "gpu": [[i] for i in range(torch.cuda.device_count())],
            "screenshot": backends,
            "monitor_index": list(range(len(get_monitor_size_list()))),
            "window_name": [None, *enum_window_names()],
        }
        with self._lock:
            self._choices = choices

    def publish(self, args: Namespace, patch: dict[str, Any] | None = None) -> None:
        values = self.values(args)
        with self._lock:
            if patch is not None and not self._committed:
                self._values.update({key: values[key] for key in patch})
            else:
                self._values = values
            applied = copy.deepcopy(self._values)
        if self._fps_event is not None and hasattr(self._fps_event, "settings_applied"):
            self._fps_event.settings_applied(applied)

    def check_interrupt(self) -> None:
        if self.stop_event.is_set() or self.run_stop.is_set():
            raise SessionInterrupted("Quest media reload requested")

    def read_frame(self, capture: Any) -> Any:
        # 实际捕获器均为单消费者 Event；事件置位后 get_frame 不再阻塞。
        while True:
            self.check_interrupt()
            if self.server is not None:
                self.server.check_error()
            self.apply_live()
            if capture.frame_set_event.wait(0.1):
                self.check_interrupt()
                return capture.get_frame()
            if hasattr(capture, "is_alive") and not capture.is_alive():
                raise RuntimeError("Capture thread stopped before supplying a frame")

    def apply_live(self) -> None:
        with self._lock:
            controls, self._controls = self._controls, set()
        for command in controls:
            if command == "refresh_windows":
                self.refresh_choices()
            elif self.server is not None:
                self.server.recenter()
        patch = self.take_live_patch()
        if not patch:
            return
        from . import utils

        args = self.args
        with self._args_lock:
            for key, value in patch.items():
                setattr(args, key, value)
            if "foreground_scale" in patch:
                self._requested_mapper = None
            depth = args.state.get("depth_model")
            if depth is not None and "foreground_scale" in patch:
                args.mapper = utils.IW3U.resolve_mapper_name(
                    mapper=None, foreground_scale=args.foreground_scale, metric_depth=depth.is_metric()
                )
            estimator = args.state.get("convergence_model")
            if estimator is not None and "convergence" in patch:
                estimator.convergence = args.convergence
                estimator.reset()
            if self.server is not None and "quest_audio_gain" in patch:
                self.server.set_audio_gain(args.quest_audio_gain)
            if self.server is not None and "quest_screen_curvature" in patch:
                self.server.set_screen_curvature(args.quest_screen_curvature)
            if self.server is not None and any(key.startswith("quest_ambient") for key in patch):
                self.server.configure_ambient(
                    args.quest_ambient, args.quest_ambient_brightness, args.quest_ambient_spread
                )
        self.publish(args, patch)

    def media_ready(self) -> None:
        first_frame = not self._committed
        if first_frame:
            self._committed = True
            self.publish(self.args)
        with self._lock:
            self._last_good = self.values(self.args)
            self._last_good_mapper = self._requested_mapper
        if first_frame:
            message = self._recovery_message or "设置已在电脑端应用；请在头显确认实际画面。"
            if self._operation == "restart_stream" and not self._recovery_message:
                self._stream_restart_count += 1
                message = f"串流已重启（第 {self._stream_restart_count} 次）；请在头显确认实际画面。"
            self._operation = None
            self.status("running", message)

    def configure(self, **settings: Any) -> Any:
        self.server.pause_media()
        self.server.configure_video(settings["width"], settings["height"], settings["fps"], settings["bitrate"])
        self.server.configure_audio(settings["audio"], settings["audio_source"], settings["audio_gain"])
        self.server.set_screen_curvature(settings.get("screen_curvature", 0.0))
        self.server.configure_ambient(
            settings.get("ambient_enabled", False),
            settings.get("ambient_brightness", 0.25),
            settings.get("ambient_spread", 2.0),
        )
        self.server.set_controller_factory(settings.get("controller_factory"))
        return self.server

    def run(self, run_once: Any, init_wxapp: bool) -> Namespace:
        from nunif.initializer import gc_collect

        from . import utils
        from .quest_menu import QuestMenu
        from .quest_server import QuestServer

        args = self.args
        address = args.bind_addr or utils.get_local_address()
        self.server = QuestServer(
            address,
            3840,
            1080,
            args.stream_fps,
            args.quest_bitrate,
            audio=False,
            screen_curvature=getattr(args, "quest_screen_curvature", 0.0),
            ambient_enabled=getattr(args, "quest_ambient", False),
            ambient_brightness=getattr(args, "quest_ambient_brightness", 0.25),
            ambient_spread=getattr(args, "quest_ambient_spread", 2.0),
        )
        self.server.set_menu(QuestMenu(self.snapshot, self.submit))
        self.server.start()
        restart_app = False
        retrying = False
        desired = self.values(args)
        try:
            self.refresh_choices()
            while not self.stop_event.is_set():
                self.run_stop.clear()
                self._committed = False
                self.status(
                    "loading",
                    "正在重启串流；设置菜单保持可用。"
                    if self._operation == "restart_stream"
                    else "正在准备媒体；设置菜单保持可用。",
                )
                try:
                    for key, value in desired.items():
                        setattr(args, key, copy.deepcopy(value))
                    args.mapper = self._requested_mapper
                    utils.set_state_args(
                        args,
                        args_lock=self._args_lock,
                        stop_event=self.run_stop,
                        fps_event=self._fps_event,
                        quest_owner=True,
                    )
                    args.state["quest_session"] = self
                    run_once(args, init_wxapp=init_wxapp)
                except SessionInterrupted:
                    pass
                except Exception as error:
                    self._operation = None
                    self.server.take_media_error()
                    self.status("error", str(error) + "；可修改设置、重试或恢复上次配置。")
                    if self.values(args) != self._last_good and not retrying and not self.run_stop.is_set():
                        self._recovery_message = "新配置失败，已恢复上次成功配置：" + str(error)
                        desired, retrying = copy.deepcopy(self._last_good), True
                        self._requested_mapper = self._last_good_mapper
                        continue
                finally:
                    self.capture = None
                    self.server.set_controller_factory(None)
                    self.server.pause_media()
                    # reset只清时序状态；释放所有旧模型引用后才能在8GB显存加载下一模型。
                    args.state["depth_model"] = None
                    args.state["convergence_model"] = None
                    gc_collect()
                if self.stop_event.is_set():
                    break
                for key, value in self._values.items():
                    setattr(args, key, copy.deepcopy(value))
                while not self.stop_event.is_set():
                    self.apply_live()
                    with self._lock:
                        request, self._request = self._request, None
                        if request is None:
                            self._lock.wait(0.1)
                    if request is None:
                        continue
                    command, patch = request
                    self._operation = command
                    if command == "restart_app":
                        restart_app = True
                        break
                    desired = copy.deepcopy(self._last_good if command == "restore" else self._values)
                    if command == "restore":
                        self._requested_mapper = self._last_good_mapper
                    desired.update(patch)
                    retrying = False
                    self._recovery_message = ""
                    break
                if restart_app:
                    break
        finally:
            self.server.stop()
            args.state["stop_event"] = self.stop_event
            args.state["fps_event"] = self._fps_event
            self.stop_event.clear()
        if restart_app:
            from .quest_restart import restart_process, write_resume

            values = copy.deepcopy(self._values)
            values["bind_addr"] = address
            resume = write_resume(values, gui=self._fps_event is not None)
            args.state["quest_restart_path"] = str(resume)
            if self._fps_event is not None:
                self._fps_event.restart_requested(str(resume))
            else:
                restart_process(resume, gui=False)
        return args
