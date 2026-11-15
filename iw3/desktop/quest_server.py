import ipaddress
import json
import math
import select
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .quest_ambient import AmbientBackground
from .quest_audio import QuestAudio
from .quest_encoder import QuestEncoder
from .quest_input import QuestController, _rotate
from .quest_screen import QuestScreen


@dataclass
class _MediaRequest:
    settings: tuple[Any, ...] | None
    done: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    cancelled: bool = False


class QuestServer:
    def __init__(
        self,
        host: str,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        audio: bool = True,
        port: int = 52301,
        discovery_port: int = 52300,
        controller_factory: Callable[[], QuestController] | None = None,
        audio_source: str = "speaker",
        audio_gain: float = 1.0,
        screen_curvature: float = 0.0,
        ambient_enabled: bool = False,
        ambient_brightness: float = 0.25,
        ambient_spread: float = 2.0,
    ) -> None:
        address = ipaddress.IPv4Address(host)
        private_networks = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8")
        if not any(address in ipaddress.IPv4Network(network) for network in private_networks):
            raise ValueError("Quest output requires an explicit private LAN IPv4 address")
        if not all(type(value) is int for value in (width, height, fps, bitrate, port, discovery_port)):
            raise ValueError("Quest video settings and ports must be integers")
        if width <= 0 or height <= 0 or width % 4 or height % 2 or fps <= 0 or not 5 <= bitrate <= 100:
            raise ValueError(
                "Quest requires positive Full SBS dimensions (width % 4 = 0, height % 2 = 0), FPS and 5-100 Mbps"
            )
        if not 0 <= port <= 65535 or not 0 <= discovery_port <= 65535:
            raise ValueError("Quest ports must be between 0 and 65535")
        if type(screen_curvature) not in (int, float) or not 0 <= screen_curvature <= 120:
            raise ValueError("Quest screen curvature must be a finite number between 0 and 120 degrees")
        self._validate_ambient(ambient_enabled, ambient_brightness, ambient_spread)
        self.host, self.port, self.discovery_port = host, port, discovery_port
        self.width, self.height, self.fps, self.bitrate = width, height, fps, bitrate
        self.audio = audio
        self.audio_source = audio_source
        self.audio_gain = audio_gain
        self._screen_curvature = float(screen_curvature)
        self._cylinder_supported: bool | None = None
        self._ambient_enabled = ambient_enabled
        self._ambient_brightness = float(ambient_brightness)
        self._ambient_spread = float(ambient_spread)
        self._ambient_background: AmbientBackground | None = None
        self._ambient_generation = 0
        self._ambient_pending: tuple[bytes, int] | None = None
        self._ambient_visible = False
        self._controller_factory = controller_factory
        self._controller: QuestController | None = None
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._layer_lock = threading.RLock()
        self._video_config_lock = threading.Lock()
        self._audio_config_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._shutdown.set()
        self._need_idr = False
        self._connection: socket.socket | None = None
        self._listener: socket.socket | None = None
        self._discovery: socket.socket | None = None
        self._source: QuestAudio | None = None
        self._pending: tuple[torch.Tensor, float, torch.cuda.Event | None, int] | None = None
        self._generation = 0
        self._video_enabled = True
        self._video_request: _MediaRequest | None = None
        self._audio_request: _MediaRequest | None = None
        self._media_error: Exception | None = None
        self._menu: Any = None
        self._screen: QuestScreen | None = None
        self._last_input: float | None = None
        self._recenter_requested = threading.Event()
        self._sent_layers: dict[str, Any] = self._layers()
        self._sent_quad = {key: self._main_video(self._sent_layers)[key].copy() for key in ("pos", "rot", "tam")}
        self._threads: list[threading.Thread] = []
        self._sent: deque[float] = deque(maxlen=120)
        self._error: Exception | None = None

    def start(self) -> None:
        if not self._shutdown.is_set():
            return
        self._error = None
        self._media_error = None
        self._video_enabled = True
        self._video_request = self._audio_request = None
        self._shutdown.clear()
        try:
            self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._listener.bind((self.host, self.port))
            self._listener.listen(1)
            self._listener.settimeout(0.2)
            self.port = self._listener.getsockname()[1]
            self._discovery = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 广播只接收固定发现字符串，不在通配地址开放视频服务。
            self._discovery.bind(("", self.discovery_port))
            self._discovery.settimeout(0.2)
            self.discovery_port = self._discovery.getsockname()[1]
        except Exception as error:
            self.stop()
            if isinstance(error, OSError):
                raise RuntimeError("Quest端口无法绑定；请先关闭电脑上的Pop3D，并检查所选局域网地址") from error
            raise
        self._spawn(self._video_loop)
        self._spawn(self._audio_loop)
        try:
            if self.audio:
                self.configure_audio(True, self.audio_source, self.audio_gain)
        except Exception:
            self.stop()
            raise
        self._spawn(self._accept_loop, self._listener)
        self._spawn(self._discovery_loop, self._discovery)

    def _spawn(self, target: Any, *args: Any) -> None:
        thread = threading.Thread(target=target, args=args, daemon=True)
        with self._condition:
            if self._shutdown.is_set():
                return
            self._threads = [existing for existing in self._threads if existing.is_alive()]
            self._threads.append(thread)
            thread.start()

    @staticmethod
    def _close(connection: socket.socket | None) -> None:
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def stop(self) -> None:
        self._shutdown.set()
        release_error = None
        with self._condition:
            if self._controller is not None:
                try:
                    self._controller.reset()
                    self._controller = None
                except Exception as error:
                    release_error = error
            connection, self._connection = self._connection, None
            self._pending = None
            for request in (self._video_request, self._audio_request):
                if request is not None:
                    request.cancelled = True
                    request.error = RuntimeError("Quest媒体已停止")
                    request.done.set()
            self._condition.notify_all()
        with self._layer_lock:
            with self._condition:
                self._reset_ambient()
            self._cylinder_supported = None
            self._reset_input()
        self._close(connection)
        self._close(self._listener)
        self._close(self._discovery)
        self._listener = self._discovery = None
        deadline = time.monotonic() + 2
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(max(0, deadline - time.monotonic()))
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        if self._threads:
            raise RuntimeError("Quest输出线程未能及时停止")
        self._sent.clear()
        if release_error is not None:
            raise RuntimeError(f"Quest手柄释放失败：{release_error}") from release_error

    def _fail(self, error: Exception) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._shutdown.set()
            self._condition.notify_all()
        self._disconnect(self._connection)
        self._close(self._listener)
        self._close(self._discovery)

    def check_error(self) -> None:
        error = self._error or self.media_error
        if error is not None:
            raise RuntimeError(f"Quest输出失败：{error}") from error

    @property
    def media_error(self) -> Exception | None:
        with self._condition:
            return self._media_error

    def take_media_error(self) -> Exception | None:
        with self._condition:
            error, self._media_error = self._media_error, None
            return error

    def _media_failed(self, error: Exception) -> None:
        with self._condition:
            if self._media_error is None:
                self._media_error = error.with_traceback(None)
            self._condition.notify_all()

    def _request_media(self, video: bool, settings: tuple[Any, ...] | None) -> None:
        request = _MediaRequest(settings)
        lock = self._video_config_lock if video else self._audio_config_lock
        if not lock.acquire(timeout=5):
            raise RuntimeError("Quest媒体重配置等待超时")
        try:
            with self._condition:
                if self._shutdown.is_set():
                    raise RuntimeError("Quest服务尚未启动")
                remove_ambient = video and self._ambient_visible
                if video:
                    self._generation += 1
                    self._pending = None
                    self._video_enabled = False
                    self._video_request = request
                    self._reset_ambient()
                else:
                    self._audio_request = request
                connection = self._connection
                self._condition.notify_all()
            if remove_ambient and connection is not None:
                with self._layer_lock:
                    try:
                        self._send_layers(connection)
                    except OSError:
                        self._disconnect(connection)
            # 不持condition等待：owner必须取得同一锁来完成命令。
            if not request.done.wait(5):
                with self._condition:
                    request.cancelled = True
                    if video:
                        self._generation += 1
                        self._pending = None
                raise RuntimeError("Quest媒体重配置执行超时")
            if request.error is not None:
                raise request.error
        finally:
            lock.release()

    def configure_video(self, width: int, height: int, fps: int, bitrate: int) -> None:
        if not all(type(value) is int for value in (width, height, fps, bitrate)):
            raise ValueError("Quest video settings must be integers")
        if width <= 0 or height <= 0 or width % 4 or height % 2 or fps <= 0 or not 5 <= bitrate <= 100:
            raise ValueError("Quest requires Full SBS dimensions, positive FPS and 5-100 Mbps")
        self._request_media(True, (width, height, fps, bitrate))

    def configure_audio(self, enabled: bool, source: str, gain: float) -> None:
        if type(enabled) is not bool or source not in {"speaker", "process"}:
            raise ValueError("Invalid Quest audio configuration")
        if type(gain) not in (int, float) or not 0 <= gain <= 4:
            raise ValueError("Quest音量增益必须是0到4之间的有限数字")
        self._request_media(False, (enabled, source, gain))

    def pause_media(self) -> None:
        self._request_media(True, None)
        self.configure_audio(False, self.audio_source, self.audio_gain)

    @staticmethod
    def _validate_ambient(enabled: bool, brightness: float, spread: float) -> None:
        if type(enabled) is not bool:
            raise ValueError("Quest ambient enabled must be a boolean")
        if type(brightness) not in (int, float) or not 0 <= brightness <= 1:
            raise ValueError("Quest ambient brightness must be a finite number between 0 and 1")
        if type(spread) not in (int, float) or not 1.2 <= spread <= 4:
            raise ValueError("Quest ambient spread must be a finite number between 1.2 and 4")

    def _reset_ambient(self) -> None:
        # 只丢弃历史；持server锁时不构造含Torch运算的生成器。
        self._ambient_background = None
        self._ambient_generation += 1
        self._ambient_pending = None
        self._ambient_visible = False

    def configure_ambient(self, enabled: bool, brightness: float, spread: float) -> None:
        self._validate_ambient(enabled, brightness, spread)
        with self._layer_lock:
            with self._condition:
                if (enabled, brightness, spread) == (
                    self._ambient_enabled,
                    self._ambient_brightness,
                    self._ambient_spread,
                ):
                    return
                self._ambient_enabled = enabled
                self._ambient_brightness = float(brightness)
                self._ambient_spread = float(spread)
                self._reset_ambient()
                connection = self._connection
                self._condition.notify_all()
            if connection is not None:
                try:
                    self._send_layers(connection)
                except OSError:
                    self._disconnect(connection)

    def set_ambient_frame(self, frame: torch.Tensor, content_rect: tuple[int, int, int, int] | None = None) -> None:
        with self._condition:
            if (
                self._shutdown.is_set()
                or not self._video_enabled
                or not self._ambient_enabled
                or self._connection is None
            ):
                return
            generation = self._ambient_generation
            background = self._ambient_background
            brightness = self._ambient_brightness
        # 实测RGB float CHW[3,H,W]→压缩RGBA[90,160,4]；提色与压缩由媒体owner在锁外完成。
        if background is None:
            background = AmbientBackground()
        payload = background.update(frame, brightness, content_rect)
        if payload is None:
            return
        with self._condition:
            if generation == self._ambient_generation and self._video_enabled and not self._shutdown.is_set():
                self._ambient_background = background
                self._ambient_pending = payload, generation
                self._condition.notify_all()

    def set_menu(self, menu: Any) -> None:
        with self._layer_lock:
            if self._menu is not None:
                self._menu.reset()
            self._menu = menu

    @property
    def cylinder_supported(self) -> bool | None:
        with self._layer_lock:
            return self._cylinder_supported

    @property
    def actual_cylinder(self) -> bool:
        # 这是已发送的配置，协议没有头显渲染成功的ACK。
        with self._layer_lock:
            return self._cylinder_supported is True and self._main_video(self._sent_layers)["tipo"] == "cilindro"

    def set_screen_curvature(self, degrees: float) -> None:
        if type(degrees) not in (int, float) or not 0 <= degrees <= 120:
            raise ValueError("Quest screen curvature must be a finite number between 0 and 120 degrees")
        with self._layer_lock:
            if degrees == self._screen_curvature:
                return
            self._screen_curvature = float(degrees)
            with self._condition:
                connection = self._connection
            if connection is not None:
                self._control(connection, "reset")
                if self._screen is not None:
                    self._screen.reset()
                try:
                    self._send_layers(connection)
                except OSError:
                    self._disconnect(connection)

    def set_controller_factory(self, factory: Callable[[], QuestController] | None) -> None:
        with self._layer_lock:
            with self._condition:
                connection = self._connection
                self._controller_factory = factory
                if self._controller is not None:
                    self._controller.reset()
                    self._controller = None
            replacement = factory() if connection is not None and factory is not None else None
            with self._condition:
                if connection is self._connection and factory is self._controller_factory:
                    self._controller = replacement
                elif replacement is not None:
                    replacement.reset()

    def recenter(self) -> None:
        self._recenter_requested.set()
        with self._condition:
            connection = self._connection
        self._reset_input()
        if connection is not None:
            self._control(connection, "reset")

    def set_audio_gain(self, gain: float) -> None:
        if type(gain) not in (int, float) or not 0 <= gain <= 4:
            raise ValueError("Quest音量增益必须是0到4之间的有限数字")
        with self._condition:
            if gain == self.audio_gain:
                return
            self.audio_gain = gain
            if self._source is not None:
                self._source.set_gain(gain)

    def set_frame_data(self, frame_data: tuple[torch.Tensor, float]) -> None:
        frame, tick = frame_data
        with self._condition:
            generation = self._generation
            width, height = self.width, self.height
            enabled = self._video_enabled
        if frame.shape != (3, height, width) or frame.dtype not in (torch.float16, torch.float32):
            raise ValueError(f"Expected float RGB CHW {(3, height, width)}")
        if frame.device.type not in ("cpu", "cuda") or not math.isfinite(tick) or tick < 0:
            raise ValueError("Quest frames require CPU/CUDA tensors and nonnegative finite timestamps")
        if self._shutdown.is_set() or not enabled:
            return
        with torch.inference_mode():
            # 实测RGB float CHW[3,H,2W]；独占副本与生产流事件保护异步编码读取。
            owned = frame.detach().clone()
            ready = None
            if frame.is_cuda:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(frame.device))
        with self._condition:
            if not self._shutdown.is_set() and self._video_enabled and generation == self._generation:
                self._pending = owned, tick, ready, generation
                self._condition.notify_all()

    def get_fps(self) -> float:
        with self._condition:
            if len(self._sent) < 2 or time.monotonic() - self._sent[-1] > 1:
                return 0.0
            return (len(self._sent) - 1) / max(self._sent[-1] - self._sent[0], 1e-6)

    def _disconnect(self, connection: socket.socket | None) -> None:
        release_error = None
        disconnected = False
        with self._layer_lock:
            with self._condition:
                if connection is not None and connection is self._connection:
                    disconnected = True
                    if self._controller is not None:
                        try:
                            self._controller.reset()
                            self._controller = None
                        except Exception as error:
                            release_error = error
                    self._connection = None
                    self._pending = None
                    self._reset_ambient()
                    if release_error is not None:
                        self._shutdown.set()
                    self._condition.notify_all()
            if disconnected:
                self._cylinder_supported = None
                self._reset_input()
        self._close(connection)
        if release_error is not None:
            self._fail(release_error)

    def _control(self, connection: socket.socket, action: str, *args: Any) -> Any:
        # 与断开/重连串行，旧连接不能在新会话中注入或释放输入。
        with self._condition:
            if connection is self._connection and self._controller is not None:
                try:
                    return getattr(self._controller, action)(*args)
                except Exception as error:
                    raise RuntimeError(f"Quest手柄控制失败：{error}") from error
        return None

    def _send(self, connection: socket.socket, kind: int, payload: bytes) -> None:
        with self._write_lock:
            connection.sendall(struct.pack("!BI", kind, len(payload)) + payload)

    def _send_json(self, connection: socket.socket, kind: int, payload: dict[str, Any]) -> None:
        self._send(connection, kind, json.dumps(payload, separators=(",", ":")).encode())

    @staticmethod
    def _main_video(layers: dict[str, Any]) -> dict[str, Any]:
        return next(layer for layer in layers["camadas"] if layer.get("tex") == "video" and layer.get("olho") == "esq")

    def _layers(
        self, pose: tuple[list[float], list[float]] | None = None, size: list[float] | None = None
    ) -> dict[str, Any]:
        position, rotation = pose if pose is not None else ([0.0, 0.0, -3.0], [0.0, 0.0, 0.0, 1.0])
        eye_width = self.width // 2
        size = size if size is not None else [3.6, 3.6 * self.height / eye_width]
        # 实测中面dict(pos[3],rot[4],tam[2]) → 双眼camadas[2]；曲面只换几何，不改SBS rect[4]。
        geometry: dict[str, Any] = {"tipo": "quad", "pos": list(position), "rot": list(rotation), "tam": list(size)}
        angle = math.radians(self._screen_curvature)
        # 原生层要求角度>0.001rad；未知/不支持能力时与射线一起保留平面。
        if self._cylinder_supported is True and angle > 0.001:
            radius = size[0] / angle
            offset = _rotate(rotation, [0.0, 0.0, radius])
            geometry = {
                "tipo": "cilindro",
                "pos": [a + b for a, b in zip(position, offset)],
                "rot": list(rotation),
                "raio": radius,
                "angulo": angle,
                "aspecto": size[0] / size[1],
            }
        layers: dict[str, Any] = {
            "camadas": [
                {
                    **geometry,
                    "tex": "video",
                    "olho": eye,
                    "alpha": False,
                    "rect": [index * eye_width, 0, eye_width, self.height],
                }
                for index, eye in enumerate(("esq", "dir"))
            ]
        }
        if self._ambient_visible:
            ambient_position = [a + b for a, b in zip(position, _rotate(rotation, [0.0, 0.0, -0.05]))]
            ambient: dict[str, Any] = {
                "tipo": "quad",
                "pos": ambient_position,
                "rot": list(rotation),
                "tam": [value * self._ambient_spread for value in size],
            }
            if self._cylinder_supported is True:
                base_angle = max(angle, math.pi / 2)
                radius = size[0] / base_angle
                ambient_angle = min(math.radians(170), base_angle * self._ambient_spread)
                offset = _rotate(rotation, [0.0, 0.0, radius])
                ambient = {
                    "tipo": "cilindro",
                    "pos": [a + b for a, b in zip(ambient_position, offset)],
                    "rot": list(rotation),
                    "raio": radius,
                    "angulo": ambient_angle,
                    "aspecto": radius * ambient_angle / (size[1] * self._ambient_spread),
                }
            layers["camadas"].insert(
                0,
                {
                    **ambient,
                    "tex": AmbientBackground.TEXTURE_ID,
                    "olho": "ambos",
                    "alpha": True,
                    "rect": [0, 0, AmbientBackground.WIDTH, AmbientBackground.HEIGHT],
                },
            )
        return layers

    def _reset_input(self) -> None:
        with self._layer_lock:
            self._last_input = None
            if self._screen is not None:
                self._screen.reset()
            if self._menu is not None:
                self._menu.reset()

    def _send_layers(self, connection: socket.socket, quad: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._layer_lock:
            if quad is None:
                quad = self._sent_quad
            layers = self._layers((quad["pos"], quad["rot"]), quad["tam"])
            if self._menu is not None:
                texture = self._menu.texture()
                if texture is not None:
                    self._send(connection, 0x10, texture)
                layers["camadas"].extend(self._menu.layers())
            self._send_json(connection, 0x11, layers)
            self._sent_layers = layers
            self._sent_quad = {key: quad[key].copy() for key in ("pos", "rot", "tam")}
            return layers

    def _video_config(self, connection: socket.socket) -> None:
        self._send_json(
            connection, 0x01, {"largura": self.width, "altura": self.height, "fps": self.fps, "codec": "hevc"}
        )

    def _send_ambient(self, connection: socket.socket) -> None:
        with self._layer_lock:
            with self._condition:
                pending = self._ambient_pending
                if (
                    pending is None
                    or pending[1] != self._ambient_generation
                    or connection is not self._connection
                    or not self._video_enabled
                    or not self._ambient_enabled
                    or self._shutdown.is_set()
                ):
                    return
                self._ambient_pending = None
            self._send(connection, 0x10, pending[0])
            with self._condition:
                if pending[1] != self._ambient_generation or not self._video_enabled:
                    return
                first = not self._ambient_visible
                self._ambient_visible = True
            if first:
                self._send_layers(connection)

    def _accept_loop(self, listener: socket.socket) -> None:
        try:
            while not self._shutdown.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with self._condition:
                    occupied = self._connection is not None or self._shutdown.is_set()
                if occupied:
                    self._close(connection)
                    continue
                connection.settimeout(0.5)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._layer_lock:
                    try:
                        self._reset_input()
                        self._cylinder_supported = None
                        self._video_config(connection)
                        with self._condition:
                            self._reset_ambient()
                        self._send_layers(connection, self._main_video(self._layers()))
                        if self._source is not None:
                            self._send_json(connection, 0x03, {"taxa": self._source.sample_rate, "canais": 2})
                    except OSError:
                        self._close(connection)
                        continue
                    try:
                        controller = self._controller_factory() if self._controller_factory is not None else None
                    except Exception as error:
                        self._close(connection)
                        self._fail(error)
                        return
                    with self._condition:
                        if self._shutdown.is_set():
                            self._close(connection)
                            return
                        self._connection = connection
                        self._controller = controller
                        self._need_idr = True
                        self._sent.clear()
                        self._condition.notify_all()
                self._spawn(self._input_loop, connection)
        except OSError as error:
            if not self._shutdown.is_set():
                self._fail(error)

    def _discovery_loop(self, discovery: socket.socket) -> None:
        reply = b"ESTUDIO3D!" + json.dumps({"nome": "iw3", "porta": self.port}).encode()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as replies:
                # Quest把发现回包的源IP作为TCP地址，必须从用户选定的网卡回复。
                replies.bind((self.host, 0))
                replies.settimeout(0.5)
                while not self._shutdown.is_set():
                    try:
                        request, peer = discovery.recvfrom(128)
                    except socket.timeout:
                        continue
                    if request == b"ESTUDIO3D?":
                        replies.sendto(reply, peer)
        except OSError as error:
            if not self._shutdown.is_set():
                self._fail(error)

    def _read_exact(self, connection: socket.socket, size: int, idle_ok: bool = False) -> bytes:
        data = bytearray()
        while len(data) < size and not self._shutdown.is_set():
            remaining = self._expire_input(connection)
            # 只缩短输入等待，不改视频/音频共用 socket 的发送超时。
            if remaining is not None and not select.select([connection], [], [], remaining)[0]:
                self._expire_input(connection)
                if idle_ok and not data:
                    continue
                raise socket.timeout("Quest input packet timed out")
            try:
                part = connection.recv(size - len(data))
            except socket.timeout:
                self._expire_input(connection)
                if idle_ok and not data:
                    continue
                raise
            if not part:
                raise ConnectionError("Quest connection closed")
            data.extend(part)
        if len(data) != size:
            raise ConnectionError("Quest output stopped")
        return bytes(data)

    def _expire_input(self, connection: socket.socket) -> float:
        with self._layer_lock:
            with self._condition:
                if connection is not self._connection:
                    return 0.5
            remaining = self._control(connection, "expire")
            remaining = 0.5 if remaining is None else remaining
            if self._last_input is not None:
                menu_remaining = 0.5 - (time.monotonic() - self._last_input)
                if menu_remaining <= 0:
                    visible = self._menu is not None and self._menu.visible
                    self._reset_input()
                    self._control(connection, "reset")
                    if visible:
                        self._send_layers(connection)
                else:
                    remaining = min(remaining, menu_remaining)
        return max(0.001, remaining)

    @staticmethod
    def _pose(data: Any) -> tuple[list[float], list[float]] | None:
        if not isinstance(data, dict):
            return None
        head = data.get("cabeca")
        if not isinstance(head, dict):
            return None
        position, rotation = head.get("pos"), head.get("rot")
        if not isinstance(position, list) or not isinstance(rotation, list) or len(position) != 3 or len(rotation) != 4:
            return None
        if not all(type(value) in (int, float) for value in position + rotation):
            return None
        try:
            position = [float(value) for value in position]
            rotation = [float(value) for value in rotation]
        except OverflowError:
            return None
        if not all(math.isfinite(value) for value in position + rotation):
            return None
        norm = math.hypot(*rotation)
        if norm < 1e-6 or not math.isfinite(norm) or max(map(abs, position)) > 10_000:
            return None
        qx, qy, qz, qw = (value / norm for value in rotation)
        forward_x, forward_z = -2 * (qx * qz + qw * qy), -(1 - 2 * (qx * qx + qy * qy))
        yaw = math.atan2(-forward_x, -forward_z) if math.hypot(forward_x, forward_z) > 0.2 else 0.0
        return (
            [position[0] - 3 * math.sin(yaw), position[1], position[2] - 3 * math.cos(yaw)],
            [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)],
        )

    def _input_loop(self, connection: socket.socket) -> None:
        anchored = False
        sent_at = 0.0
        try:
            with self._layer_lock:
                with self._condition:
                    if connection is not self._connection:
                        return
                screen = QuestScreen(self.height / (self.width // 2))
                self._screen = screen
            while not self._shutdown.is_set():
                self._expire_input(connection)
                header = self._read_exact(connection, 5, idle_ok=True)
                size = int.from_bytes(header[1:], "big")
                if size > 65536:
                    raise ValueError("Quest input exceeds 64 KiB")
                payload = self._read_exact(connection, size)
                if header[0] == 0x82:
                    with self._condition:
                        if connection is self._connection:
                            self._need_idr = True
                elif header[0] == 0x84:
                    try:
                        hello = json.loads(payload)
                    except (ValueError, RecursionError):
                        continue
                    if not isinstance(hello, dict) or type(hello.get("cilindro")) is not bool:
                        continue
                    with self._layer_lock:
                        with self._condition:
                            if connection is not self._connection:
                                return
                        if hello["cilindro"] != self._cylinder_supported:
                            self._cylinder_supported = hello["cilindro"]
                            self._control(connection, "reset")
                            screen.reset()
                            self._send_layers(connection)
                elif header[0] == 0x83:
                    try:
                        data = json.loads(payload)
                    except (ValueError, RecursionError):
                        with self._layer_lock:
                            with self._condition:
                                if connection is not self._connection:
                                    return
                            visible = self._menu is not None and self._menu.visible
                            self._reset_input()
                            self._control(connection, "reset")
                            if visible:
                                self._send_layers(connection)
                        continue
                    with self._layer_lock:
                        with self._condition:
                            if connection is not self._connection:
                                return
                        pose = self._pose(data)
                        recentered = pose is not None and (
                            not anchored or data.get("recentrou") is True or self._recenter_requested.is_set()
                        )
                        if recentered:
                            self._recenter_requested.clear()
                            assert pose is not None
                            self._control(connection, "reset")
                            self._reset_input()
                            screen.recenter(pose)
                            anchored = True
                        self._last_input = time.monotonic()
                        video_quad = self._sent_quad
                        if isinstance(data, dict) and data.get("estado") != "focado" and self._menu is not None:
                            self._menu.reset()
                        suppress_pc = self._menu.update(data, video_quad) if self._menu is not None else False
                        if suppress_pc:
                            self._control(connection, "reset")
                            screen.reset()
                        else:
                            suppress_pc = screen.update(data, self._main_video(self._sent_layers))
                        quad = screen.quad
                        layers = self._layers((quad["pos"], quad["rot"]), quad["tam"])
                        if self._menu is not None:
                            layers["camadas"].extend(self._menu.layers())
                        now = time.monotonic()
                        # 变化层最多30Hz；停止抓取时发送最终值，点击始终使用已发几何。
                        if self._menu is not None:
                            texture = self._menu.texture()
                            if texture is not None:
                                self._send(connection, 0x10, texture)
                        if recentered or (
                            layers != self._sent_layers and (not screen.active or now - sent_at >= 1 / 30)
                        ):
                            self._send_layers(connection, quad)
                            sent_at = now
                        if suppress_pc:
                            self._control(connection, "reset")
                        else:
                            self._control(connection, "update", data, self._main_video(self._sent_layers))
        except (OSError, ValueError):
            pass
        except Exception as error:
            self._fail(error)
        finally:
            self._disconnect(connection)

    def _video_loop(self) -> None:
        current: socket.socket | None = None
        encoder: QuestEncoder | None = None
        streams: dict[torch.device, torch.cuda.Stream] = {}
        last_pts = -1
        while not self._shutdown.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._shutdown.is_set()
                        or self._video_request is not None
                        or (self._video_enabled and self._connection is not None and self._pending is not None)
                    )
                )
                if self._shutdown.is_set():
                    return
                request, self._video_request = self._video_request, None
                connection, pending = self._connection, self._pending
                self._pending = None
                force_idr, self._need_idr = self._need_idr, False
            if request is not None:
                try:
                    encoder = None
                    self._video_enabled = False
                    if request.settings is not None and not request.cancelled:
                        width, height, fps, bitrate = request.settings
                        encoder = QuestEncoder(width, height, fps, bitrate)
                        with self._layer_lock:
                            with self._condition:
                                if request.cancelled or self._shutdown.is_set():
                                    encoder = None
                                    continue
                                self.width, self.height, self.fps, self.bitrate = width, height, fps, bitrate
                                self._video_enabled, self._need_idr = True, True
                                self._sent.clear()
                                connection = self._connection
                            if connection is not None:
                                try:
                                    quad = self._sent_quad.copy()
                                    quad["tam"] = [quad["tam"][0], quad["tam"][0] * height / (width // 2)]
                                    if self._screen is not None:
                                        self._screen.set_aspect(height / (width // 2))
                                    self._video_config(connection)
                                    self._send_layers(connection, quad)
                                except OSError:
                                    self._disconnect(connection)
                except Exception as error:
                    request.error = error
                    self._media_failed(error)
                finally:
                    request.done.set()
                continue
            assert connection is not None and pending is not None
            frame, tick, ready, generation = pending
            try:
                if current is not connection:
                    if current is not None:
                        encoder = None
                    current, last_pts = connection, -1
                    force_idr = True
                if encoder is None:
                    encoder = QuestEncoder(self.width, self.height, self.fps, self.bitrate)
                    force_idr = True
                pts = max(last_pts + 1, int(tick * 1_000_000))
                if frame.is_cuda:
                    if frame.device not in streams:
                        streams[frame.device] = torch.cuda.Stream(device=frame.device)
                    stream = streams[frame.device]
                    assert ready is not None
                    ready.wait(stream)
                    with torch.cuda.stream(stream):
                        frame.record_stream(stream)
                        packets = encoder.encode(frame, pts, force_idr=force_idr)
                else:
                    packets = encoder.encode(frame, pts, force_idr=force_idr)
                with self._condition:
                    stale = generation != self._generation or connection is not self._connection
                if stale:
                    continue
                last_pts = pts
                try:
                    # 已编码P帧只在代次被废弃时跳过；新代次始终从IDR重建引用。
                    for packet in packets:
                        self._send(
                            connection, 0x02, struct.pack("!QB", packet.pts, int(packet.is_keyframe)) + bytes(packet)
                        )
                    if packets:
                        with self._condition:
                            self._sent.append(time.monotonic())
                    self._send_ambient(connection)
                except OSError:
                    self._disconnect(connection)
            except Exception as error:
                encoder = None
                with self._layer_lock:
                    with self._condition:
                        remove_ambient = self._ambient_visible
                        self._video_enabled = False
                        self._pending = None
                        self._reset_ambient()
                    if remove_ambient:
                        try:
                            self._send_layers(connection)
                        except OSError:
                            self._disconnect(connection)
                self._media_failed(error)
            finally:
                del frame, ready, pending

    def _audio_loop(self) -> None:
        source: QuestAudio | None = None
        try:
            while not self._shutdown.is_set():
                with self._condition:
                    self._condition.wait(0.005)
                    request, self._audio_request = self._audio_request, None
                    connection = self._connection
                if request is not None:
                    try:
                        assert request.settings is not None
                        enabled, name, gain = request.settings
                        if request.cancelled:
                            continue
                        created = False
                        if source is not None and (not enabled or name != self.audio_source):
                            with self._condition:
                                self._source = None
                            source.stop()
                            source = None
                        if enabled and not request.cancelled:
                            if source is None:
                                source = QuestAudio(gain=gain, source=name)
                                source.start()
                                created = True
                            else:
                                source.set_gain(gain)
                        with self._layer_lock:
                            if request.cancelled or self._shutdown.is_set():
                                if source is not None:
                                    source.stop()
                                source = None
                                continue
                            with self._condition:
                                self._source = source
                                self.audio, self.audio_source, self.audio_gain = enabled, name, gain
                                connection = self._connection
                            if created and source is not None and connection is not None:
                                try:
                                    self._send_json(connection, 0x03, {"taxa": source.sample_rate, "canais": 2})
                                except OSError:
                                    self._disconnect(connection)
                    except Exception as error:
                        request.error = error
                        if source is not None:
                            try:
                                source.stop()
                            except Exception:
                                pass
                        source = self._source = None
                        self.audio = False
                        self._media_failed(error)
                    finally:
                        request.done.set()
                    continue
                if source is None:
                    continue
                try:
                    source.check_error()
                    chunks = source.get_chunks()
                    with self._condition:
                        connection = self._connection
                    if connection is not None:
                        try:
                            for chunk in chunks:
                                self._send(connection, 0x04, struct.pack("!Q", time.monotonic_ns() // 1000) + chunk)
                        except OSError:
                            self._disconnect(connection)
                except Exception as error:
                    self._media_failed(error)
                    try:
                        source.stop()
                    except Exception as stop_error:
                        self._media_failed(stop_error)
                    source = self._source = None
                    self.audio = False
        finally:
            if source is not None:
                try:
                    source.stop()
                except Exception as error:
                    self._media_failed(error)
            self._source = None
