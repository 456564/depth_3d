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
from typing import Any

import torch

from .quest_audio import QuestAudio
from .quest_encoder import QuestEncoder
from .quest_input import QuestController
from .quest_screen import QuestScreen


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
        self.host, self.port, self.discovery_port = host, port, discovery_port
        self.width, self.height, self.fps, self.bitrate = width, height, fps, bitrate
        self.audio = audio
        self.audio_source = audio_source
        self.audio_gain = audio_gain
        self._controller_factory = controller_factory
        self._controller: QuestController | None = None
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._shutdown.set()
        self._need_idr = False
        self._connection: socket.socket | None = None
        self._listener: socket.socket | None = None
        self._discovery: socket.socket | None = None
        self._source: QuestAudio | None = None
        self._pending: tuple[torch.Tensor, float, torch.cuda.Event | None] | None = None
        self._threads: list[threading.Thread] = []
        self._sent: deque[float] = deque(maxlen=120)
        self._error: Exception | None = None

    def start(self) -> None:
        if not self._shutdown.is_set():
            return
        self._error = None
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
            if self.audio:
                self._source = QuestAudio(gain=self.audio_gain, source=self.audio_source)
                self._source.start()
        except Exception as error:
            self.stop()
            if isinstance(error, OSError):
                raise RuntimeError("Quest端口无法绑定；请先关闭电脑上的Pop3D，并检查所选局域网地址") from error
            raise
        self._spawn(self._accept_loop, self._listener)
        self._spawn(self._discovery_loop, self._discovery)
        self._spawn(self._video_loop)
        self._spawn(self._audio_loop)

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
            self._condition.notify_all()
        self._close(connection)
        self._close(self._listener)
        self._close(self._discovery)
        self._listener = self._discovery = None
        deadline = time.monotonic() + 2
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(max(0, deadline - time.monotonic()))
        if self._source is not None:
            self._source.stop()
            self._source = None
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
        if self._error is not None:
            raise RuntimeError(f"Quest输出失败：{self._error}") from self._error

    def set_audio_gain(self, gain: float) -> None:
        if self._source is not None:
            self._source.set_gain(gain)
        self.audio_gain = gain

    def set_frame_data(self, frame_data: tuple[torch.Tensor, float]) -> None:
        frame, tick = frame_data
        if frame.shape != (3, self.height, self.width) or frame.dtype not in (torch.float16, torch.float32):
            raise ValueError(f"Expected float RGB CHW {(3, self.height, self.width)}")
        if frame.device.type not in ("cpu", "cuda") or not math.isfinite(tick) or tick < 0:
            raise ValueError("Quest frames require CPU/CUDA tensors and nonnegative finite timestamps")
        if self._shutdown.is_set():
            return
        with torch.inference_mode():
            # 实测RGB float CHW[3,H,2W]；独占副本与生产流事件保护异步编码读取。
            owned = frame.detach().clone()
            ready = None
            if frame.is_cuda:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(frame.device))
        with self._condition:
            if not self._shutdown.is_set():
                self._pending = owned, tick, ready
                self._condition.notify_all()

    def get_fps(self) -> float:
        with self._condition:
            if len(self._sent) < 2 or time.monotonic() - self._sent[-1] > 1:
                return 0.0
            return (len(self._sent) - 1) / max(self._sent[-1] - self._sent[0], 1e-6)

    def _disconnect(self, connection: socket.socket | None) -> None:
        release_error = None
        with self._condition:
            if connection is not None and connection is self._connection:
                if self._controller is not None:
                    try:
                        self._controller.reset()
                        self._controller = None
                    except Exception as error:
                        release_error = error
                self._connection = None
                if release_error is not None:
                    self._shutdown.set()
                self._condition.notify_all()
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

    def _layers(
        self, pose: tuple[list[float], list[float]] | None = None, size: list[float] | None = None
    ) -> dict[str, Any]:
        position, rotation = pose if pose is not None else ([0, 0, -3], [0, 0, 0, 1])
        eye_width = self.width // 2
        size = size if size is not None else [3.6, 3.6 * self.height / eye_width]
        return {
            "camadas": [
                {
                    "tipo": "quad",
                    "tex": "video",
                    "olho": eye,
                    "alpha": False,
                    "rect": [index * eye_width, 0, eye_width, self.height],
                    "pos": list(position),
                    "rot": list(rotation),
                    "tam": list(size),
                }
                for index, eye in enumerate(("esq", "dir"))
            ]
        }

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
                try:
                    self._send_json(
                        connection,
                        0x01,
                        {"largura": self.width, "altura": self.height, "fps": self.fps, "codec": "hevc"},
                    )
                    self._send_json(connection, 0x11, self._layers())
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
            remaining = self._control(connection, "expire")
            # 只缩短输入等待，不改视频/音频共用 socket 的发送超时。
            if remaining is not None and not select.select([connection], [], [], remaining)[0]:
                self._control(connection, "expire")
                if idle_ok and not data:
                    continue
                raise socket.timeout("Quest input packet timed out")
            try:
                part = connection.recv(size - len(data))
            except socket.timeout:
                self._control(connection, "expire")
                if idle_ok and not data:
                    continue
                raise
            if not part:
                raise ConnectionError("Quest connection closed")
            data.extend(part)
        if len(data) != size:
            raise ConnectionError("Quest output stopped")
        return bytes(data)

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
        screen = QuestScreen(self.height / (self.width // 2))
        anchored = False
        sent_layers = self._layers()
        sent_at = 0.0
        try:
            while not self._shutdown.is_set():
                self._control(connection, "expire")
                header = self._read_exact(connection, 5, idle_ok=True)
                size = int.from_bytes(header[1:], "big")
                if size > 65536:
                    raise ValueError("Quest input exceeds 64 KiB")
                payload = self._read_exact(connection, size)
                if header[0] == 0x82:
                    with self._condition:
                        if connection is self._connection:
                            self._need_idr = True
                elif header[0] == 0x83:
                    try:
                        data = json.loads(payload)
                    except (ValueError, RecursionError):
                        screen.reset()
                        self._control(connection, "reset")
                        continue
                    pose = self._pose(data)
                    recentered = pose is not None and (not anchored or data.get("recentrou") is True)
                    if recentered:
                        assert pose is not None
                        self._control(connection, "reset")
                        screen.recenter(pose)
                        anchored = True
                    suppress_pc = screen.update(data)
                    quad = screen.quad
                    layers = self._layers((quad["pos"], quad["rot"]), quad["tam"])
                    now = time.monotonic()
                    # 变化层最多30Hz；停止抓取时发送最终值，点击始终使用已发几何。
                    if recentered or (layers != sent_layers and (not screen.active or now - sent_at >= 1 / 30)):
                        self._send_json(connection, 0x11, layers)
                        sent_layers, sent_at = layers, now
                    if suppress_pc:
                        self._control(connection, "reset")
                    else:
                        self._control(connection, "update", data, sent_layers["camadas"][0])
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
        try:
            while not self._shutdown.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._shutdown.is_set() or (self._connection is not None and self._pending is not None)
                    )
                    if self._shutdown.is_set():
                        return
                    connection, pending = self._connection, self._pending
                    self._pending = None
                    force_idr, self._need_idr = self._need_idr, False
                assert connection is not None and pending is not None
                if current is not connection:
                    encoder = QuestEncoder(self.width, self.height, self.fps, self.bitrate)
                    current, last_pts = connection, -1
                    force_idr = True
                assert encoder is not None
                frame, tick, ready = pending
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
                last_pts = pts
                try:
                    # 已编码P帧不可跳过，否则后续引用帧无法解码；仅替换编码前的原始帧。
                    for packet in packets:
                        self._send(
                            connection, 0x02, struct.pack("!QB", packet.pts, int(packet.is_keyframe)) + bytes(packet)
                        )
                    if packets:
                        with self._condition:
                            self._sent.append(time.monotonic())
                except OSError:
                    self._disconnect(connection)
        except Exception as error:
            self._fail(error)

    def _audio_loop(self) -> None:
        source = self._source
        if source is None:
            return
        try:
            while not self._shutdown.wait(0.005):
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
            self._fail(error)
