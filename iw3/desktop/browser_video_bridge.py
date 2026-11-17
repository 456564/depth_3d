"""仅接收本机扩展的压缩视频字节与播放时钟，不访问远端媒体地址。"""

import hmac
import json
import math
import re
import secrets
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from .media_video_source import MediaSnapshot

MAX_CHUNK_BYTES = 8 * 1024 * 1024
MAX_MEDIA_BYTES = 256 * 1024 * 1024
STATE_TTL = 1.0
COMMANDS = frozenset({"pause_toggle", "next", "previous", "exit_fullscreen"})
EXTENSION_ORIGIN = re.compile(r"chrome-extension://[a-p]{32}\Z")


class BridgeRequestError(Exception):
    def __init__(self, code: int, message: str) -> None:
        self.code, self.message = code, message


def _number(value: Any, low: float, high: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _identifier(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and 0 < len(value) <= 256 and all(32 < ord(c) < 127 for c in value)


def _validate_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BridgeRequestError(400, "Invalid playback state")
    media_available = value.get("media_available", True)
    dimensions = (value.get("width"), value.get("height"))
    valid_dimensions = all(type(size) is int and 0 < size <= 8192 for size in dimensions) or (
        media_available is False and all(type(size) is int and size == 0 for size in dimensions)
    )
    if (
        not _identifier(value.get("session"))
        or not _identifier(value.get("generation"))
        or type(value.get("serial")) is not int
        or not 0 <= value["serial"] <= 2**53
        or not _number(value.get("media_time"), 0, 604800)
        or type(value.get("paused")) is not bool
        or not _number(value.get("playback_rate"), 0.05, 16)
        or type(media_available) is not bool
        or not valid_dimensions
        or (value.get("duration") is not None and not _number(value["duration"], 0, 604800))
    ):
        raise BridgeRequestError(400, "Invalid playback state")
    validated = {
        k: value.get(k)
        for k in (
            "session",
            "generation",
            "serial",
            "media_time",
            "paused",
            "playback_rate",
            "duration",
            "width",
            "height",
        )
    }
    validated["media_available"] = media_available
    return validated


class BrowserVideoBridge:
    def __init__(self, port: int = 52302) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Invalid loopback bridge port")
        self.port = port
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._token = ""
        self._origin: str | None = None
        self._state: dict[str, Any] | None = None
        self._received_at = 0.0
        self._data = bytearray()
        self._immutable_data = b""
        self._revision = 0
        self._sequence = 0
        self._commands: deque[tuple[str, str, float]] = deque(maxlen=16)

    def start(self) -> None:
        if self._server is not None:
            return
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _reply(self, code: int, value: dict[str, Any], allow_origin: bool = True) -> None:
                body = json.dumps(value, allow_nan=False).encode("utf-8") if code != 204 else b""
                self.send_response(code)
                origin = self.headers.get("Origin", "")
                if allow_origin and owner._origin_allowed(origin):
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _request_body(self, limit: int) -> bytes:
                if self.headers.get("Transfer-Encoding") is not None:
                    raise BridgeRequestError(400, "Chunked requests are not supported")
                length = self.headers.get("Content-Length", "")
                if not length.isdecimal():
                    raise BridgeRequestError(400, "Content-Length required")
                if len(length) > 10:
                    raise BridgeRequestError(413, "Media buffer limit exceeded")
                count = int(length)
                if count > limit:
                    raise BridgeRequestError(413, "Media buffer limit exceeded")
                self.connection.settimeout(2)
                data = self.rfile.read(count)
                if len(data) != count:
                    raise BridgeRequestError(400, "Incomplete request body")
                return data

            def do_OPTIONS(self) -> None:
                origin = self.headers.get("Origin", "")
                if not owner._origin_allowed(origin) or self.headers.get("Host") != f"127.0.0.1:{owner.port}":
                    self._reply(403, {"error": "Extension origin required"}, False)
                    return
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, X-IW3-Token, X-IW3-Session, X-IW3-Generation, X-IW3-Sequence",
                )
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                origin = self.headers.get("Origin", "")
                try:
                    if self.headers.get("Host") != f"127.0.0.1:{owner.port}" or not owner._origin_allowed(origin):
                        raise BridgeRequestError(403, "Extension origin required")
                    if self.path not in {"/hello", "/state", "/chunk", "/media"}:
                        raise BridgeRequestError(404, "Unknown bridge endpoint")
                    if self.path != "/hello":
                        owner._authenticate(origin, self.headers.get("X-IW3-Token", ""))
                    limit = (
                        MAX_CHUNK_BYTES if self.path == "/chunk" else MAX_MEDIA_BYTES if self.path == "/media" else 8192
                    )
                    data = self._request_body(limit)
                    if self.path == "/hello":
                        if data not in {b"", b"{}"}:
                            raise BridgeRequestError(400, "Invalid pairing request")
                        result = owner._hello(origin)
                    elif self.path == "/state":
                        if self.headers.get_content_type() != "application/json":
                            raise BridgeRequestError(400, "JSON state required")
                        try:
                            value = json.loads(data)
                        except (ValueError, UnicodeError):
                            raise BridgeRequestError(400, "Invalid playback state") from None
                        result = owner._receive_state(_validate_state(value))
                    else:
                        if self.headers.get_content_type() != "application/octet-stream":
                            raise BridgeRequestError(400, "Compressed bytes required")
                        session = self.headers.get("X-IW3-Session")
                        generation = self.headers.get("X-IW3-Generation")
                        sequence = self.headers.get("X-IW3-Sequence", "")
                        if (
                            not _identifier(session)
                            or not _identifier(generation)
                            or not sequence.isdecimal()
                            or len(sequence) > 16
                        ):
                            raise BridgeRequestError(400, "Invalid media headers")
                        result = owner._receive_media(session, generation, int(sequence), data, self.path == "/media")
                    self._reply(200, result)
                except BridgeRequestError as error:
                    self._reply(error.code, {"error": error.message}, error.code != 403)
                except (TimeoutError, ConnectionError, OSError):
                    self.close_connection = True

        server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        server.daemon_threads = True
        self.port = server.server_port
        self._token = secrets.token_urlsafe(32)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=3)
        with self._lock:
            self._token, self._origin, self._state = "", None, None
            self._data.clear()
            self._immutable_data = b""
            self._commands.clear()

    def _origin_allowed(self, origin: str) -> bool:
        with self._lock:
            return EXTENSION_ORIGIN.fullmatch(origin) is not None and self._origin in {None, origin}

    def _hello(self, origin: str) -> dict[str, Any]:
        with self._lock:
            if self._origin not in {None, origin} or not self._token:
                raise BridgeRequestError(403, "Extension pairing rejected")
            self._origin = origin
            return {"token": self._token}

    def _authenticate(self, origin: str, token: str) -> None:
        with self._lock:
            if (
                origin != self._origin
                or not self._token
                or not token.isascii()
                or not hmac.compare_digest(token, self._token)
            ):
                raise BridgeRequestError(403, "Extension authentication required")

    def _receive_state(self, value: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._server is None:
                raise BridgeRequestError(503, "Bridge stopped")
            previous = self._state
            if previous is not None and previous["session"] == value["session"]:
                if value["serial"] < previous["serial"] or (
                    value["serial"] == previous["serial"] and value != previous
                ):
                    raise BridgeRequestError(409, "Stale playback state")
            # 页面心跳与媒体能力分开；能力切换同样废弃旧字节与待发命令。
            changed = previous is None or (
                previous["session"],
                previous["generation"],
                previous["media_available"],
            ) != (
                value["session"],
                value["generation"],
                value["media_available"],
            )
            if changed:
                self._data.clear()
                self._immutable_data = b""
                self._revision += 1
                self._sequence = 0
                self._commands.clear()
            now = monotonic()
            self._state, self._received_at = value, now
            commands = [
                {"name": name, "generation": generation}
                for name, generation, queued_at in self._commands
                if generation == value["generation"]
                and now - queued_at <= STATE_TTL
                and (name != "pause_toggle" or value["media_available"])
            ]
            self._commands.clear()
            return {
                "commands": commands,
                "need_media": value["media_available"] and not self._data,
                "next_sequence": self._sequence,
            }

    def _receive_media(
        self, session: str, generation: str, sequence: int, data: bytes, replace: bool
    ) -> dict[str, Any]:
        with self._lock:
            if self._server is None:
                raise BridgeRequestError(503, "Bridge stopped")
            state = self._state
            if (
                state is None
                or not state["media_available"]
                or monotonic() - self._received_at > STATE_TTL
                or (session, generation) != (state["session"], state["generation"])
                or sequence != self._sequence
            ):
                raise BridgeRequestError(409, "Stale or out-of-order compressed media")
            total = len(data) if replace else len(self._data) + len(data)
            if not data or total > MAX_MEDIA_BYTES:
                raise BridgeRequestError(413, "Media buffer limit exceeded")
            if replace:
                self._data = bytearray(data)
            else:
                self._data.extend(data)
            self._immutable_data = b""
            self._revision += 1
            self._sequence += 1
            return {"next_sequence": self._sequence, "bytes": len(self._data)}

    def snapshot(self) -> "MediaSnapshot | None":
        from .media_video_source import MediaSnapshot

        with self._lock:
            state = self._state
            if (
                state is None
                or not state["media_available"]
                or not self._data
                or monotonic() - self._received_at > STATE_TTL
            ):
                return None
            if not self._immutable_data:
                self._immutable_data = bytes(self._data)
            return MediaSnapshot(
                generation=state["generation"],
                data=self._immutable_data,
                revision=self._revision,
                media_time=state["media_time"],
                paused=state["paused"],
                playback_rate=state["playback_rate"],
                received_at=self._received_at,
                duration=state["duration"],
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            fresh = state is not None and monotonic() - self._received_at <= STATE_TTL
            media_available = state is not None and state["media_available"]
            status = "source" if fresh and media_available and self._data else "waiting"
            return {
                "state": status,
                "status": status,
                "fresh": fresh,
                "media_available": media_available,
                "generation": state["generation"] if state is not None else None,
                "width": state["width"] if state is not None else 0,
                "height": state["height"] if state is not None else 0,
                "media_time": state["media_time"] if state is not None else 0.0,
                "paused": state["paused"] if state is not None else True,
                "bytes": len(self._data),
                "revision": self._revision,
                "message": "压缩视频源已就绪" if status == "source" else "等待浏览器压缩视频源；不会回退录屏",
            }

    def queue_command(self, name: str, generation: str | None = None) -> bool:
        with self._lock:
            state = self._state
            now = monotonic()
            if (
                name not in COMMANDS
                or state is None
                or (name == "pause_toggle" and not state["media_available"])
                or now - self._received_at > STATE_TTL
                or (generation is not None and generation != state["generation"])
            ):
                return False
            self._commands.append((name, state["generation"], now))
            return True
