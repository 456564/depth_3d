"""只读匿名 CDN 媒体；DNS固定公网地址、逐跳校验、有界且可取消。"""

import http.client
import io
import ipaddress
import re
import socket
import ssl
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlsplit

IO_TIMEOUT = 2.0
MAX_READ = 1024 * 1024
MAX_MEDIA_SIZE = 4 * 1024**3
QUALITY_OPTIONS = frozenset({"auto", "2160", "1440", "1080", "720"})
_SAFE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_CODEC = re.compile(r"[A-Za-z0-9_.-]{0,40}\Z")
_CDNS = (".douyinvod.com", ".douyincdn.com")
_DNS_SLOT = threading.Lock()


class MediaHTTPError(ValueError):
    """消息只包含固定原因；不带 URL、响应体或原始异常。"""


def validate_url(value: Any) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= 4096 or any(not 32 < ord(c) < 127 for c in value):
        raise MediaHTTPError("媒体地址格式无效")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        valid = (
            parsed.scheme == "https"
            and parsed.port in {None, 443}
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and re.fullmatch(r"[a-z0-9.-]+", host) is not None
            and any(host.endswith(suffix) for suffix in _CDNS)
        )
    except ValueError:
        valid = False
    if not valid:
        raise MediaHTTPError("媒体地址不在允许的 HTTPS CDN 范围")
    return value


def validate_source(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("kind"), str)
        or value.get("kind") not in {"vod", "live"}
        or not isinstance(value.get("id"), str)
        or _SAFE_ID.fullmatch(value["id"]) is None
        or not isinstance(value.get("renditions"), list)
        or not 1 <= len(value["renditions"]) <= 8
    ):
        raise MediaHTTPError("媒体来源描述无效")
    renditions, ids = [], set()
    for item in value["renditions"]:
        if not isinstance(item, dict):
            raise MediaHTTPError("媒体档位无效")
        identifier, codec = item.get("id"), item.get("codec")
        width, height, bitrate = (item.get(k, -1) for k in ("width", "height", "bitrate"))
        urls = item.get("urls")
        if (
            not isinstance(identifier, str)
            or _SAFE_ID.fullmatch(identifier) is None
            or identifier in ids
            or not isinstance(codec, str)
            or _CODEC.fullmatch(codec) is None
            or any(type(size) is not int or not 0 <= size <= 8192 for size in (width, height))
            or not (width > 0 and height > 0 or value["kind"] == "live" and width == height == 0)
            or type(bitrate) is not int
            or not 0 <= bitrate <= 10**9
            or not isinstance(urls, list)
            or not 1 <= len(urls) <= 3
        ):
            raise MediaHTTPError("媒体档位无效")
        ids.add(identifier)
        renditions.append(
            {
                "id": identifier,
                "width": width,
                "height": height,
                "bitrate": bitrate,
                "codec": codec,
                "urls": [validate_url(url) for url in urls],
            }
        )
    return {"kind": value["kind"], "id": value["id"], "renditions": renditions}


def select_renditions(source: dict[str, Any], quality: str) -> list[dict[str, Any]]:
    if quality not in QUALITY_OPTIONS:
        raise ValueError("Invalid browser quality")
    items = list(source["renditions"])
    if source["kind"] == "vod":
        items.sort(key=lambda item: (item["width"] * item["height"], item["bitrate"]), reverse=True)
    if quality != "auto":
        items = [item for item in items if min(item["width"], item["height"]) <= int(quality)]
    return items


def resolve_public(host: str, cancelled: Callable[[], bool]) -> list[tuple[Any, ...]]:
    # 系统 DNS 不能安全强杀；迟到结果只写局部槽，绝不自行建立连接。
    deadline = time.monotonic() + IO_TIMEOUT
    while not _DNS_SLOT.acquire(timeout=0.02):
        if cancelled() or time.monotonic() >= deadline:
            raise MediaHTTPError("媒体 DNS 繁忙或已取消")
    if cancelled():
        _DNS_SLOT.release()
        raise MediaHTTPError("媒体 DNS 已取消")
    done = threading.Event()
    result: list[Any] = []

    def resolve() -> None:
        try:
            result.append(socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM))
        except OSError:
            pass
        finally:
            _DNS_SLOT.release()
            done.set()

    threading.Thread(target=resolve, daemon=True, name="iw3-media-dns").start()
    while not done.wait(0.02):
        if cancelled() or time.monotonic() >= deadline:
            raise MediaHTTPError("媒体 DNS 超时或已取消")
    if cancelled() or not result or not result[0]:
        raise MediaHTTPError("媒体 DNS 不可用")
    records = result[0]
    if any(not ipaddress.ip_address(record[4][0]).is_global for record in records):
        raise MediaHTTPError("媒体 DNS 包含非公网地址")
    return records[:4]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, cancelled: Callable[[], bool]) -> None:
        self._tls_context = ssl.create_default_context()
        super().__init__(host, timeout=IO_TIMEOUT, context=self._tls_context)
        self.cancelled = cancelled

    def connect(self) -> None:
        for family, kind, proto, _, address in resolve_public(self.host, self.cancelled):
            if self.cancelled():
                raise MediaHTTPError("媒体连接已取消")
            raw = socket.socket(family, kind, proto)
            self.sock = raw
            raw.settimeout(IO_TIMEOUT)
            try:
                raw.connect(address)
                if self.cancelled():
                    raise MediaHTTPError("媒体连接已取消")
                self.sock = self._tls_context.wrap_socket(raw, server_hostname=self.host)
                if self.cancelled():
                    self.sock.close()
                    self.sock = None
                    raise MediaHTTPError("媒体连接已取消")
                return
            except (OSError, ValueError):
                raw.close()
                self.sock = None
        raise MediaHTTPError("媒体 TLS 连接失败")


class MediaHTTPIO(io.RawIOBase):
    def __init__(self, url: str, cancelled: Callable[[], bool], *, live: bool = False) -> None:
        super().__init__()
        self._url = validate_url(url)
        self.cancelled, self.live = cancelled, live
        self._position = 0
        self._size: int | None = None
        self._connection: Any = None
        self._transport_socket: Any = None
        self._response: Any = None
        self._response_position: int | None = None
        self._deadline: float | None = None
        self._abort = False
        self._signature = b""
        self._done = threading.Event()
        self._watcher = threading.Thread(target=self._watch, daemon=True, name="iw3-media-deadline")
        self._watcher.start()

    def _watch(self) -> None:
        while not self._done.wait(0.02):
            if self.cancelled() or self._deadline is not None and time.monotonic() >= self._deadline:
                self._abort = True
                self._disconnect()
                return

    def _check(self) -> None:
        if self.closed or self._abort or self.cancelled():
            raise MediaHTTPError("媒体读取超时或已取消")

    def _disconnect(self) -> None:
        connection, self._connection = self._connection, None
        response, self._response = self._response, None
        self._response_position = None
        transport, self._transport_socket = self._transport_socket, None
        sock = transport if transport is not None else connection.sock if connection is not None else None
        # will_close 会清 connection.sock，但 response.fp 仍持有实际连接。
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if connection is not None:
            connection.close()
        if response is not None:
            response.close()

    def _open(self) -> None:
        self._check()
        current = self._url
        self._deadline = time.monotonic() + IO_TIMEOUT
        try:
            for _ in range(4):
                self._check()
                parsed = urlsplit(validate_url(current))
                assert parsed.hostname is not None
                connection = _PinnedHTTPSConnection(
                    parsed.hostname, lambda: self.closed or self._abort or self.cancelled()
                )
                self._connection = connection
                headers = {
                    "Accept": "*/*",
                    "Referer": "https://live.douyin.com/" if self.live else "https://www.douyin.com/",
                    "Accept-Encoding": "identity",
                }
                if not self.live:
                    headers["Range"] = f"bytes={self._position}-"
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                connection.request("GET", target, headers=headers)
                self._transport_socket = connection.sock
                response = connection.getresponse()
                self._response = response
                self._check()
                if response.status in {301, 302, 303, 307, 308}:
                    next_url = validate_url(urljoin(current, response.getheader("Location", "")))
                    self._disconnect()
                    current = next_url
                    continue
                if response.status not in {200, 206}:
                    raise MediaHTTPError(f"媒体 HTTP {response.status}")
                mime = response.getheader("Content-Type", "").split(";", 1)[0].lower().strip()
                allowed = (
                    {"application/octet-stream", "video/x-flv", "video/flv"}
                    if self.live
                    else {"application/octet-stream", "video/mp4", "application/mp4", "video/quicktime"}
                )
                if mime not in allowed or response.getheader("Content-Encoding", "identity") not in {"", "identity"}:
                    raise MediaHTTPError("媒体响应格式无效")
                if self.live:
                    if response.status != 200:
                        raise MediaHTTPError("直播响应状态无效")
                else:
                    length = response.getheader("Content-Length", "")
                    if len(length) > 12 or not length.isdecimal() or not 0 < int(length) <= MAX_MEDIA_SIZE:
                        raise MediaHTTPError("媒体长度无效")
                    size = int(length)
                    if response.status == 206:
                        match = re.fullmatch(
                            r"bytes (\d{1,12})-(\d{1,12})/(\d{1,12})", response.getheader("Content-Range", "")
                        )
                        if match is None:
                            raise MediaHTTPError("媒体 Range 响应无效")
                        start, end, size = map(int, match.groups())
                        if start != self._position or not start <= end < size or end - start + 1 != int(length):
                            raise MediaHTTPError("媒体 Range 偏移不匹配")
                    elif self._position != 0:
                        raise MediaHTTPError("媒体不支持 Range 跳转")
                    if not 0 < size <= MAX_MEDIA_SIZE or self._size is not None and self._size != size:
                        raise MediaHTTPError("媒体长度发生变化")
                    self._size = size
                self._response_position = self._position
                self._deadline = None
                return
            raise MediaHTTPError("媒体重定向次数过多")
        except MediaHTTPError:
            self._disconnect()
            raise
        except (OSError, ValueError, http.client.HTTPException):
            self._disconnect()
            raise MediaHTTPError("媒体网络请求失败") from None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return not self.live

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check()
        if self.live:
            raise io.UnsupportedOperation("Live media cannot seek")
        if whence == io.SEEK_END:
            if self._size is None:
                self._open()
            assert self._size is not None
            target = self._size + offset
        elif whence == io.SEEK_CUR:
            target = self._position + offset
        elif whence == io.SEEK_SET:
            target = offset
        else:
            raise ValueError("Invalid seek mode")
        if not 0 <= target <= MAX_MEDIA_SIZE:
            raise MediaHTTPError("媒体跳转偏移无效")
        # PyAV 反复跳到末尾查询尺寸后会回原位；等实际读偏移改变再断开 Range 响应。
        self._position = target
        return target

    def read(self, size: int = -1) -> bytes:
        self._check()
        if size == 0 or not self.live and self._size is not None and self._position >= self._size:
            return b""
        if self._response is not None and self._response_position != self._position:
            self._disconnect()
        if self._response is None:
            self._open()
        self._deadline = time.monotonic() + IO_TIMEOUT
        try:
            data = self._response.read1(min(32768 if size < 0 else size, MAX_READ))
            self._check()
            self._position += len(data)
            self._response_position = self._position
            if self.live and len(self._signature) < 3:
                self._signature = (self._signature + data)[:3]
                if self._signature != b"FLV"[: len(self._signature)]:
                    raise MediaHTTPError("直播不是 FLV 压缩流")
            return data
        except MediaHTTPError:
            raise
        except (OSError, ValueError, http.client.HTTPException, AttributeError):
            raise MediaHTTPError("媒体网络读取失败") from None
        finally:
            self._deadline = None

    def close(self) -> None:
        self._done.set()
        self._disconnect()
        super().close()
        if threading.current_thread() is not self._watcher:
            self._watcher.join(timeout=0.2)
