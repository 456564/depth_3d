"""原媒体来源：边界校验、档位顺序、Range读取和可取消网络。"""

import importlib
import importlib.util
import io
import threading
import time

import pytest


def module():
    assert importlib.util.find_spec("iw3.desktop.media_http_source") is not None, "缺少受限媒体读取模块"
    return importlib.import_module("iw3.desktop.media_http_source")


def descriptor(kind="vod"):
    return {
        "kind": kind,
        "id": "media:one",
        "renditions": [
            {
                "id": "1080",
                "width": 1080,
                "height": 1920,
                "bitrate": 2000,
                "codec": "h264",
                "urls": ["https://media.douyinvod.com/video.mp4?private=one"],
            },
            {
                "id": "1440",
                "width": 1440,
                "height": 2560,
                "bitrate": 4000,
                "codec": "hevc",
                "urls": ["https://media.douyinvod.com/high.mp4?private=two"],
            },
        ],
    }


def test_source_schema_preserves_only_bounded_valid_descriptor():
    actual = module().validate_source(descriptor())
    assert actual == descriptor()
    assert actual is not descriptor()


@pytest.mark.parametrize(
    "url",
    [
        "http://media.douyinvod.com/x",
        "https://127.0.0.1/x",
        "https://[::1]/x",
        "https://media.douyinvod.com.evil.test/x",
        "https://user@media.douyinvod.com/x",
        "https://media.douyinvod.com:444/x",
        "https://media.douyinvod.com/x#fragment",
        "https://media.douyinvod.com/x\n",
        "https://media.douyinvod.com/" + "x" * 4096,
    ],
)
def test_source_rejects_unsafe_urls_without_returning_them(url):
    value = descriptor()
    value["renditions"][0]["urls"] = [url]
    with pytest.raises(ValueError) as error:
        module().validate_source(value)
    assert url not in str(error.value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(kind="webrtc"),
        lambda value: value.update(kind=[]),
        lambda value: value.update(id="https://private.test/"),
        lambda value: value.update(renditions=value["renditions"] * 5),
        lambda value: value["renditions"][0].update(width=True),
        lambda value: value["renditions"][0].update(width=0),
        lambda value: value["renditions"][0].update(urls=["https://media.douyinvod.com/x"] * 4),
        lambda value: value["renditions"][0].update(codec="https://secret.test/"),
    ],
)
def test_source_rejects_invalid_dimensions_ids_and_unbounded_candidates(mutate):
    value = descriptor()
    mutate(value)
    with pytest.raises(ValueError):
        module().validate_source(value)


def test_vod_sorts_real_pixels_and_caps_short_side_but_live_preserves_priority():
    reader = module()
    assert [r["id"] for r in reader.select_renditions(descriptor(), "auto")] == ["1440", "1080"]
    assert [r["id"] for r in reader.select_renditions(descriptor(), "1080")] == ["1080"]
    assert reader.select_renditions(descriptor(), "720") == []
    live = descriptor("live")
    live["renditions"][0].update(width=0, height=0, bitrate=0)
    assert reader.validate_source(live) == live
    assert [r["id"] for r in reader.select_renditions(live, "auto")] == ["1080", "1440"]


def test_public_dns_rejects_mixed_private_and_public_answers(monkeypatch):
    reader = module()
    monkeypatch.setattr(
        reader.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("8.8.8.8", 443)), (2, 1, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(reader.MediaHTTPError):
        reader.resolve_public("media.douyinvod.com", lambda: False)


def test_blocked_dns_can_be_cancelled_without_opening_a_late_connection(monkeypatch):
    reader = module()
    release, cancel = threading.Event(), threading.Event()
    monkeypatch.setattr(reader.socket, "getaddrinfo", lambda *args, **kwargs: release.wait(3))
    timer = threading.Timer(0.05, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(reader.MediaHTTPError):
            reader.resolve_public("media.douyinvod.com", cancel.is_set)
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        timer.join()


def test_cancelled_dns_never_accumulates_one_new_resolver_per_video(monkeypatch):
    reader = module()
    release, cancel = threading.Event(), threading.Event()
    calls = []

    def stalled_dns(*args, **kwargs):
        calls.append(1)
        release.wait(3)
        return []

    monkeypatch.setattr(reader.socket, "getaddrinfo", stalled_dns)
    monkeypatch.setattr(reader, "IO_TIMEOUT", 0.06)
    timer = threading.Timer(0.02, cancel.set)
    timer.start()
    try:
        with pytest.raises(reader.MediaHTTPError):
            reader.resolve_public("media.douyinvod.com", cancel.is_set)
        with pytest.raises(reader.MediaHTTPError):
            reader.resolve_public("other.douyinvod.com", lambda: False)
        assert len(calls) == 1
    finally:
        release.set()
        timer.join()


class Response:
    def __init__(self, data, status=206, start=0, size=None, mime="video/mp4", location=None):
        self.data = io.BytesIO(data)
        self.status = status
        self.headers = {"Content-Type": mime, "Content-Length": str(len(data))}
        if status == 206:
            self.headers["Content-Range"] = f"bytes {start}-{start + len(data) - 1}/{size or len(data)}"
        if location:
            self.headers["Location"] = location
        self.closed = False

    def getheader(self, key, default=None):
        return self.headers.get(key, default)

    def read1(self, size):
        return self.data.read(size)

    def close(self):
        self.closed = True


def fake_transport(monkeypatch, responses):
    reader = module()
    calls = []

    class Connection:
        def __init__(self, host, cancelled):
            self.host, self.sock = host, None

        def request(self, method, target, headers):
            calls.append((self.host, target, headers))

        def getresponse(self):
            return responses.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(reader, "_PinnedHTTPSConnection", Connection)
    return reader, calls


def test_range_reader_seeks_without_whole_file_download_and_checks_content_range(monkeypatch):
    reader, calls = fake_transport(monkeypatch, [Response(b"abcdefgh", size=8), Response(b"fgh", start=5, size=8)])
    source = reader.MediaHTTPIO("https://media.douyinvod.com/x", lambda: False)
    try:
        assert source.read(3) == b"abc"
        assert source.seek(5) == 5
        assert source.read(3) == b"fgh"
        assert source.seek(0, io.SEEK_END) == 8
        assert calls[1][2]["Range"] == "bytes=5-"
    finally:
        source.close()


def test_initial_size_probe_keeps_unread_response_for_return_to_start(monkeypatch):
    response = Response(b"abcdefgh")
    reader, calls = fake_transport(monkeypatch, [response])
    source = reader.MediaHTTPIO("https://media.douyinvod.com/x", lambda: False)
    try:
        assert source.seek(-1, io.SEEK_END) == 7
        assert source.tell() == 7
        assert source.seek(0) == 0
        assert not response.closed
        assert source.read(3) == b"abc"
        assert len(calls) == 1
        assert response.data.tell() == 3
    finally:
        source.close()


def test_repeated_size_probes_preserve_response_at_actual_read_position(monkeypatch):
    response = Response(b"abcdefgh")
    reader, calls = fake_transport(monkeypatch, [response])
    source = reader.MediaHTTPIO("https://media.douyinvod.com/x", lambda: False)
    try:
        assert source.read(3) == b"abc"
        for _ in range(3):
            assert source.seek(-1, io.SEEK_END) == 7
            assert source.seek(3) == 3
        assert not response.closed
        assert source.read(3) == b"def"
        assert len(calls) == 1
        assert response.data.tell() == 6
    finally:
        source.close()


def test_changed_read_offset_reopens_only_when_bytes_are_requested(monkeypatch):
    response = Response(b"abcdefgh")
    reader, calls = fake_transport(monkeypatch, [response, Response(b"fgh", start=5, size=8)])
    source = reader.MediaHTTPIO("https://media.douyinvod.com/x", lambda: False)
    try:
        assert source.read(3) == b"abc"
        assert source.seek(5) == 5
        assert source.read(0) == b""
        assert not response.closed
        assert len(calls) == 1
        assert source.read(2) == b"fg"
        assert response.closed
        assert len(calls) == 2
        assert calls[1][2]["Range"] == "bytes=5-"
        assert source.read(1) == b"h"
        assert source.tell() == 8
    finally:
        source.close()


def test_redirect_revalidates_cdn_and_never_contacts_private_destination(monkeypatch):
    reader, calls = fake_transport(monkeypatch, [Response(b"", status=302, location="https://127.0.0.1/secret")])
    source = reader.MediaHTTPIO("https://media.douyinvod.com/x", lambda: False)
    with pytest.raises(reader.MediaHTTPError):
        source.read(3)
    source.close()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        Response(b"abc", status=200),
        Response(b"abc", start=0, size=8),
        Response(b"abc", start=5, size=8, mime="text/html"),
    ],
)
def test_range_offset_and_response_mime_are_checked(monkeypatch, response):
    reader, _ = fake_transport(monkeypatch, [response])
    source = reader.MediaHTTPIO("https://media.douyinvod.com/x", lambda: False)
    source.seek(5)
    with pytest.raises(reader.MediaHTTPError):
        source.read(3)
    source.close()


def test_live_reader_is_nonseekable_requires_flv_and_keeps_only_requested_bytes(monkeypatch):
    reader, calls = fake_transport(monkeypatch, [Response(b"FLV" + b"a" * 100, status=200, mime="video/x-flv")])
    source = reader.MediaHTTPIO("https://pull.douyincdn.com/live.flv", lambda: False, live=True)
    try:
        assert not source.seekable()
        assert source.read(3) == b"FLV"
        assert source.read(4) == b"aaaa"
        assert "Range" not in calls[0][2]
        with pytest.raises(OSError):
            source.seek(0)
    finally:
        source.close()


def test_live_reader_rejects_non_flv_media(monkeypatch):
    reader, _ = fake_transport(monkeypatch, [Response(b"not flv", status=200, mime="video/x-flv")])
    source = reader.MediaHTTPIO("https://pull.douyincdn.com/live.flv", lambda: False, live=True)
    with pytest.raises(reader.MediaHTTPError):
        source.read(10)
    source.close()


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_cancel_interrupts_blocked_headers_and_body_and_leaves_no_reader_thread(monkeypatch, phase):
    reader = module()
    entered, release, cancel = threading.Event(), threading.Event(), threading.Event()

    class BlockingResponse(Response):
        def read1(self, size):
            entered.set()
            release.wait(3)
            raise OSError("private address must never appear")

        def close(self):
            release.set()

    class BlockingConnection:
        sock = None

        def __init__(self, *args):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            if phase == "headers":
                entered.set()
                release.wait(3)
            return BlockingResponse(b"abc")

        def close(self):
            release.set()

    monkeypatch.setattr(reader, "_PinnedHTTPSConnection", BlockingConnection)
    stream = reader.MediaHTTPIO("https://media.douyinvod.com/x", cancel.is_set)
    failures = []

    def consume():
        try:
            stream.read(3)
        except reader.MediaHTTPError as error:
            failures.append(str(error))

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert entered.wait(1)
        started = time.monotonic()
        cancel.set()
        worker.join(0.5)
        assert not worker.is_alive() and time.monotonic() - started < 0.5
        assert failures and "private" not in str(failures)
    finally:
        stream.close()
        release.set()
        worker.join(1)
    assert not stream._watcher.is_alive()


def test_absolute_header_deadline_stops_even_when_socket_timeout_would_be_reset(monkeypatch):
    reader = module()
    release = threading.Event()

    class SlowHeaders:
        sock = None

        def __init__(self, *args):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            release.wait(3)
            return Response(b"abc")

        def close(self):
            release.set()

    monkeypatch.setattr(reader, "IO_TIMEOUT", 0.06)
    monkeypatch.setattr(reader, "_PinnedHTTPSConnection", SlowHeaders)
    source = reader.MediaHTTPIO("https://media.douyinvod.com/x", lambda: False)
    started = time.monotonic()
    try:
        with pytest.raises(reader.MediaHTTPError):
            source.read(3)
        assert time.monotonic() - started < 0.5
    finally:
        source.close()


def test_media_owner_stop_cancels_network_before_joining_decoder(monkeypatch):
    import torch

    from iw3.desktop import media_video_source as video

    reader = module()
    monkeypatch.setattr(video, "media_now", time.monotonic)
    entered, release = threading.Event(), threading.Event()

    class BlockingResponse(Response):
        def read1(self, size):
            entered.set()
            release.wait(3)
            raise OSError("interrupted")

        def close(self):
            release.set()

    class Connection:
        sock = None

        def __init__(self, *args):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return BlockingResponse(b"abc")

        def close(self):
            release.set()

    class ProbeReader:
        def __init__(self, source, *args, **kwargs):
            source.read(3)
            raise ValueError("no complete media")

    monkeypatch.setattr(reader, "_PinnedHTTPSConnection", Connection)
    monkeypatch.setattr(video, "_NVDECReader", ProbeReader)
    source = video.MediaVideoSource(
        lambda: video.MediaSnapshot(generation="active", source=descriptor(), received_at=time.monotonic()),
        torch.device("cuda:0"),
    )
    source.start()
    try:
        assert entered.wait(1)
        started = time.monotonic()
        source.stop()
        assert not source.is_alive() and time.monotonic() - started < 0.5
        assert not source.get_status()["fallback_required"]
    finally:
        release.set()
        source.stop()


def test_will_close_response_still_shuts_down_actual_transport_before_buffered_close(monkeypatch):
    import http.client

    reader = module()
    release, cancel, entered = threading.Event(), threading.Event(), threading.Event()

    class Trickle(io.RawIOBase):
        def __init__(self):
            self.header = bytearray(
                b"HTTP/1.1 200 OK\r\nContent-Type: video/x-flv\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )

        def readable(self):
            return True

        def readinto(self, buffer):
            if self.header:
                amount = min(len(buffer), len(self.header))
                buffer[:amount] = self.header[:amount]
                del self.header[:amount]
                return amount
            entered.set()
            if release.wait(0.01):
                return 0
            buffer[0] = 65
            return 1

    class Socket:
        def makefile(self, *args, **kwargs):
            return io.BufferedReader(Trickle())

        def shutdown(self, how):
            release.set()

    transport = Socket()
    response = http.client.HTTPResponse(transport)
    response.begin()
    assert response.will_close

    class Connection:
        def __init__(self, *args):
            self.sock = transport

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            self.sock = None
            return response

        def close(self):
            pass

    monkeypatch.setattr(reader, "IO_TIMEOUT", 0.06)
    monkeypatch.setattr(reader, "_PinnedHTTPSConnection", Connection)
    source = reader.MediaHTTPIO("https://pull.douyincdn.com/live", cancel.is_set, live=True)
    failures = []

    def consume():
        try:
            source.read(3)
        except reader.MediaHTTPError:
            failures.append(True)

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert entered.wait(1)
        cancel.set()
        worker.join(0.3)
        assert not worker.is_alive(), "will_close不能让慢速chunk header绕过取消"
        source.close()
        assert not source._watcher.is_alive()
        assert failures
    finally:
        release.set()
        worker.join(1)
        source.close()


@pytest.mark.hwaccel
@pytest.mark.parametrize("live", [False, True])
def test_real_http_filelike_to_strict_nvdec_keeps_rgb_and_supports_stream_or_seek(monkeypatch, live):
    from fractions import Fraction

    import av
    import torch

    from iw3.desktop.media_video_source import _NVDECReader
    from nunif.utils.video import pyav_init_cuda_primary_context

    pyav_init_cuda_primary_context(max_devices=1)
    output = io.BytesIO()
    with av.open(output, mode="w", format="flv" if live else "mp4") as container:
        encoder = container.add_stream("libx264", rate=30)
        encoder.width, encoder.height, encoder.pix_fmt = 96, 64, "yuv420p"
        encoder.options = {"crf": "12", "g": "2", "bf": "0"}
        for index in range(20):
            frame = av.VideoFrame(96, 64, "rgb24")
            frame.planes[0].update(bytes((index * 10, 50, 180)) * (96 * 64))
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in encoder.encode(frame):
                container.mux(packet)
        for packet in encoder.encode():
            container.mux(packet)
    data, calls = output.getvalue(), []
    network = module()

    class Connection:
        sock = None

        def __init__(self, host, cancelled):
            self.start = 0

        def request(self, method, target, headers):
            self.start = int(headers["Range"].split("=")[1].split("-")[0]) if "Range" in headers else 0
            calls.append(self.start)

        def getresponse(self):
            return Response(
                data[self.start :],
                status=200 if live else 206,
                start=self.start,
                size=len(data),
                mime="video/x-flv" if live else "video/mp4",
            )

        def close(self):
            pass

    monkeypatch.setattr(network, "_PinnedHTTPSConnection", Connection)
    stream = network.MediaHTTPIO("https://media.douyincdn.com/sample", lambda: False, live=live)
    decoder = _NVDECReader(stream, torch.device("cuda:0"), live=live)
    try:
        with torch.inference_mode():
            for index, target in enumerate([0, 0.2, 0.1, 0.3]):
                rgb, timestamp = decoder.next_frame() if live else decoder.frame_at(target)
                assert rgb.shape == (3, 64, 96) and rgb.dtype == torch.float32 and rgb.is_cuda
                assert torch.isfinite(rgb).all() and 0 <= rgb.min() <= rgb.max() <= 1
                assert timestamp == pytest.approx(index / 30 if live else target, abs=1 / 30)
                assert float(rgb[0].mean()) * 255 == pytest.approx(index * 10 if live else target * 300, abs=8)
        assert decoder.stream.codec_context.is_hwaccel
        assert calls and (not live or calls == [0])
    finally:
        decoder.close()
        stream.close()
