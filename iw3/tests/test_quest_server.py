import importlib
import json
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch


def read_exact(client: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = client.recv(size - len(data))
        if not part:
            raise ConnectionError("unexpected EOF")
        data.extend(part)
    return bytes(data)


def read_message(client: socket.socket) -> tuple[int, bytes]:
    header = read_exact(client, 5)
    return header[0], read_exact(client, int.from_bytes(header[1:], "big"))


def send_json(client: socket.socket, kind: int, value: Any) -> None:
    payload = json.dumps(value).encode()
    client.sendall(struct.pack("!BI", kind, len(payload)) + payload)


class EncodedPacket:
    def __init__(self, payload: bytes, pts: int, keyframe: bool) -> None:
        self.payload, self.pts, self.is_keyframe = payload, pts, keyframe

    def __bytes__(self) -> bytes:
        return self.payload


class EncoderDouble:
    def __init__(self, width: int, height: int, fps: int, bitrate: int) -> None:
        pass

    def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False) -> list[EncodedPacket]:
        return [EncodedPacket(bytes([round(float(frame[0, 0, 0]) * 255)]), pts, force_idr)]


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = importlib.import_module("iw3.desktop.quest_server")
    monkeypatch.setattr(module, "QuestEncoder", EncoderDouble)
    return module


@pytest.fixture
def server(module: Any) -> Iterator[Any]:
    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def connect(server: Any) -> socket.socket:
    client = socket.create_connection(("127.0.0.1", server.port), timeout=1)
    client.settimeout(1)
    return client


def initial_messages(client: socket.socket) -> dict[str, Any]:
    kind, payload = read_message(client)
    assert kind == 0x01
    assert json.loads(payload) == {"largura": 128, "altura": 64, "fps": 30, "codec": "hevc"}
    kind, payload = read_message(client)
    assert kind == 0x11
    return json.loads(payload)


@pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8", "224.0.0.1", "255.255.255.255", "::1", "localhost"])
def test_reject_non_private_ipv4_bind(module: Any, host: str) -> None:
    with pytest.raises(ValueError):
        module.QuestServer(host, 128, 64, 30, 35, audio=False)


@pytest.mark.parametrize(
    "width,height,fps,bitrate", [(126, 64, 30, 35), (128, 63, 30, 35), (128, 64, 0, 35), (128, 64, 30, 101)]
)
def test_reject_bad_video_geometry(module: Any, width: int, height: int, fps: int, bitrate: int) -> None:
    with pytest.raises(ValueError):
        module.QuestServer("127.0.0.1", width, height, fps, bitrate, audio=False)


def fullscreen_action_server(module: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    target = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False)
    target._shutdown.clear()
    target._connection = object()
    target._last_input = 100.0
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    seen: list[str] = []

    def exit_fullscreen() -> bool:
        seen.append("escape")
        return True

    target._controller = SimpleNamespace(exit_fullscreen=exit_fullscreen, reset=lambda: None)
    target._menu = SimpleNamespace(visible=True)
    target._menu.reset = lambda: setattr(target._menu, "visible", False)
    return target, seen


def test_exit_fullscreen_uses_only_the_current_focused_menu_controller(module: Any, monkeypatch) -> None:
    target, seen = fullscreen_action_server(module, monkeypatch)
    assert target.exit_fullscreen()
    assert seen == ["escape"]


@pytest.mark.parametrize(
    "blocked",
    ["connection", "controller", "menu", "focus", "missing_input", "stale", "hidden", "shutdown", "paused_video"],
)
def test_exit_fullscreen_rejects_inactive_input_without_sending(module: Any, monkeypatch, blocked: str) -> None:
    target, seen = fullscreen_action_server(module, monkeypatch)
    if blocked in {"connection", "controller", "menu"}:
        setattr(target, "_" + blocked, None)
    elif blocked == "focus":
        target._menu.visible = False
    elif blocked == "missing_input":
        target._last_input = None
    elif blocked == "stale":
        target._last_input = 99.5
    elif blocked == "hidden":
        target._suspended = True
    elif blocked == "shutdown":
        target._shutdown.set()
    else:
        target._video_enabled = False
    assert not target.exit_fullscreen()
    assert seen == [] and target._error is None and target._media_error is None


def test_fullscreen_command_failure_does_not_fail_the_media_service(module: Any, monkeypatch) -> None:
    target, _ = fullscreen_action_server(module, monkeypatch)

    def reject():
        raise OSError("Windows rejected the Quest key tap")

    target._controller.exit_fullscreen = reject
    with pytest.raises(RuntimeError, match="Windows rejected"):
        target.exit_fullscreen()
    assert not target._shutdown.is_set() and target._error is None and target._media_error is None


def test_reconnection_does_not_reuse_the_previous_fullscreen_click_authorization(module: Any, monkeypatch) -> None:
    target, seen = fullscreen_action_server(module, monkeypatch)
    old, peer = socket.socketpair()
    with old, peer:
        target._connection = old
        target._disconnect(old)
    target._connection = object()

    def exit_fullscreen() -> bool:
        seen.append("new escape")
        return True

    target._controller = SimpleNamespace(exit_fullscreen=exit_fullscreen)
    assert not target.exit_fullscreen()
    assert seen == [] and target._last_input is None and not target._menu.visible


def test_tcp_menu_exit_reaches_guarded_windows_input_and_failure_keeps_media_alive(module: Any, monkeypatch) -> None:
    from iw3.desktop.quest_input import QuestController
    from iw3.desktop.quest_menu import QuestMenu

    from .test_quest_menu import packet, point
    from .test_quest_session import make_session
    from .test_quest_windows import backend, patch_windows_api

    api = patch_windows_api(monkeypatch)
    control = QuestController(backend())
    owner = make_session()
    target = module.QuestServer(
        "127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0, controller_factory=lambda: control
    )
    owner.server = target
    menu = QuestMenu(owner.snapshot, owner.submit)
    target.set_menu(menu)

    def wait_for(predicate):
        deadline = time.monotonic() + 1
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(0.01)

    target.start()
    try:
        with connect(target) as client:
            initial_messages(client)
            send_json(client, 0x83, packet())
            send_json(client, 0x83, packet(b=True))
            wait_for(lambda: menu.visible)
            with menu._lock:
                menu._act("page", 4)
                menu._render()
                rect = next(r for r, action, value, _ in menu._buttons if value == "exit_fullscreen")
                pos = point(menu, (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
            send_json(client, 0x83, packet(pos=pos))
            send_json(client, 0x83, packet(pos=pos, gatilho=True))
            wait_for(lambda: len(api.batches) == 1)
            assert api.batches == [[(1, 0, 0x1B), (1, 2, 0x1B)]]
            assert menu.visible and not owner.run_stop.is_set()
            api.foreground = 9
            send_json(client, 0x83, packet(pos=pos))
            send_json(client, 0x83, packet(pos=pos, gatilho=True))
            wait_for(lambda: "退出未发送" in menu._message)
            assert len(api.batches) == 1 and target._connection is not None
            target.check_error()
            target.set_frame_data((torch.ones(3, 64, 128), 1.0))
            while read_message(client)[0] != 0x02:
                pass
            assert not owner.run_stop.is_set() and target._media_error is None
    finally:
        target.stop()


def test_discovery_exact_wire_format(server: Any) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(0.1)
        client.sendto(b"unrelated", ("127.0.0.1", server.discovery_port))
        with pytest.raises(socket.timeout):
            client.recvfrom(1024)
        client.sendto(b"ESTUDIO3D?", ("127.0.0.1", server.discovery_port))
        reply, _ = client.recvfrom(1024)
        assert reply.startswith(b"ESTUDIO3D!")
        assert json.loads(reply[10:]) == {"nome": "iw3", "porta": server.port}


def test_discovery_advertises_selected_adapter(module: Any) -> None:
    server = module.QuestServer("127.0.0.2", 128, 64, 30, 35, audio=False, port=0, discovery_port=0)
    server.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(1)
            client.sendto(b"ESTUDIO3D?", ("127.0.0.1", server.discovery_port))
            reply, peer = client.recvfrom(1024)
            assert peer[0] == "127.0.0.2"
            assert json.loads(reply[10:])["porta"] == server.port
    finally:
        server.stop()


def test_restart_with_same_ports_after_server_closed_active_connection(server: Any) -> None:
    with connect(server) as first:
        initial_messages(first)
        ports = server.port, server.discovery_port
        server.stop()
        server.start()
        assert (server.port, server.discovery_port) == ports
        with connect(server) as second:
            initial_messages(second)


def test_full_sbs_uses_identical_world_planes_and_first_idr(server: Any) -> None:
    with connect(server) as client:
        layers = initial_messages(client)["camadas"]
        left, right = layers
        assert left["tex"] == right["tex"] == "video"
        assert left["olho"] == "esq" and right["olho"] == "dir"
        assert left["rect"] == [0, 0, 64, 64] and right["rect"] == [64, 0, 64, 64]
        assert left["pos"] == right["pos"] == [0, 0, -3]
        assert left["rot"] == right["rot"] == [0, 0, 0, 1]
        assert left["tam"] == right["tam"] == [3.6, 3.6]
        server.set_frame_data((torch.full((3, 64, 128), 1.0), 1.0))
        kind, payload = read_message(client)
        assert kind == 0x02 and payload[:9] == struct.pack("!QB", 1_000_000, 1)
        assert payload[9:] == b"\xff"


def test_fragmented_pose_recenter_and_idr(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        pose: dict[str, Any] = {"cabeca": {"pos": [1, 2, 3], "rot": [0, 0, 0, 1]}, "recentrou": False}
        payload = json.dumps(pose).encode()
        packet = struct.pack("!BI", 0x83, len(payload)) + payload
        client.sendall(packet[:2])
        client.sendall(packet[2:8])
        client.sendall(packet[8:])
        kind, payload = read_message(client)
        layers = json.loads(payload)["camadas"]
        assert kind == 0x11 and layers[0]["pos"] == layers[1]["pos"] == [1, 2, 0]
        server.set_frame_data((torch.zeros(3, 64, 128), 2.0))
        assert read_message(client)[1][8] == 1
        server.set_frame_data((torch.zeros(3, 64, 128), 2.0))
        kind, payload = read_message(client)
        assert kind == 0x02 and payload[8] == 0
        assert struct.unpack("!Q", payload[:8])[0] == 2_000_001
        client.sendall(b"\x82\x00\x00\x00\x00")
        pose["cabeca"]["pos"] = [4, 5, 6]
        pose["recentrou"] = True
        send_json(client, 0x83, pose)
        layers = json.loads(read_message(client)[1])["camadas"]
        assert layers[0]["pos"] == layers[1]["pos"] == [4, 5, 3]
        server.set_frame_data((torch.zeros(3, 64, 128), 3.0))
        assert read_message(client)[1][8] == 1


@pytest.mark.parametrize(
    "bad_pose",
    [
        None,
        [],
        {"pos": [float("nan"), 0, 0], "rot": [0, 0, 0, 1]},
        {"pos": [0, 0, 0], "rot": [0, 0, 0, 0]},
        {"pos": [10**400, 0, 0], "rot": [0, 0, 0, 1]},
    ],
)
def test_invalid_pose_does_not_crash_or_move_screen(server: Any, bad_pose: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, {"cabeca": bad_pose, "recentrou": True})
        client.sendall(b"\x83\x00\x00\x00\x01{")
        server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
        assert read_message(client)[0] == 0x02
        server.check_error()


def test_oversized_input_and_partial_eof_allow_reconnect(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        client.sendall(b"\x83\x00\x01\x00\x01")
        assert client.recv(1) == b""
    with connect(server) as client:
        initial_messages(client)
        client.sendall(b"\x83\x00")
        client.shutdown(socket.SHUT_WR)
        assert client.recv(1) == b""
    with connect(server) as client:
        initial_messages(client)
        server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
        assert read_message(client)[1][8] == 1
    server.check_error()


def test_yaw_anchors_screen_ahead_in_local_space(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, {"cabeca": {"pos": [1, 2, 3], "rot": [0, 0.7071067811865476, 0, 0.7071067811865476]}})
        kind, payload = read_message(client)
        layers = json.loads(payload)["camadas"]
        assert kind == 0x11
        assert layers[0]["pos"] == pytest.approx([-2, 2, 3])
        assert layers[0]["rot"] == pytest.approx([0, 0.7071067811865476, 0, 0.7071067811865476])
        assert layers[0]["pos"] == layers[1]["pos"]
        assert layers[0]["rot"] == layers[1]["rot"]


def test_existing_listener_is_preserved_on_port_conflict(module: Any, server: Any) -> None:
    conflicting = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=server.port, discovery_port=0)
    with pytest.raises(RuntimeError, match="Pop3D"):
        conflicting.start()
    conflicting.stop()
    with connect(server) as client:
        initial_messages(client)


def test_latest_raw_frame_replaces_stale_and_owns_copy(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()

    class SlowEncoder(EncoderDouble):
        def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False) -> list[EncodedPacket]:
            if float(frame[0, 0, 0]) == 0:
                entered.set()
                assert release.wait(2)
            return super().encode(frame, pts, force_idr)

    monkeypatch.setattr(module, "QuestEncoder", SlowEncoder)
    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0)
    server.start()
    try:
        with connect(server) as client:
            initial_messages(client)
            server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
            assert entered.wait(1)
            server.set_frame_data((torch.full((3, 64, 128), 0.25), 2.0))
            latest = torch.ones(3, 64, 128)
            server.set_frame_data((latest, 3.0), take_ownership=True)
            latest.zero_()
            release.set()
            assert read_message(client)[1][9:] == b"\x00"
            assert read_message(client)[1][9:] == b"\xff"
            deadline = time.monotonic() + 0.2
            while server.get_fps() == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert server.get_fps() > 0
    finally:
        release.set()
        server.stop()


def test_backpressure_stop_is_bounded(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()

    class LargeEncoder(EncoderDouble):
        def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False) -> list[EncodedPacket]:
            entered.set()
            return [EncodedPacket(b"x" * (8 * 1024 * 1024), pts, True)]

    monkeypatch.setattr(module, "QuestEncoder", LargeEncoder)
    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0)
    server.start()
    with connect(server) as client:
        initial_messages(client)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
        assert entered.wait(1)
        started = time.monotonic()
        server.stop()
        server.stop()
        assert time.monotonic() - started < 2
        server.check_error()


def test_encoder_failure_reaches_caller(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEncoder(EncoderDouble):
        def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False) -> list[EncodedPacket]:
            raise RuntimeError("injected NVENC failure")

    monkeypatch.setattr(module, "QuestEncoder", BrokenEncoder)
    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0)
    server.start()
    try:
        with connect(server) as client:
            initial_messages(client)
            server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                try:
                    server.check_error()
                except RuntimeError as error:
                    assert "NVENC" in str(error.__cause__)
                    break
                time.sleep(0.01)
            else:
                pytest.fail("encoder failure did not reach caller")
    finally:
        server.stop()


def test_audio_sends_every_chunk_without_video(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    chunks: deque[bytes] = deque()

    class AudioDouble:
        sample_rate = 48000

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"gain": 1.0, "source": "speaker"}

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def check_error(self) -> None:
            pass

        def get_chunks(self) -> list[bytes]:
            result = []
            while chunks:
                result.append(chunks.popleft())
            return result

    monkeypatch.setattr(module, "QuestAudio", AudioDouble)
    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=True, port=0, discovery_port=0)
    server.start()
    try:
        with connect(server) as client:
            initial_messages(client)
            kind, payload = read_message(client)
            assert kind == 0x03 and json.loads(payload) == {"taxa": 48000, "canais": 2}
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            deadline = time.monotonic() + 1
            while server._audio_discard_pending and time.monotonic() < deadline:
                threading.Event().wait(0.001)
            assert not server._audio_discard_pending
            chunks.extend([b"\x01\x00\x02\x00" * 480, b"\x03\x00\x04\x00" * 480, b"\x05\x00\x06\x00" * 480])
            for value in (1, 3, 5):
                kind, payload = read_message(client)
                assert kind == 0x04 and len(payload) == 1928
                assert payload[8:] == bytes([value, 0, value + 1, 0]) * 480
    finally:
        server.stop()


def test_audio_source_and_live_gain_reach_capture(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    source = MagicMock()
    capture = MagicMock(return_value=source)
    source.get_chunks.return_value = []
    monkeypatch.setattr(module, "QuestAudio", capture)
    server = module.QuestServer(
        "127.0.0.1",
        128,
        64,
        30,
        35,
        audio=True,
        port=0,
        discovery_port=0,
        audio_source="process",
        audio_gain=1.5,
    )
    server.start()
    try:
        capture.assert_called_once_with(gain=1.5, source="process")
        server.set_audio_gain(2.25)
        source.set_gain.assert_called_once_with(2.25)
    finally:
        server.stop()
    source.stop.assert_called_once()


@pytest.mark.hwaccel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVENC GPU")
def test_cuda_producer_stream_owns_frame_through_real_tcp_encode(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import av
    import numpy as np

    from iw3.desktop.quest_encoder import QuestEncoder
    from nunif.utils.video import pyav_init_cuda_primary_context

    pyav_init_cuda_primary_context()
    monkeypatch.setattr(module, "QuestEncoder", QuestEncoder)
    server = module.QuestServer("127.0.0.1", 640, 180, 30, 35, audio=False, port=0, discovery_port=0)
    server.start()
    try:
        with connect(server) as client:
            assert read_message(client)[0] == 0x01
            assert read_message(client)[0] == 0x11
            producer = torch.cuda.Stream()
            with torch.cuda.stream(producer):
                frame = torch.zeros(3, 180, 640, device="cuda")
                frame[0, :, :320] = 1
                frame[1, :, 320:] = 1
                server.set_frame_data((frame, 1.0))
                frame.zero_()
            kind, payload = read_message(client)
            assert kind == 0x02 and payload[8] == 1
            decoder = av.CodecContext.create("hevc", "r")
            pictures = decoder.decode(av.Packet(payload[9:])) + decoder.decode(None)
            assert len(pictures) == 1
            pixels = pictures[0].to_ndarray(format="rgb24")
            np.testing.assert_allclose(pixels[90, 160], [255, 0, 0], atol=10)
            np.testing.assert_allclose(pixels[90, 480], [0, 255, 0], atol=10)
            server.check_error()
    finally:
        server.stop()


@pytest.mark.hwaccel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA NVENC")
def test_4k_prepares_next_frame_while_encoding_and_discards_stale_frames(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import av
    import numpy as np

    from iw3.desktop.quest_encoder import PreparedCudaFrame, QuestEncoder
    from nunif.utils.video import pyav_init_cuda_primary_context

    pyav_init_cuda_primary_context(max_devices=1)
    width, height = 7680, 2160
    encoding, release = threading.Event(), threading.Event()
    preparations: list[torch.cuda.Event] = []
    consumed: list[int] = []
    transferred_pointers: list[int] = []

    class ObservedEncoder(QuestEncoder):
        def cuda_preparer(self):
            prepare = super().cuda_preparer()
            assert prepare is not None

            def observe(frame: torch.Tensor) -> PreparedCudaFrame | None:
                ticket = prepare(frame)
                assert ticket is not None
                preparations.append(ticket.ready)
                return ticket

            return observe

        def encode(
            self,
            frame: torch.Tensor,
            pts: int,
            force_idr: bool = False,
            *,
            prepared: PreparedCudaFrame | None = None,
        ) -> list[av.Packet]:
            if prepared is not None:
                consumed.append(pts)
            if pts == 2_000_000:
                if prepared is not None:
                    prepared.ready.synchronize()
                encoding.set()
                assert release.wait(10)
            if pts in (5_000_000, 6_000_000):
                transferred_pointers.append(frame.data_ptr())
            return super().encode(frame, pts, force_idr, prepared=prepared)

    monkeypatch.setattr(module, "QuestEncoder", ObservedEncoder)
    server = module.QuestServer("127.0.0.1", width, height, 30, 35, audio=False, port=0, discovery_port=0)
    producer = torch.cuda.Stream()

    def submit(tick: int, swap: bool = False, *, take_ownership: bool = False) -> int:
        with torch.cuda.stream(producer):
            rgb = torch.zeros(3, height, width, device="cuda", dtype=torch.float32)
            rgb[int(swap), :, : width // 2] = 1
            rgb[1 - int(swap), :, width // 2 :] = 1
            if take_ownership:
                server.set_frame_data((rgb, float(tick)), take_ownership=True)
            else:
                server.set_frame_data((rgb, float(tick)))
                rgb.zero_()
            return rgb.data_ptr()

    server.start()
    try:
        with connect(server) as client:
            client.settimeout(10)
            assert [read_message(client)[0] for _ in range(2)] == [0x01, 0x11]
            submit(1)
            assert read_message(client)[0] == 0x02
            assert not preparations  # 首帧先确认可用的编码路径。
            submit(2)
            assert encoding.wait(10)
            submit(3)
            assert len(preparations) == 2
            # 转色未完成时，新帧仍替换待编码槽，但不能排入更多转色。
            with monkeypatch.context() as patch:
                patch.setattr(preparations[-1], "query", lambda: False)
                submit(4)
                assert len(preparations) == 2
                server.discard_video_frames()
                # 调用方只返回地址，在编码线程恢复前就已释放 RGB 引用。
                transferred_pointer = submit(5, swap=True, take_ownership=True)
                release.set()
                kind, after_switch = read_message(client)
                assert kind == 0x02 and struct.unpack("!QB", after_switch[:9]) == (5_000_000, 1)
                assert transferred_pointers == [transferred_pointer]
            preparations[-1].synchronize()
            following_pointer = submit(6, take_ownership=True)
            kind, following = read_message(client)
            assert kind == 0x02 and int.from_bytes(following[:8], "big") == 6_000_000
            assert transferred_pointers == [transferred_pointer, following_pointer]
            assert consumed == [2_000_000, 6_000_000]
            decoder = av.CodecContext.create("hevc", "r")
            pictures = decoder.decode(av.Packet(after_switch[9:])) + decoder.decode(av.Packet(following[9:]))
            pictures += decoder.decode(None)
            assert len(pictures) == 2
            for picture, (left, right) in zip(pictures, ((1, 0), (0, 1)), strict=True):
                pixels = picture.to_ndarray(format="rgb24")
                np.testing.assert_allclose(pixels[height // 2, width // 4], np.eye(3)[left] * 255, atol=10)
                np.testing.assert_allclose(pixels[height // 2, width * 3 // 4], np.eye(3)[right] * 255, atol=10)
            server.check_error()
    finally:
        release.set()
        server.stop()
