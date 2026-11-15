import importlib
import json
import struct
import threading
import time
import weakref
from typing import Any

import pytest
import torch

from .test_quest_control_server import RecordedInput, entrada
from .test_quest_server import EncoderDouble, connect, initial_messages, read_message, send_json


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = importlib.import_module("iw3.desktop.quest_server")
    monkeypatch.setattr(module, "QuestEncoder", EncoderDouble)
    return module


def make_server(module: Any, **kwargs: Any) -> Any:
    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0, **kwargs)
    server.start()
    return server


def test_configure_without_frames_sends_config_before_new_idr_and_preserves_pts(module: Any) -> None:
    server = make_server(module)
    try:
        with connect(server) as client:
            initial_messages(client)
            server.set_frame_data((torch.zeros(3, 64, 128), 2.0))
            assert read_message(client)[1][:9] == struct.pack("!QB", 2_000_000, 1)
            server.pause_media()
            server.configure_video(256, 96, 60, 45)
            kind, payload = read_message(client)
            assert kind == 0x01
            assert json.loads(payload) == {"largura": 256, "altura": 96, "fps": 60, "codec": "hevc"}
            kind, payload = read_message(client)
            assert kind == 0x11
            assert json.loads(payload)["camadas"][0]["rect"] == [0, 0, 128, 96]
            server.set_frame_data((torch.ones(3, 96, 256), 1.0))
            kind, payload = read_message(client)
            assert kind == 0x02 and payload[:9] == struct.pack("!QB", 2_000_001, 1)
            assert server._connection is not None
    finally:
        server.stop()
    assert not server._threads


def test_reconfigure_rejects_inflight_old_generation(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()

    class SlowEncoder(EncoderDouble):
        def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False):
            entered.set()
            assert release.wait(2)
            return super().encode(frame, pts, force_idr)

    monkeypatch.setattr(module, "QuestEncoder", SlowEncoder)
    server = make_server(module)
    finished = threading.Event()
    errors: list[Exception] = []

    def configure() -> None:
        try:
            server.configure_video(256, 96, 60, 45)
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    try:
        with connect(server) as client:
            initial_messages(client)
            server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
            assert entered.wait(1)
            worker = threading.Thread(target=configure)
            worker.start()
            # Request reaches the owner even while an old frame is encoding.
            with server._condition:
                assert server._condition.wait_for(lambda: server._video_request is not None, timeout=1)
            release.set()
            assert finished.wait(2)
            worker.join()
            assert not errors
            assert read_message(client)[0] == 0x01
            assert read_message(client)[0] == 0x11
            server.set_frame_data((torch.ones(3, 96, 256), 2.0))
            assert read_message(client)[0] == 0x02
    finally:
        release.set()
        server.stop()


def test_media_failure_keeps_management_connection_available(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEncoder(EncoderDouble):
        def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False):
            raise RuntimeError("injected NVENC failure")

    monkeypatch.setattr(module, "QuestEncoder", BrokenEncoder)
    server = make_server(module)
    try:
        with connect(server) as client:
            initial_messages(client)
            server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
            with server._condition:
                assert server._condition.wait_for(lambda: server.media_error is not None, timeout=1)
            assert "NVENC" in str(server.take_media_error())
            assert server.media_error is None and not server._shutdown.is_set()
            send_json(client, 0x83, entrada())
            assert read_message(client)[0] == 0x11
            monkeypatch.setattr(module, "QuestEncoder", EncoderDouble)
            server.configure_video(128, 64, 30, 35)
            assert read_message(client)[0] == 0x01
            assert read_message(client)[0] == 0x11
            server.set_frame_data((torch.ones(3, 64, 128), 2.0))
            assert read_message(client)[0] == 0x02
            server.check_error()
    finally:
        server.stop()


def test_audio_config_failure_is_synchronous_and_does_not_disconnect(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owners: list[int] = []

    class BrokenAudio:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            owners.append(threading.get_ident())
            raise RuntimeError("injected audio start")

        def stop(self) -> None:
            owners.append(threading.get_ident())

    monkeypatch.setattr(module, "QuestAudio", BrokenAudio)
    server = make_server(module)
    try:
        with connect(server) as client:
            initial_messages(client)
            with pytest.raises(RuntimeError, match="injected audio start"):
                server.configure_audio(True, "process", 1.5)
            assert owners and all(owner != threading.get_ident() for owner in owners)
            assert len(set(owners)) == 1
            assert server._connection is not None and not server._shutdown.is_set()
            send_json(client, 0x83, entrada())
            assert read_message(client)[0] == 0x11
    finally:
        server.stop()


def test_controller_factory_can_be_replaced_on_same_connection(module: Any) -> None:
    from iw3.desktop.quest_input import QuestController

    old, new = RecordedInput(), RecordedInput()
    server = make_server(module, controller_factory=lambda: QuestController(old))
    try:
        with connect(server) as client:
            initial_messages(client)
            send_json(client, 0x83, entrada())
            read_message(client)
            send_json(client, 0x83, entrada(trigger=True))
            old.wait("down")
            server.set_controller_factory(lambda: QuestController(new))
            old.wait("up")
            send_json(client, 0x83, entrada(trigger=True))
            new.wait((0.5, 0.5))
            assert "down" not in new.events
            send_json(client, 0x83, entrada())
            send_json(client, 0x83, entrada(trigger=True))
            new.wait("down")
    finally:
        server.stop()


def test_controller_replacement_serializes_with_handshake(module: Any) -> None:
    from iw3.desktop.quest_input import QuestController

    entered, release, configured = threading.Event(), threading.Event(), threading.Event()
    old, new = RecordedInput(), RecordedInput()

    def factory() -> QuestController:
        entered.set()
        assert release.wait(2)
        return QuestController(old)

    server = make_server(module, controller_factory=factory)

    def replace() -> None:
        server.set_controller_factory(lambda: QuestController(new))
        configured.set()

    worker = threading.Thread(target=replace)
    try:
        with connect(server) as client:
            initial_messages(client)
            assert entered.wait(1)
            worker.start()
            assert not configured.wait(0.05)
            release.set()
            assert configured.wait(2)
            send_json(client, 0x83, entrada())
            assert read_message(client)[0] == 0x11
            new.wait((0.5, 0.5))
            assert not old.events
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(2)
        server.stop()


def test_recenter_command_restores_screen_on_next_pose(module: Any) -> None:
    server = make_server(module)
    try:
        with connect(server) as client:
            initial_messages(client)
            send_json(client, 0x83, entrada())
            read_message(client)
            server.recenter()
            data = entrada()
            data["cabeca"]["pos"] = [1, 2, 3]
            send_json(client, 0x83, data)
            kind, payload = read_message(client)
            assert kind == 0x11 and json.loads(payload)["camadas"][0]["pos"] == [1, 2, 0]
    finally:
        server.stop()


def test_pause_releases_owner_encoder_even_without_connection(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[weakref.ReferenceType] = []

    class TrackedEncoder(EncoderDouble):
        def __init__(self, *args: Any) -> None:
            instances.append(weakref.ref(self))

    monkeypatch.setattr(module, "QuestEncoder", TrackedEncoder)
    server = make_server(module)
    try:
        server.configure_video(256, 96, 60, 45)
        assert len(instances) == 1 and instances[0]() is not None
        server.pause_media()
        assert instances[0]() is None
    finally:
        server.stop()


def test_pause_releases_encoded_tensor_reference(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    frames: list[weakref.ReferenceType] = []

    class TrackedEncoder(EncoderDouble):
        def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False):
            frames.append(weakref.ref(frame))
            return super().encode(frame, pts, force_idr)

    monkeypatch.setattr(module, "QuestEncoder", TrackedEncoder)
    server = make_server(module)
    try:
        with connect(server) as client:
            initial_messages(client)
            server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
            assert read_message(client)[0] == 0x02
            server.pause_media()
            assert frames[0]() is None
    finally:
        server.stop()


def test_paused_producer_does_not_copy_or_queue_old_geometry(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    server = make_server(module)
    try:
        server.pause_media()
        frame = torch.zeros(3, 64, 128)

        def forbidden(*args: Any) -> None:
            raise AssertionError("paused media must not copy frames")

        monkeypatch.setattr(torch.Tensor, "clone", forbidden)
        server.set_frame_data((frame, 1.0))
        assert server._pending is None
    finally:
        server.stop()


def test_connection_handshake_serializes_with_reconfiguration(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release, configured = threading.Event(), threading.Event(), threading.Event()
    server = make_server(module)
    original = server._video_config

    def stalled_handshake(connection: Any) -> None:
        original(connection)
        if not entered.is_set():
            entered.set()
            assert release.wait(2)

    monkeypatch.setattr(server, "_video_config", stalled_handshake)

    def configure() -> None:
        server.configure_video(256, 96, 60, 45)
        configured.set()

    worker = threading.Thread(target=configure)
    try:
        with connect(server) as client:
            assert read_message(client)[0] == 0x01
            assert entered.wait(1)
            worker.start()
            with server._condition:
                assert server._condition.wait_for(lambda: server._generation > 0, timeout=1)
            release.set()
            kind, payload = read_message(client)
            assert kind == 0x11 and json.loads(payload)["camadas"][0]["rect"] == [0, 0, 64, 64]
            assert configured.wait(2)
            assert read_message(client)[0] == 0x01
            kind, payload = read_message(client)
            assert kind == 0x11 and json.loads(payload)["camadas"][0]["rect"] == [0, 0, 128, 96]
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(2)
        server.stop()


def test_live_gain_is_fast_and_does_not_queue_owner_commands(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    source = MagicMock(sample_rate=48000)
    source.get_chunks.return_value = []
    capture = MagicMock(return_value=source)
    monkeypatch.setattr(module, "QuestAudio", capture)
    server = make_server(module)
    try:
        server.configure_audio(True, "speaker", 1.0)
        with connect(server) as client:
            initial_messages(client)
            assert read_message(client)[0] == 0x03

            def forbidden(*args: Any) -> None:
                raise AssertionError("gain must not wait for owner commands")

            monkeypatch.setattr(server, "_request_media", forbidden)
            started = time.monotonic()
            for _ in range(50):
                server.set_audio_gain(1.0)
            server.set_audio_gain(2.0)
            assert time.monotonic() - started < 0.1
            capture.assert_called_once_with(source="speaker", gain=1.0)
            source.start.assert_called_once()
            source.set_gain.assert_called_once_with(2.0)
    finally:
        server.stop()


def test_audio_config_serializes_with_connection_handshake(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    entered, release, configured = threading.Event(), threading.Event(), threading.Event()
    source = MagicMock(sample_rate=48000)
    source.get_chunks.return_value = []
    monkeypatch.setattr(module, "QuestAudio", MagicMock(return_value=source))

    def factory() -> None:
        entered.set()
        assert release.wait(2)

    server = make_server(module, controller_factory=factory)

    def configure() -> None:
        server.configure_audio(True, "speaker", 1.0)
        configured.set()

    worker = threading.Thread(target=configure)
    try:
        with connect(server) as client:
            initial_messages(client)
            assert entered.wait(1)
            worker.start()
            assert not configured.wait(0.05)
            release.set()
            assert configured.wait(2)
            kind, payload = read_message(client)
            assert kind == 0x03 and json.loads(payload) == {"taxa": 48000, "canais": 2}
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(2)
        server.stop()


def test_audio_source_switch_and_runtime_failure_preserve_connection(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances: list[Any] = []

    class Audio:
        sample_rate = 48000

        def __init__(self, source: str, gain: float) -> None:
            self.source, self.gain = source, gain
            self.failed = False
            self.stopped = threading.Event()
            self.owner = threading.get_ident()
            instances.append(self)

        def start(self) -> None:
            assert self.owner == threading.get_ident()

        def stop(self) -> None:
            assert self.owner == threading.get_ident()
            self.stopped.set()

        def set_gain(self, gain: float) -> None:
            self.gain = gain

        def check_error(self) -> None:
            if self.failed:
                raise RuntimeError("audio capture failed")

        def get_chunks(self) -> list[bytes]:
            return []

    monkeypatch.setattr(module, "QuestAudio", Audio)
    server = make_server(module)
    try:
        with connect(server) as client:
            initial_messages(client)
            server.configure_audio(True, "speaker", 1.0)
            assert read_message(client)[0] == 0x03
            server.configure_audio(True, "process", 2.0)
            assert instances[0].stopped.is_set()
            assert read_message(client)[0] == 0x03
            assert instances[1].source == "process" and instances[1].gain == 2.0
            instances[1].failed = True
            assert instances[1].stopped.wait(1)
            assert not server._shutdown.is_set()
            assert "audio capture failed" in str(server.take_media_error())
            send_json(client, 0x83, entrada())
            assert read_message(client)[0] == 0x11
    finally:
        server.stop()


class MenuDouble:
    def __init__(self) -> None:
        self.visible = False
        self.dirty = False
        self.seen_quads: list[dict[str, Any]] = []
        self.changed = threading.Event()

    def update(self, data: dict[str, Any], video_quad: dict[str, Any]) -> bool:
        self.seen_quads.append(video_quad)
        if data.get("show"):
            self.visible = self.dirty = True
        self.changed.set()
        return self.visible

    def reset(self) -> None:
        self.visible = False

    def texture(self) -> bytes | None:
        dirty, self.dirty = self.dirty, False
        return b"menu payload" if dirty else None

    def layers(self) -> list[dict[str, Any]]:
        return [{"tipo": "quad", "tex": "iw3_settings"}, {"tipo": "laser", "mao": "dir"}] if self.visible else []


def test_stale_input_thread_cannot_install_its_screen(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from iw3.desktop.quest_screen import QuestScreen

    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False)
    server._shutdown.clear()
    server._connection = MagicMock()
    current_screen = server._screen = QuestScreen(1)

    def closed(*args: Any, **kwargs: Any) -> bytes:
        raise ConnectionError("old connection closed")

    monkeypatch.setattr(server, "_read_exact", closed)
    server._input_loop(MagicMock())
    assert server._screen is current_screen


def test_stale_input_expiry_cannot_hide_current_menu(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False)
    server._connection = MagicMock()
    menu = server._menu = MenuDouble()
    menu.visible = True
    last_input = server._last_input = time.monotonic() - 1
    monkeypatch.setattr(server, "_send_json", lambda *args: None)
    server._expire_input(MagicMock())
    assert menu.visible and server._last_input == last_input


@pytest.mark.parametrize("payload", [json.dumps(entrada()).encode(), b"{"])
def test_already_read_old_input_cannot_mutate_new_menu(
    module: Any, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    from unittest.mock import MagicMock

    from iw3.desktop.quest_screen import QuestScreen

    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False)
    server._shutdown.clear()
    old_connection, new_connection = MagicMock(), MagicMock()
    server._connection = old_connection
    menu = server._menu = MenuDouble()
    menu.visible = True
    current_screen = QuestScreen(1)
    reads = iter([bytes([0x83]) + len(payload).to_bytes(4, "big"), payload])

    def read(*args: Any, **kwargs: Any) -> bytes:
        try:
            value = next(reads)
        except StopIteration:
            raise ConnectionError("old connection closed") from None
        if value is payload:
            server._connection = new_connection
            server._screen = current_screen
        return value

    monkeypatch.setattr(server, "_read_exact", read)
    monkeypatch.setattr(server, "_send_json", lambda *args: None)
    server._input_loop(old_connection)
    assert menu.visible and not menu.changed.is_set()
    assert server._screen is current_screen and server._connection is new_connection


def test_disconnect_finishes_reset_before_next_handshake(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    server = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False)
    old_connection, new_connection = MagicMock(), MagicMock()
    server._connection = old_connection
    menu = server._menu = MenuDouble()
    reset_entered, finish_reset = threading.Event(), threading.Event()
    reconnect_started, reconnected = threading.Event(), threading.Event()
    original_reset = server._reset_input

    def delayed_reset() -> None:
        reset_entered.set()
        assert finish_reset.wait(2)
        original_reset()

    def reconnect() -> None:
        reconnect_started.set()
        with server._layer_lock:
            with server._condition:
                server._connection = new_connection
            menu.visible = True
        reconnected.set()

    monkeypatch.setattr(server, "_reset_input", delayed_reset)
    disconnect_worker = threading.Thread(target=server._disconnect, args=(old_connection,))
    reconnect_worker = threading.Thread(target=reconnect)
    disconnect_worker.start()
    try:
        assert reset_entered.wait(1)
        reconnect_worker.start()
        assert reconnect_started.wait(1)
        reconnected.wait(0.1)
    finally:
        finish_reset.set()
        disconnect_worker.join(2)
        if reconnect_worker.ident is not None:
            reconnect_worker.join(2)
    assert not disconnect_worker.is_alive() and not reconnect_worker.is_alive()
    assert server._connection is new_connection and menu.visible
    old_connection.close.assert_called_once()
    new_connection.close.assert_not_called()


def test_menu_owns_input_and_layers_survive_video_reconfigure_then_expire(module: Any) -> None:
    from iw3.desktop.quest_input import QuestController

    backend = RecordedInput()
    server = make_server(module, controller_factory=lambda: QuestController(backend))
    menu = MenuDouble()
    server.set_menu(menu)
    try:
        with connect(server) as client:
            initial_messages(client)
            send_json(client, 0x83, entrada())
            read_message(client)
            send_json(client, 0x83, entrada(trigger=True))
            backend.wait("down")
            data = entrada(trigger=True)
            data["show"] = True
            data["maos"]["dir"].update(aperto=True, pos=[0.4, 0, 0])
            send_json(client, 0x83, data)
            assert read_message(client) == (0x10, b"menu payload")
            kind, payload = read_message(client)
            layers = json.loads(payload)["camadas"]
            assert kind == 0x11 and [layer["tipo"] for layer in layers] == ["quad", "quad", "quad", "laser"]
            assert layers[0]["pos"] == [0, 0, -3]
            backend.wait("up")
            server.configure_video(256, 96, 60, 45)
            assert read_message(client)[0] == 0x01
            kind, payload = read_message(client)
            assert kind == 0x11 and len(json.loads(payload)["camadas"]) == 4
            kind, payload = read_message(client)
            assert kind == 0x11 and len(json.loads(payload)["camadas"]) == 2
            assert not menu.visible
            assert backend.events.count("down") == 1
    finally:
        server.stop()


def test_malformed_input_removes_visible_menu_layers(module: Any) -> None:
    server = make_server(module)
    menu = MenuDouble()
    server.set_menu(menu)
    try:
        with connect(server) as client:
            initial_messages(client)
            data = entrada()
            data["show"] = True
            send_json(client, 0x83, data)
            assert read_message(client)[0] == 0x10
            assert read_message(client)[0] == 0x11
            client.sendall(b"\x83\x00\x00\x00\x01{")
            kind, payload = read_message(client)
            assert kind == 0x11 and len(json.loads(payload)["camadas"]) == 2
    finally:
        server.stop()


def test_stop_resets_menu_state(module: Any) -> None:
    server = make_server(module)
    menu = MenuDouble()
    server.set_menu(menu)
    menu.visible = True
    server.stop()
    assert not menu.visible


@pytest.mark.hwaccel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVENC GPU")
def test_real_nvenc_reconfigure_keeps_socket_and_decodes_new_geometry(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import av

    from iw3.desktop.quest_encoder import QuestEncoder
    from nunif.utils.video import pyav_init_cuda_primary_context

    pyav_init_cuda_primary_context()
    monkeypatch.setattr(module, "QuestEncoder", QuestEncoder)
    server = make_server(module)
    try:
        server.configure_video(640, 180, 30, 35)
        with connect(server) as client:
            assert read_message(client)[0] == 0x01
            assert read_message(client)[0] == 0x11
            for index, (width, height, fps, bitrate) in enumerate(((640, 180, 30, 35), (768, 256, 60, 45))):
                if index:
                    server.pause_media()
                    server.configure_video(width, height, fps, bitrate)
                    assert read_message(client)[0] == 0x01
                    assert read_message(client)[0] == 0x11
                server.set_frame_data((torch.full((3, height, width), 0.5, device="cuda"), 1.0))
                kind, payload = read_message(client)
                assert kind == 0x02 and payload[8] == 1
                decoder = av.CodecContext.create("hevc", "r")
                pictures = decoder.decode(av.Packet(payload[9:])) + decoder.decode(None)
                assert len(pictures) == 1 and (pictures[0].width, pictures[0].height) == (width, height)
                assert struct.unpack("!Q", payload[:8])[0] == 1_000_000 + index
    finally:
        server.stop()
    assert not server._threads
