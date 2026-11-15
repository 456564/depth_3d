import json
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from .test_quest_menu import menu as menu
from .test_quest_menu import packet, point
from .test_quest_server import EncodedPacket, EncoderDouble, connect, initial_messages, read_message, send_json
from .test_quest_server import module as module
from .test_quest_server import server as server


def wait_until(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 1
    while not predicate() and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    assert predicate()


def state(client: socket.socket, server: Any, value: str) -> None:
    previous = server._last_input
    send_json(client, 0x83, {"estado": value})
    wait_until(lambda: server._last_input != previous)


def quiet(client: socket.socket) -> None:
    client.settimeout(0.1)
    try:
        with pytest.raises(socket.timeout):
            read_message(client)
    finally:
        client.settimeout(1)


@pytest.mark.parametrize("restored", ["focado", "visivel"])
def test_sleep_blocks_video_until_resume_resends_configuration_and_first_idr(server: Any, restored: str) -> None:
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, {"estado": "focado", "cabeca": {"pos": [1, 2, 3], "rot": [0, 0, 0, 1]}})
        positioned = json.loads(read_message(client)[1])
        assert positioned["camadas"][0]["pos"] == [1, 2, 0]
        for tick, expected_key in [(1.0, 1), (2.0, 0)]:
            server.set_frame_data((torch.zeros(3, 64, 128), tick))
            assert read_message(client)[1][8] == expected_key
        state(client, server, "oculto")
        server.set_frame_data((torch.ones(3, 64, 128), 9.0))
        quiet(client)
        send_json(client, 0x83, {"estado": restored, "cabeca": {"pos": [8, 8, 8], "rot": [0, 0, 0, 1]}})
        assert read_message(client)[0] == 0x01
        assert json.loads(read_message(client)[1]) == positioned
        wait_until(lambda: not server._suspended)
        server.set_frame_data((torch.zeros(3, 64, 128), 11.0))
        kind, payload = read_message(client)
        assert kind == 0x02 and payload[8:] == b"\x01\x00"
        assert int.from_bytes(payload[:8], "big") == 11_000_000
        state(client, server, restored)
        server.set_frame_data((torch.zeros(3, 64, 128), 12.0))
        assert read_message(client)[1][8] == 0
        server.check_error()


@pytest.mark.parametrize("blocked_at", ["encoding", "send"])
def test_frame_already_inflight_when_sleep_arrives_is_not_sent(
    module: Any, monkeypatch: pytest.MonkeyPatch, blocked_at: str
) -> None:
    entered, release = threading.Event(), threading.Event()

    class SlowEncoder(EncoderDouble):
        def encode(self, frame: torch.Tensor, pts: int, force_idr: bool = False) -> list[EncodedPacket]:
            if float(frame[0, 0, 0]) == 0.5:
                entered.set()
                if blocked_at == "encoding":
                    assert release.wait(2)
            return super().encode(frame, pts, force_idr)

    monkeypatch.setattr(module, "QuestEncoder", SlowEncoder)
    target = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0)
    target.start()
    holding_write_lock = False
    try:
        with connect(target) as client:
            initial_messages(client)
            state(client, target, "focado")
            if blocked_at == "send":
                target._write_lock.acquire()
                holding_write_lock = True
            target.set_frame_data((torch.full((3, 64, 128), 0.5), 1.0))
            assert entered.wait(1)
            if blocked_at == "send":
                threading.Event().wait(0.02)
            state(client, target, "oculto")
            release.set()
            if blocked_at == "send":
                target._write_lock.release()
                holding_write_lock = False
            quiet(client)
            send_json(client, 0x83, {"estado": "focado"})
            assert read_message(client)[0] == 0x01
            assert read_message(client)[0] == 0x11
            wait_until(lambda: not target._suspended)
            target.set_frame_data((torch.ones(3, 64, 128), 2.0))
            assert read_message(client)[1][8:] == b"\x01\xff"
    finally:
        release.set()
        if holding_write_lock:
            target._write_lock.release()
        target.stop()


def test_sleep_discards_audio_and_ambient_and_reconnect_clears_sleep(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    target = module.QuestServer(
        "127.0.0.1", 128, 64, 30, 35, audio=True, port=0, discovery_port=0, ambient_enabled=True
    )
    target.start()
    try:
        with connect(target) as client:
            initial_messages(client)
            assert read_message(client)[0] == 0x03
            state(client, target, "focado")
            target.set_ambient_frame(torch.ones(3, 64, 64))
            target.set_frame_data((torch.zeros(3, 64, 128), 1.0))
            assert [read_message(client)[0] for _ in range(3)] == [0x02, 0x10, 0x11]
            state(client, target, "oculto")
            chunks.append(b"old audio")
            wait_until(lambda: not chunks)
            target.set_ambient_frame(torch.zeros(3, 64, 64))
            target.set_frame_data((torch.ones(3, 64, 128), 2.0))
            quiet(client)
            send_json(client, 0x83, {"estado": "focado"})
            assert read_message(client)[0] == 0x01
            assert read_message(client)[0] == 0x03
            kind, payload = read_message(client)
            assert kind == 0x11 and len(json.loads(payload)["camadas"]) == 2
            wait_until(lambda: not target._suspended and not target._audio_discard_pending)
            chunks.append(b"fresh audio")
            kind, payload = read_message(client)
            assert kind == 0x04 and payload[8:] == b"fresh audio"
            target.set_ambient_frame(torch.ones(3, 64, 64))
            target.set_frame_data((torch.zeros(3, 64, 128), 3.0))
            assert [read_message(client)[0] for _ in range(3)] == [0x02, 0x10, 0x11]
            state(client, target, "oculto")
        wait_until(lambda: target._connection is None)
        with connect(target) as client:
            initial_messages(client)
            assert read_message(client)[0] == 0x03
            target.set_frame_data((torch.ones(3, 64, 128), 4.0))
            assert read_message(client)[1][8:] == b"\x01\xff"
        target.check_error()
    finally:
        target.stop()


@pytest.mark.parametrize("value", ["visivel", "other", None])
def test_visible_and_unknown_states_keep_legacy_streaming(server: Any, value: str | None) -> None:
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, {} if value is None else {"estado": value})
        wait_until(lambda: server._last_input is not None)
        server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
        assert read_message(client)[0] == 0x02


def test_reconfigure_while_asleep_does_not_resume_media_until_visible(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        state(client, server, "focado")
        state(client, server, "oculto")
        server.configure_video(256, 96, 45, 20)
        kind, payload = read_message(client)
        assert kind == 0x01 and json.loads(payload) == {"largura": 256, "altura": 96, "fps": 45, "codec": "hevc"}
        assert read_message(client)[0] == 0x11
        server.set_frame_data((torch.ones(3, 96, 256), 1.0))
        quiet(client)
        send_json(client, 0x83, {"estado": "visivel"})
        assert read_message(client)[0] == 0x01
        assert read_message(client)[0] == 0x11
        wait_until(lambda: not server._suspended)
        server.set_frame_data((torch.zeros(3, 96, 256), 2.0))
        assert read_message(client)[1][8:] == b"\x01\x00"


@pytest.mark.parametrize("transition", ["resume", "reconnect"])
@pytest.mark.parametrize("blocked_at", ["before_read", "after_read"])
def test_queued_audio_is_discarded_even_if_owner_reads_after_resume_or_reconnect(
    module: Any, monkeypatch: pytest.MonkeyPatch, transition: str, blocked_at: str
) -> None:
    chunks: deque[bytes] = deque()
    block, entered, release = threading.Event(), threading.Event(), threading.Event()

    class PausedAudio:
        sample_rate = 48000

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"gain": 1.0, "source": "speaker"}

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def check_error(self) -> None:
            if block.is_set() and blocked_at == "before_read":
                entered.set()
                assert release.wait(2)

        def get_chunks(self) -> list[bytes]:
            result = []
            while chunks:
                result.append(chunks.popleft())
            if block.is_set() and blocked_at == "after_read":
                entered.set()
                assert release.wait(2)
            return result

    monkeypatch.setattr(module, "QuestAudio", PausedAudio)
    target = module.QuestServer("127.0.0.1", 128, 64, 30, 35, audio=True, port=0, discovery_port=0)
    target.start()
    try:
        with connect(target) as client:
            initial_messages(client)
            assert read_message(client)[0] == 0x03
            state(client, target, "focado")
            wait_until(lambda: not target._audio_discard_pending)
            block.set()
            assert entered.wait(1)
            if transition == "reconnect":
                client.close()
                wait_until(lambda: target._connection is None)
                chunks.append(b"disconnected audio")
                with connect(target) as second:
                    initial_messages(second)
                    assert read_message(second)[0] == 0x03
                    wait_until(lambda: target._connection is not None)
                    release.set()
                    quiet(second)
                    chunks.append(b"fresh audio")
                    kind, payload = read_message(second)
                    assert kind == 0x04 and payload[8:] == b"fresh audio"
                return
            state(client, target, "oculto")
            chunks.append(b"hidden audio")
            send_json(client, 0x83, {"estado": "visivel"})
            assert [read_message(client)[0] for _ in range(3)] == [0x01, 0x03, 0x11]
            wait_until(lambda: not target._suspended)
            release.set()
            quiet(client)
            chunks.append(b"fresh audio")
            kind, payload = read_message(client)
            assert kind == 0x04 and payload[8:] == b"fresh audio"
    finally:
        release.set()
        target.stop()


def test_moved_curved_screen_and_settings_click_survive_sleep(
    server: Any, menu: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen_clock = [time.monotonic()]
    monkeypatch.setattr("iw3.desktop.quest_screen.time", SimpleNamespace(monotonic=lambda: screen_clock[0]))
    panel, _, commands = menu
    server.set_menu(panel)
    with connect(server) as client:
        initial_messages(client)
        first = packet()
        first["cabeca"] = {"pos": [0, 0, 0], "rot": [0, 0, 0, 1]}
        send_json(client, 0x83, first)
        assert read_message(client)[0] == 0x11
        send_json(client, 0x84, {"cilindro": True})
        assert read_message(client)[0] == 0x11
        server.set_screen_curvature(60)
        assert read_message(client)[0] == 0x11
        send_json(client, 0x83, packet())
        send_json(client, 0x83, packet(aperto=True))
        wait_until(lambda: server._screen.active)
        screen_clock[0] += 0.1
        send_json(client, 0x83, packet(aperto=True, pos=[0.4, 0.2, -0.5], analogico=[0.8, 0]))
        send_json(client, 0x83, packet())
        kind, payload = read_message(client)
        before = json.loads(payload)
        assert kind == 0x11 and before["camadas"][0]["tipo"] == "cilindro"
        cylinder = before["camadas"][0]
        assert cylinder["pos"][:2] == pytest.approx([0.4, 0.2])
        assert cylinder["pos"][2] - cylinder["raio"] == pytest.approx(-3.5)
        assert cylinder["raio"] * cylinder["angulo"] > 3.6
        state(client, server, "oculto")
        send_json(client, 0x83, {"estado": "focado"})
        assert read_message(client)[0] == 0x01
        assert json.loads(read_message(client)[1]) == before
        wait_until(lambda: not server._suspended)
        send_json(client, 0x83, packet())
        send_json(client, 0x83, packet(b=True))
        assert read_message(client)[0] == 0x10
        kind, payload = read_message(client)
        assert kind == 0x11 and any(layer.get("tex") == "iw3_settings" for layer in json.loads(payload)["camadas"])
        button = next(
            rect for rect, name, value, _ in panel._buttons if name == "adjust" and value == ("divergence", 1, "x")
        )
        position = point(panel, (button[0] + button[2]) / 2, (button[1] + button[3]) / 2)
        send_json(client, 0x83, packet(pos=position))
        send_json(client, 0x83, packet(pos=position, gatilho=True))
        wait_until(lambda: commands == [("set", {"divergence": 1.25})])
        server.check_error()
