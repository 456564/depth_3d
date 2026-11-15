import socket
import threading
import time
from typing import Any

import pytest

from iw3.desktop.quest_input import QuestController
from iw3.desktop.quest_server import QuestServer

from .test_quest_server import connect, initial_messages, read_message, send_json


class RecordedInput:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.condition = threading.Condition()

    def record(self, event: Any) -> bool:
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()
        return True

    def move(self, u: float, v: float) -> bool:
        return self.record((u, v))

    def mouse_down(self) -> bool:
        return self.record("down")

    def mouse_up(self) -> None:
        self.record("up")

    def tap_key(self, direction: str) -> bool:
        return self.record(("key", direction))

    def release_keys(self) -> None:
        pass

    def wait(self, event: Any, count: int = 1) -> None:
        with self.condition:
            assert self.condition.wait_for(lambda: self.events.count(event) >= count, timeout=2), self.events


def entrada(trigger: bool = False, stick: float = 0, focused: bool = True) -> dict[str, Any]:
    return {
        "estado": "focado" if focused else "visivel",
        "cabeca": {"pos": [0, 0, 0], "rot": [0, 0, 0, 1]},
        "maos": {
            "esq": {"pos": [0, 0, 0], "rot": [0, 0, 0, 1], "aperto": False, "analogico": [0, 0]},
            "dir": {
                "pos": [0, 0, 0],
                "rot": [0, 0, 0, 1],
                "gatilho": trigger,
                "aperto": False,
                "analogico": [0, stick],
            },
        },
    }


@pytest.fixture
def controlled():
    backend = RecordedInput()
    server = QuestServer(
        "127.0.0.1",
        128,
        64,
        30,
        35,
        audio=False,
        port=0,
        discovery_port=0,
        controller_factory=lambda: QuestController(backend),
    )
    server.start()
    try:
        yield server, backend
    finally:
        server.stop()


def test_tcp_aim_drag_discrete_video_switch_and_focus_release(controlled) -> None:
    server, backend = controlled
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, entrada())
        assert read_message(client)[0] == 0x11
        backend.wait((0.5, 0.5))
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down")
        send_json(client, 0x83, entrada(trigger=True))
        send_json(client, 0x83, entrada(trigger=False))
        backend.wait("up")
        for stick in (-0.9, -0.9, 0.9, 0, -0.9):
            send_json(client, 0x83, entrada(stick=stick))
        backend.wait(("key", "down"), count=2)
        assert backend.events.count("down") == 1
        assert ("key", "up") not in backend.events
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down", count=2)
        send_json(client, 0x83, entrada(trigger=True, focused=False))
        backend.wait("up", count=2)
    server.check_error()


def test_tcp_disconnect_releases_and_reconnect_does_not_replay_held_trigger(controlled) -> None:
    server, backend = controlled
    client = connect(server)
    initial_messages(client)
    send_json(client, 0x83, entrada())
    read_message(client)
    send_json(client, 0x83, entrada(trigger=True))
    backend.wait("down")
    client.close()
    backend.wait("up")
    with connect(server) as second:
        initial_messages(second)
        send_json(second, 0x83, entrada(trigger=True))
        read_message(second)
        send_json(second, 0x83, entrada())
        backend.wait((0.5, 0.5), count=4)
        assert backend.events.count("down") == 1
        send_json(second, 0x83, entrada(trigger=True))
        backend.wait("down", count=2)
    backend.wait("up", count=2)


@pytest.mark.parametrize("keep_alive", [False, True])
def test_no_controller_input_for_500ms_releases_even_when_other_packets_arrive(controlled, keep_alive: bool) -> None:
    server, backend = controlled
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, entrada())
        read_message(client)
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down")
        if keep_alive:
            for _ in range(9):
                client.sendall(b"\x82\x00\x00\x00\x00")
                threading.Event().wait(0.08)
        backend.wait("up")
    assert backend.events.count("up") == 1
    server.check_error()


def test_stop_releases_controller_before_returning(controlled) -> None:
    server, backend = controlled
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, entrada())
        read_message(client)
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down")
        server.stop()
        assert backend.events.count("up") == 1
        assert not server._threads


def test_controller_factory_failure_is_reported_and_connection_closed() -> None:
    def failed():
        raise RuntimeError("control setup failed")

    server = QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0, controller_factory=failed)
    server.start()
    try:
        with connect(server) as client:
            client.settimeout(1)
            try:
                while client.recv(4096):
                    pass
            except (ConnectionError, socket.timeout):
                pass
        deadline = time.monotonic() + 1
        while not server._shutdown.is_set() and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        with pytest.raises(RuntimeError, match="control setup failed"):
            server.check_error()
    finally:
        server.stop()


def test_fragmented_payload_cannot_keep_mouse_held_after_input_stops(controlled) -> None:
    server, backend = controlled
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, entrada())
        read_message(client)
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down")
        client.sendall(b"\x82\x00\x00\x04\x00")
        for _ in range(9):
            try:
                client.sendall(b"x")
            except ConnectionError:
                break
            threading.Event().wait(0.08)
        backend.wait("up")
        with backend.condition:
            assert backend.events.count("up") == 1


@pytest.mark.parametrize("packet", [b"\x82\x00\x00\x00\x00", b"\x82"])
def test_late_non_input_packet_does_not_extend_release_deadline(controlled, packet: bytes) -> None:
    server, backend = controlled
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, entrada())
        read_message(client)
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down")
        started = time.monotonic()
        threading.Event().wait(0.4)
        client.sendall(packet)
        backend.wait("up")
        assert time.monotonic() - started < 0.7


def test_recenter_releases_and_requires_trigger_release_before_new_click(controlled) -> None:
    server, backend = controlled
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, entrada())
        read_message(client)
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down")
        data = entrada(trigger=True)
        data["recentrou"] = True
        send_json(client, 0x83, data)
        assert read_message(client)[0] == 0x11
        backend.wait("up")
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait((0.5, 0.5), count=4)
        assert backend.events.count("down") == 1


def test_backend_os_error_is_reported_and_releases_mouse(controlled) -> None:
    server, backend = controlled
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, entrada())
        read_message(client)
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down")

        def failed_move(u: float, v: float) -> bool:
            raise OSError("injected Windows failure")

        backend.move = failed_move
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("up")
        with pytest.raises(RuntimeError, match="injected Windows failure"):
            server.check_error()


def test_stop_still_closes_transport_when_mouse_release_fails(controlled) -> None:
    server, backend = controlled
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x83, entrada())
        read_message(client)
        send_json(client, 0x83, entrada(trigger=True))
        backend.wait("down")
        original_release = backend.mouse_up

        def failed_release() -> None:
            raise OSError("injected release failure")

        backend.mouse_up = failed_release
        try:
            with pytest.raises(RuntimeError, match="injected release failure"):
                server.stop()
            assert server._connection is None and server._listener is None and not server._threads
        finally:
            backend.mouse_up = original_release
        server.stop()
        assert backend.events.count("up") == 1
