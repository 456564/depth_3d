import json
import socket
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from iw3.desktop.quest_input import QuestController
from iw3.desktop.quest_server import QuestServer

from .test_quest_control_server import RecordedInput, entrada
from .test_quest_server import connect, initial_messages, read_message, send_json


def packet(
    hand: str = "dir",
    grip: bool = False,
    position: list[float] | None = None,
    stick: list[float] | None = None,
    trigger: bool = False,
) -> dict[str, Any]:
    data = entrada(trigger=trigger)
    data["maos"][hand].update(aperto=grip, pos=position or [0, 0, 0], analogico=stick or [0, 0])
    return data


def read_layers(client: socket.socket) -> list[dict[str, Any]]:
    kind, payload = read_message(client)
    assert kind == 0x11
    layers = [layer for layer in json.loads(payload)["camadas"] if layer.get("tex") == "video"]
    left, right = layers
    assert left["olho"] == "esq" and right["olho"] == "dir"
    assert left["rect"] == [0, 0, 64, 64] and right["rect"] == [64, 0, 64, 64]
    assert all(left[key] == right[key] for key in ("pos", "rot", "tam"))
    return layers


@pytest.fixture
def screen_server() -> Iterator[QuestServer]:
    server = QuestServer("127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.parametrize("hand", ["esq", "dir"])
def test_either_hand_can_move_screen_without_pc_control(screen_server, hand: str) -> None:
    with connect(screen_server) as client:
        initial_messages(client)
        send_json(client, 0x83, packet())
        read_layers(client)
        send_json(client, 0x83, packet(hand, grip=True))
        threading.Event().wait(0.04)
        send_json(client, 0x83, packet(hand, grip=True, position=[0.2, 0.1, 0]))
        send_json(client, 0x83, packet())
        layers = read_layers(client)
        assert layers[0]["pos"] == pytest.approx([0.2, 0.1, -3])
        assert layers[0]["tam"] == [3.6, 3.6]
    screen_server.check_error()


@pytest.mark.parametrize("hand", ["esq", "dir"])
def test_same_hand_stick_changes_distance_and_size_and_recenter_restores_defaults(screen_server, hand: str) -> None:
    with connect(screen_server) as client:
        initial_messages(client)
        send_json(client, 0x83, packet())
        read_layers(client)
        send_json(client, 0x83, packet(hand, grip=True))
        threading.Event().wait(0.06)
        send_json(client, 0x83, packet(hand, grip=True, stick=[0.9, 0.9]))
        layers = read_layers(client)
        assert layers[0]["pos"][2] < -3
        assert layers[0]["tam"][0] > 3.6
        data = packet()
        data["recentrou"] = True
        send_json(client, 0x83, data)
        layers = read_layers(client)
        assert layers[0]["pos"] == [0, 0, -3]
        assert layers[0]["tam"] == [3.6, 3.6]


def test_screen_grab_releases_pc_mouse_and_does_not_click_or_switch_video() -> None:
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
        with connect(server) as client:
            initial_messages(client)
            send_json(client, 0x83, packet())
            read_layers(client)
            send_json(client, 0x83, packet(trigger=True))
            backend.wait("down")
            send_json(client, 0x83, packet("dir", grip=True, trigger=True))
            backend.wait("up")
            # 抓取先撤下独立射线；这包不改变视频几何。
            assert read_layers(client)[0]["pos"] == [0, 0, -3]
            threading.Event().wait(0.05)
            send_json(client, 0x83, packet("dir", grip=True, position=[0.3, 0, 0], stick=[0.8, -0.9], trigger=True))
            layers = read_layers(client)
            send_json(client, 0x83, packet(trigger=True))
            send_json(client, 0x83, packet(trigger=True))
            backend.wait((0.5 - layers[0]["pos"][0] / layers[0]["tam"][0], 0.5))
            assert backend.events.count("down") == 1
            send_json(client, 0x83, packet())
            send_json(client, 0x83, packet(trigger=True))
            backend.wait("down", count=2)
            assert not any(isinstance(event, tuple) and event[0] == "key" for event in backend.events)
            assert backend.events.count("down") == 2
        server.check_error()
    finally:
        server.stop()


def test_screen_changes_are_not_replayed_after_reconnect(screen_server) -> None:
    with connect(screen_server) as client:
        initial_messages(client)
        send_json(client, 0x83, packet())
        read_layers(client)
        send_json(client, 0x83, packet(grip=True))
        threading.Event().wait(0.04)
        send_json(client, 0x83, packet(grip=True, position=[0.4, 0, 0]))
        send_json(client, 0x83, packet())
        read_layers(client)
    with screen_server._condition:
        assert screen_server._condition.wait_for(lambda: screen_server._connection is None, timeout=1)
    with connect(screen_server) as client:
        layers = initial_messages(client)["camadas"]
        assert layers[0]["pos"] == [0, 0, -3]
        assert layers[0]["tam"] == [3.6, 3.6]


def test_releasing_grip_flushes_final_geometry_inside_layer_rate_limit(screen_server, monkeypatch) -> None:
    from iw3.desktop import quest_server

    monkeypatch.setattr(quest_server, "time", SimpleNamespace(monotonic=lambda: 1.0))
    with connect(screen_server) as client:
        initial_messages(client)
        send_json(client, 0x83, packet())
        read_layers(client)
        send_json(client, 0x83, packet(grip=True))
        send_json(client, 0x83, packet(grip=True, position=[0.25, 0, 0]))
        send_json(client, 0x83, packet())
        assert read_layers(client)[0]["pos"] == pytest.approx([0.25, 0, -3])
