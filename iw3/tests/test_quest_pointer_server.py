import json
import math
import select
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from iw3.desktop.quest_input import QuestController

from .test_quest_control_server import RecordedInput
from .test_quest_reconfigure import MenuDouble
from .test_quest_screen_server import packet
from .test_quest_server import connect, initial_messages, read_message, send_json
from .test_quest_server import module as module


class PointerInput(RecordedInput):
    allowed = True

    def move(self, u: float, v: float) -> bool:
        self.record((u, v))
        return self.allowed


@pytest.fixture
def pointed(module):
    backend = PointerInput()
    server = module.QuestServer(
        "127.0.0.1",
        128,
        64,
        30,
        35,
        audio=False,
        port=0,
        discovery_port=0,
        controller_factory=lambda: QuestController(backend),
        ambient_enabled=True,
    )
    server.start()
    try:
        yield server, backend
    finally:
        server.stop()


def layers_until(client: socket.socket, present: bool) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        kind, payload = read_message(client)
        if kind == 0x11:
            layers = json.loads(payload)["camadas"]
            if any(layer["tipo"] == "laser" for layer in layers) is present:
                return layers
    pytest.fail("pointer layers did not arrive")


def start_pointer(server, client):
    initial_messages(client)
    send_json(client, 0x83, packet())
    layers = layers_until(client, True)
    assert layers[-1] == {"tipo": "laser", "mao": "dir", "ponto": [0.0, 0.0, -3.0]}
    return layers


def test_native_pointer_follows_valid_input_and_stays_above_ambient(pointed) -> None:
    server, backend = pointed
    with connect(server) as client:
        start_pointer(server, client)
        backend.wait((0.5, 0.5))
        server.set_ambient_frame(torch.ones(3, 64, 64))
        server.set_frame_data((torch.zeros(3, 64, 128), 1.0))
        layers = layers_until(client, True)
        assert [layer.get("tex", layer["tipo"]) for layer in layers] == ["video", "video", "iw3_ambient", "laser"]
        server.configure_ambient(True, 0.3, 2)
        assert layers_until(client, True)[-1]["tipo"] == "laser"
    server.check_error()


@pytest.mark.parametrize("blocked", ["focus", "miss", "backend", "grip", "malformed"])
def test_pointer_clear_is_immediate_and_releases_owned_input(pointed, blocked: str) -> None:
    server, backend = pointed
    with connect(server) as client:
        start_pointer(server, client)
        send_json(client, 0x83, packet(trigger=True))
        backend.wait("down")
        data: Any = packet(trigger=True)
        if blocked == "focus":
            data["estado"] = "visivel"
        elif blocked == "miss":
            data["maos"]["dir"]["pos"] = [10, 0, 0]
        elif blocked == "backend":
            backend.allowed = False
        elif blocked == "grip":
            data["maos"]["dir"]["aperto"] = True
        else:
            data = []
        send_json(client, 0x83, data)
        assert all(layer["tipo"] != "laser" for layer in layers_until(client, False))
        backend.wait("up")
        assert server._pointer is None
    server.check_error()


def test_menu_replaces_video_pointer_with_its_own_laser(pointed) -> None:
    server, backend = pointed
    menu = MenuDouble()
    server.set_menu(menu)
    with connect(server) as client:
        start_pointer(server, client)
        data = packet()
        data["show"] = True
        send_json(client, 0x83, data)
        assert read_message(client) == (0x10, b"menu payload")
        layers = layers_until(client, True)
        assert [layer.get("tex", layer["tipo"]) for layer in layers] == ["video", "video", "iw3_settings", "laser"]
        assert "ponto" not in layers[-1] and server._pointer is None
    server.check_error()


def test_pointer_expires_on_a_live_connection_without_new_input(pointed) -> None:
    server, _ = pointed
    with connect(server) as client:
        start_pointer(server, client)
        started = time.monotonic()
        layers_until(client, False)
        assert time.monotonic() - started < 0.7
        assert server._pointer is None
    server.check_error()


def test_pointer_is_not_replayed_after_disconnect(pointed) -> None:
    server, _ = pointed
    with connect(server) as first:
        start_pointer(server, first)
    with server._condition:
        assert server._condition.wait_for(lambda: server._connection is None, timeout=1)
    with connect(server) as second:
        assert all(layer["tipo"] != "laser" for layer in initial_messages(second)["camadas"])
        assert server._pointer is None


def test_recenter_clears_the_sent_pointer_without_waiting_for_another_input(pointed) -> None:
    server, _ = pointed
    with connect(server) as client:
        start_pointer(server, client)
        server.recenter()
        assert all(layer["tipo"] != "laser" for layer in layers_until(client, False))
        assert server._pointer is None and server._last_input is None
        assert server._sent_quad["pos"] == [0, 0, -3]
    server.check_error()


def test_pausing_media_clears_pointer_and_blocks_pc_moves_until_resumed(pointed) -> None:
    server, backend = pointed
    with connect(server) as client:
        start_pointer(server, client)
        server.pause_media()
        layers_until(client, False)
        moves = len(backend.events)
        send_json(client, 0x83, packet(position=[0.2, 0, 0]))
        threading.Event().wait(0.05)
        assert len(backend.events) == moves and server._pointer is None
    server.check_error()


def test_pointer_updates_are_limited_to_30hz_but_clear_is_not(pointed, module, monkeypatch) -> None:
    server, backend = pointed
    now = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    with connect(server) as client:
        start_pointer(server, client)
        now[0] += 0.001
        send_json(client, 0x83, packet(position=[0.4, 0, 0]))
        backend.wait((0.5 + 0.4 / 3.6, 0.5))
        assert not select.select([client], [], [], 0.01)[0]
        now[0] += 0.034
        send_json(client, 0x83, packet(position=[0.4, 0, 0]))
        assert layers_until(client, True)[-1]["ponto"] == pytest.approx([0.4, 0, -3])
        now[0] += 0.001
        data = packet()
        data["estado"] = "visivel"
        send_json(client, 0x83, data)
        layers_until(client, False)
        assert server._pointer is None


def test_curved_pointer_world_point_matches_the_sent_surface_and_pc_uv(pointed) -> None:
    server, backend = pointed
    server.set_screen_curvature(60)
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x84, {"cilindro": True})
        assert read_message(client)[0] == 0x11
        send_json(client, 0x83, packet(position=[0.4, 0, 0]))
        layers = layers_until(client, True)
        video, pointer = layers[0], layers[-1]
        radius, angle = video["raio"], video["angulo"]
        assert pointer["ponto"] == pytest.approx([0.4, 0, video["pos"][2] - math.sqrt(radius * radius - 0.16)])
        backend.wait((0.5 + math.asin(0.4 / radius) / angle, 0.5))
