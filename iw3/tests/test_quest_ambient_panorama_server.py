from types import SimpleNamespace

import pytest
import torch

from iw3.desktop import quest_ambient, quest_server

from .test_quest_ambient_panorama import rgba
from .test_quest_ambient_server import read_layers
from .test_quest_server import EncoderDouble, connect, read_message


def test_paused_animation_changes_only_rgba_and_never_reads_source_again(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(quest_ambient, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(quest_server, "QuestEncoder", EncoderDouble)
    server = quest_server.QuestServer(
        "127.0.0.1",
        72,
        64,
        30,
        35,
        audio=False,
        port=0,
        discovery_port=0,
        ambient_enabled=True,
        ambient_style="aurora",
        ambient_360=True,
    )
    server.start()
    try:
        with connect(server) as client:
            assert read_message(client)[0] == 0x01
            read_layers(client)
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            frame = torch.ones(3, 64, 36)
            server.set_ambient_frame(frame, sbs=torch.ones(3, 64, 72))
            server.set_frame_data((torch.zeros(3, 64, 72), 1))
            assert read_message(client)[0] == 0x02
            kind, first = read_message(client)
            assert kind == 0x10 and rgba(first).shape == (316, 320, 4)
            assert len(read_layers(client)) == 9
            monkeypatch.setattr(
                quest_ambient.F, "interpolate", lambda *_, **__: pytest.fail("paused animation resampled source")
            )
            clock[0] = 2
            server.refresh_ambient_frame()
            kind, second = read_message(client)
            assert kind == 0x10 and first != second
            server.discard_video_frames()
            assert len(read_layers(client)) == 2
            clock[0] = 4
            server.refresh_ambient_frame()
            assert server._ambient_pending is None
            server.check_error()
    finally:
        server.stop()


def test_disable_and_reconnect_cannot_reuse_old_six_face_texture(monkeypatch) -> None:
    monkeypatch.setattr(quest_server, "QuestEncoder", EncoderDouble)
    server = quest_server.QuestServer(
        "127.0.0.1",
        128,
        64,
        30,
        35,
        audio=False,
        port=0,
        discovery_port=0,
        ambient_enabled=True,
        ambient_style="stars",
        ambient_360=True,
    )
    server.start()
    try:
        with connect(server) as client:
            assert read_message(client)[0] == 0x01
            read_layers(client)
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            server.set_ambient_frame(torch.ones(3, 64, 64))
            server.set_frame_data((torch.zeros(3, 64, 128), 1))
            assert read_message(client)[0] == 0x02
            assert read_message(client)[0] == 0x10
            assert len(read_layers(client)) == 9
            server.configure_ambient(False, 0.25, 2, style="stars", panoramic=True)
            assert len(read_layers(client)) == 2
            server.refresh_ambient_frame()
            assert server._ambient_pending is None
        with server._condition:
            assert server._condition.wait_for(lambda: server._connection is None, timeout=1)
        with connect(server) as client:
            assert read_message(client)[0] == 0x01
            assert len(read_layers(client)) == 2
            server.refresh_ambient_frame()
            assert server._ambient_pending is None
        server.check_error()
    finally:
        server.stop()
