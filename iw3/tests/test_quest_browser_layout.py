import math
import socket
import threading

import pytest
import torch

from iw3.desktop import quest_server
from iw3.desktop.quest_input import ray_to_uv

from .test_quest_ambient import background, pixels
from .test_quest_ambient_server import main_video, read_layers
from .test_quest_server import EncoderDouble, connect, initial_messages, read_message, send_json


@pytest.mark.parametrize("width,height,size", [(128, 36, (3.6, 2.025)), (72, 64, (1.1390625, 2.025))])
def test_browser_reference_keeps_viewing_height_without_padding_video(width, height, size):
    server = quest_server.QuestServer("127.0.0.1", width, height, 30, 35, reference_aspect=16 / 9)
    left = main_video(server._layers()["camadas"])
    assert left["tam"] == pytest.approx(size)
    assert left["rect"] == [0, 0, width // 2, height]
    assert server._sent_quad["tam"] == pytest.approx((3.6, 2.025))
    # 已发送窄视频不能反推参考幕面，否则每次重发都会缩小。
    for _ in range(3):
        layers = server._layers((server._sent_quad["pos"], server._sent_quad["rot"]), server._sent_quad["tam"])
        assert main_video(layers["camadas"])["tam"] == pytest.approx(size)


def test_portrait_curves_share_reference_radius_and_pointer_surface():
    server = quest_server.QuestServer(
        "127.0.0.1", 72, 64, 30, 35, reference_aspect=16 / 9, screen_curvature=100, ambient_spread=4
    )
    server._cylinder_supported = server._ambient_visible = True
    left, right, ambient = server._layers()["camadas"]
    radius = 3.6 / math.radians(100)
    assert left["raio"] == right["raio"] == ambient["raio"] == pytest.approx(radius)
    assert left["pos"] == right["pos"] == ambient["pos"]
    assert left["angulo"] == pytest.approx(math.radians(31.640625))
    assert ambient["angulo"] == pytest.approx(math.radians(170))
    assert left["raio"] * left["angulo"] / left["aspecto"] == pytest.approx(2.025)
    assert ambient["raio"] * ambient["angulo"] / ambient["aspecto"] == pytest.approx(8.1)
    assert ray_to_uv({"pos": [0, 0, 0], "rot": [0, 0, 0, 1]}, left) == pytest.approx((0.5, 0.5))


def test_portrait_tiny_curve_falls_back_before_emitting_invalid_native_arc():
    server = quest_server.QuestServer("127.0.0.1", 72, 64, 30, 35, reference_aspect=16 / 9, screen_curvature=0.1)
    server._cylinder_supported = server._ambient_visible = True
    assert all(layer["tipo"] == "quad" for layer in server._layers()["camadas"])


def test_portrait_ambient_fills_reference_gaps_on_the_first_paused_frame():
    frame = torch.ones(3, 64, 36)
    rgba = pixels(background().update(frame, 0.67, tick=0, video_rect=(0.341796875, 0, 0.658203125, 1)))
    # spread=2时参考u=.1/.9映射为纹理x=.3/.7；纯视频中心不能被填光遮住。
    assert rgba[90, 96, 3] == rgba[90, 223, 3] == 255
    assert rgba[90, 160, 3] == 0
    assert rgba[90, 96, :3].tolist() == [171, 171, 171]


@pytest.mark.parametrize("value", [True, 0, -1, math.inf, math.nan, "16:9"])
def test_reference_aspect_rejects_invalid_values_before_changing_layout(value):
    with pytest.raises(ValueError, match="reference aspect"):
        quest_server.QuestServer("127.0.0.1", 128, 64, 30, 35, reference_aspect=value)
    server = quest_server.QuestServer("127.0.0.1", 128, 64, 30, 35)
    with pytest.raises(ValueError, match="reference aspect"):
        server.set_reference_aspect(value)
    assert server._reference_aspect is None


@pytest.mark.parametrize("rect", [(0, 0, 0, 1), (-1, 0, 1, 1), (0, 0, 1, math.nan), [0, 0, 1, 1]])
def test_ambient_rejects_invalid_reference_video_bounds(rect):
    with pytest.raises(ValueError, match="video_rect"):
        background().update(torch.ones(3, 16, 16), 0.5, video_rect=rect)


def test_video_reconfiguration_preserves_reference_and_source_switch_restores_screenshot(monkeypatch):
    monkeypatch.setattr(quest_server, "QuestEncoder", EncoderDouble)
    server = quest_server.QuestServer(
        "127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0, reference_aspect=16 / 9
    )
    server.start()
    try:
        with connect(server) as client:
            initial_messages(client)
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            server.configure_video(72, 64, 30, 35)
            assert read_message(client)[0] == 0x01
            assert main_video(read_layers(client))["tam"] == pytest.approx((1.1390625, 2.025))
            assert server._sent_quad["tam"] == pytest.approx((3.6, 2.025))
            server.configure_video(128, 36, 30, 35)
            assert read_message(client)[0] == 0x01
            assert main_video(read_layers(client))["tam"] == pytest.approx((3.6, 2.025))
            server.set_reference_aspect(None)
            assert main_video(read_layers(client))["tam"] == pytest.approx((3.6, 2.025))
            server.configure_video(128, 64, 30, 35)
            assert read_message(client)[0] == 0x01
            assert main_video(read_layers(client))["tam"] == pytest.approx((3.6, 3.6))
    finally:
        server.stop()


def test_paused_cache_handles_off_on_and_curvature_but_never_reconnects_old_colors(monkeypatch):
    monkeypatch.setattr(quest_server, "QuestEncoder", EncoderDouble)
    server = quest_server.QuestServer(
        "127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0, ambient_enabled=True
    )
    server.start()
    try:
        with connect(server) as client:
            initial_messages(client)
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            send_json(client, 0x84, {"cilindro": True})
            read_layers(client)
            server.configure_ambient(False, 0.25, 2)
            read_layers(client)
            # 关闭背景时仍更新有效输入缓存；开启时无需再送视频帧。
            server.set_ambient_frame(torch.ones(3, 64, 64), sbs=torch.ones(3, 64, 128))
            server.configure_ambient(True, 0.25, 2)
            read_layers(client)
            server.refresh_ambient_frame()
            assert read_message(client)[0] == 0x10
            assert len(read_layers(client)) == 3
            server.set_screen_curvature(100)
            assert len(read_layers(client)) == 2
            server.refresh_ambient_frame()
            assert read_message(client)[0] == 0x10
            assert read_layers(client)[-1]["tipo"] == "cilindro"
        with server._condition:
            assert server._condition.wait_for(lambda: server._connection is None, timeout=1)
        assert server._ambient_frame is None
        with connect(server) as client:
            initial_messages(client)
            server.configure_ambient(True, 0.9, 2)
            read_layers(client)
            server.refresh_ambient_frame()
            client.settimeout(0.1)
            with pytest.raises(socket.timeout):
                read_message(client)
    finally:
        server.stop()


def test_paused_ambient_refresh_sends_rgba_without_encoding_or_consuming_idr(monkeypatch):
    calls = []

    class Encoder(EncoderDouble):
        def encode(self, frame, pts, force_idr=False):
            calls.append(force_idr)
            return super().encode(frame, pts, force_idr)

    monkeypatch.setattr(quest_server, "QuestEncoder", Encoder)
    server = quest_server.QuestServer(
        "127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0, ambient_enabled=True
    )
    server.start()
    try:
        with connect(server) as client:
            initial_messages(client)
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            frame, stereo = torch.ones(3, 64, 64), torch.ones(3, 64, 128)
            server.set_ambient_frame(frame, sbs=stereo)
            server.set_frame_data((stereo, 1))
            assert read_message(client)[0] == 0x02
            assert read_message(client)[0] == 0x10
            read_layers(client)
            server.configure_ambient(True, 0.8, 3)
            assert len(read_layers(client)) == 2
            with server._condition:
                server._need_idr = True
            server.refresh_ambient_frame()
            assert read_message(client)[0] == 0x10
            assert len(read_layers(client)) == 3
            assert len(calls) == 1 and server._need_idr
            server.refresh_ambient_frame()
            client.settimeout(0.1)
            with pytest.raises(socket.timeout):
                read_message(client)
            server.check_error()
    finally:
        server.stop()


def test_source_invalidation_discards_slow_encoded_frame_and_ambient_cache(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    idrs = []

    class SlowEncoder(EncoderDouble):
        def encode(self, frame, pts, force_idr=False):
            idrs.append(force_idr)
            if len(idrs) == 1:
                entered.set()
                assert release.wait(2)
            return super().encode(frame, pts, force_idr)

    monkeypatch.setattr(quest_server, "QuestEncoder", SlowEncoder)
    server = quest_server.QuestServer(
        "127.0.0.1", 128, 64, 30, 35, audio=False, port=0, discovery_port=0, ambient_enabled=True
    )
    server.start()
    try:
        with connect(server) as client:
            initial_messages(client)
            with server._condition:
                assert server._condition.wait_for(lambda: server._connection is not None, timeout=1)
            server.set_ambient_frame(torch.ones(3, 64, 64))
            server.set_frame_data((torch.ones(3, 64, 128), 1))
            assert entered.wait(1)
            previous = server._generation
            server.discard_video_frames()
            assert server._generation > previous and server._video_enabled
            server.refresh_ambient_frame()
            release.set()
            server.set_frame_data((torch.zeros(3, 64, 128), 2))
            kind, payload = read_message(client)
            assert kind == 0x02 and payload[-1] == 0
            assert idrs == [True, True]
            client.settimeout(0.1)
            with pytest.raises(socket.timeout):
                read_message(client)
    finally:
        release.set()
        server.stop()
