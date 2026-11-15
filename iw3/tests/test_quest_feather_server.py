import math
import threading
from typing import Any

import pytest
import torch

from iw3.desktop import quest_server

from .test_quest_ambient_server import publish, read_layers
from .test_quest_ambient_server import server as server
from .test_quest_server import connect, initial_messages, send_json


@pytest.mark.parametrize(
    "changes",
    [
        {"ambient_fill_borders": 1},
        {"ambient_fill_borders": None},
        {"ambient_feather": True},
        {"ambient_feather": -0.01},
        {"ambient_feather": 0.101},
        {"ambient_feather": math.nan},
        {"ambient_feather": math.inf},
        {"ambient_feather": "0.03"},
    ],
)
def test_feather_configuration_rejects_invalid_values_without_changing_state(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="ambient"):
        quest_server.QuestServer("127.0.0.1", 128, 64, 30, 35, **changes)
    target = quest_server.QuestServer("127.0.0.1", 128, 64, 30, 35)
    values: dict[str, Any] = {"fill_borders": True, "feather": 0.03}
    values.update({key.removeprefix("ambient_"): value for key, value in changes.items()})
    with pytest.raises(ValueError, match="ambient"):
        target.configure_ambient(True, 0.25, 2, **values)
    assert not target._ambient_enabled and target._ambient_generation == 0


@pytest.mark.parametrize(
    "degrees,supported,scale",
    [(0, True, (2, 2)), (30, True, (2, 2)), (120, True, (17 / 12, 2)), (120, False, (2, 2)), (120, None, (2, 2))],
)
def test_final_sbs_and_same_surface_scale_reach_pixel_work_outside_locks(
    server: Any,
    monkeypatch: pytest.MonkeyPatch,
    degrees: float,
    supported: bool | None,
    scale: tuple[float, float],
) -> None:
    original = quest_server.AmbientBackground.update
    source, sbs = torch.ones(3, 64, 64), torch.ones(3, 64, 128)
    seen = threading.Event()

    def observe(background: Any, frame: Any, brightness: float, content_rect: Any, **options: Any) -> bytes | None:
        assert not server._layer_lock._is_owned() and not server._condition._is_owned()
        assert frame is source and options["sbs"] is sbs and content_rect == (8, 4, 56, 60)
        assert options["scale"] == pytest.approx(scale)
        assert options["fill_borders"] is False and options["feather"] == 0.1
        seen.set()
        return original(background, frame, brightness, content_rect, **options)

    monkeypatch.setattr(quest_server.AmbientBackground, "update", observe)
    server.configure_ambient(True, 0.25, 2, False, 0.1)
    server.set_screen_curvature(degrees)
    with connect(server) as client:
        initial_messages(client)
        if supported is not None:
            send_json(client, 0x84, {"cilindro": supported})
            read_layers(client)
        server.set_ambient_frame(source, (8, 4, 56, 60), sbs=sbs)
        assert seen.is_set() and server._ambient_pending is not None


def test_fill_and_feather_change_discard_visible_and_pending_masks(server: Any) -> None:
    with connect(server) as client:
        initial_messages(client)
        publish(server, client)
        previous = server._ambient_generation
        server.configure_ambient(True, 0.25, 2, False, 0)
        assert len(read_layers(client)) == 2
        assert server._ambient_generation > previous
        assert server._ambient_pending is None and server._ambient_background is None
        _, layers = publish(server, client)
        assert layers[-1]["tex"] == "iw3_ambient"


@pytest.mark.parametrize("degrees", [30, 120])
def test_expanded_cylinder_uv_lands_on_the_exact_main_surface(server: Any, degrees: float) -> None:
    server.set_screen_curvature(degrees)
    with connect(server) as client:
        initial_messages(client)
        send_json(client, 0x84, {"cilindro": True})
        read_layers(client)
        _, layers = publish(server, client)
        left, right, ambient = layers
        assert left["pos"] == right["pos"] == ambient["pos"]
        assert left["rot"] == right["rot"] == ambient["rot"]
        assert left["raio"] == right["raio"] == ambient["raio"]
        assert ambient["angulo"] == pytest.approx(math.radians(60 if degrees == 30 else 170))
        main_height = left["raio"] * left["angulo"] / left["aspecto"]
        ambient_height = ambient["raio"] * ambient["angulo"] / ambient["aspecto"]
        assert ambient_height == pytest.approx(main_height * 2)
        for u in (0, 0.25, 0.5, 0.75, 1):
            mapped_u = 0.5 + (u - 0.5) * left["angulo"] / ambient["angulo"]
            main_angle, ambient_angle = (u - 0.5) * left["angulo"], (mapped_u - 0.5) * ambient["angulo"]
            assert math.sin(main_angle) * left["raio"] == pytest.approx(math.sin(ambient_angle) * ambient["raio"])
            assert math.cos(main_angle) * left["raio"] == pytest.approx(math.cos(ambient_angle) * ambient["raio"])
        for v in (0, 0.5, 1):
            mapped_v = 0.5 + (v - 0.5) / 2
            assert (0.5 - v) * main_height == pytest.approx((0.5 - mapped_v) * ambient_height)
