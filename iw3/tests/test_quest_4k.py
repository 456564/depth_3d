from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from iw3.desktop.quest_4k import playing_4k


def test_4k_cadence_requires_current_playing_video():
    reader = NS(width=3840, height=2160, fps=60.0, live=False, _uses_fused_color=True)
    state = NS(paused=False, playback_rate=1.0, source={"kind": "vod"}, position_at=lambda _: 1.0)
    media = NS(_reader=reader, snapshot=lambda: state)
    token = (1, "valid")
    capture = NS(source_kind="browser", _source=media, is_current_generation=lambda value: value == token)
    assert playing_4k(capture, token)
    for target, field, invalid in (
        (capture, "source_kind", "desktop"),
        (media, "_reader", None),
        (reader, "height", 1080),
        (reader, "fps", 30),
        (reader, "live", True),
        (reader, "_uses_fused_color", False),
        (state, "paused", True),
        (state, "playback_rate", 2),
        (state, "source", {"kind": "live"}),
        (state, "position_at", lambda _: None),
    ):
        saved = getattr(target, field)
        setattr(target, field, invalid)
        assert not playing_4k(capture, token), field
        setattr(target, field, saved)
    assert not playing_4k(capture, None)
    assert not playing_4k(capture, (0, "stale"))


@pytest.mark.parametrize("paused", (False, True))
def test_busy_slot_wait_stops_at_pause_or_changed_generation(monkeypatch, paused):
    from threading import Condition, Event

    from iw3.desktop import quest_4k

    token = (1, "old")
    reader = NS(width=3840, height=2160, fps=60.0, live=False, _uses_fused_color=True)
    state = NS(paused=False, playback_rate=1.0, source={"kind": "vod"}, position_at=lambda _: 1.0)
    capture = NS(
        source_kind="browser",
        _source=NS(_reader=reader, snapshot=lambda: state),
        get_frame_generation=lambda: token,
        is_current_generation=lambda _: True,
    )
    server = NS(
        _condition=Condition(),
        _connection=object(),
        _video_enabled=True,
        _suspended=False,
        _generation=3,
        _pending=(None, None, None, 3, None),
        _shutdown=Event(),
        width=7680,
        height=2160,
    )
    session = NS(server=server, capture=capture, stop_event=Event(), run_stop=Event())

    def changed(_):
        if paused:
            state.paused = True
        else:
            capture.is_current_generation = lambda _: False

    sleep = Mock(side_effect=changed)
    monotonic = iter((0, 0, 0.00075, 0.02))
    monkeypatch.setattr(quest_4k.time, "perf_counter", lambda: next(monotonic, 0.02))
    monkeypatch.setattr(quest_4k.time, "sleep", sleep)
    quest_4k.wait_for_slot(session, capture)
    assert sleep.call_count == 1
    assert 0 < sleep.call_args.args[0] <= 0.00075


@pytest.mark.hwaccel
def test_4k_graph_matches_current_eager_input_and_preserves_previous_output():
    import sys

    import torch

    from iw3.base_depth_model import BaseDepthModel
    from iw3.depth_model_factory import create_depth_model
    from iw3.desktop.quest_4k import Quest4KForward
    from nunif.utils.video import pyav_init_cuda_primary_context

    if sys.platform != "win32" or not torch.cuda.is_available():
        pytest.skip("4K Quest Graph uses Windows CUDA")
    if not BaseDepthModel.has_model_checkpoint_file("Any_V2_S"):
        pytest.skip("local Any_V2_S checkpoint required")
    pyav_init_cuda_primary_context(max_devices=1)
    device = torch.device("cuda:0")
    depth = create_depth_model("Any_V2_S")
    depth.load(gpu=0, resolution=392)
    x = torch.linspace(0, 1, 2160 * 3840, device=device).reshape(1, 2160, 3840).expand(3, -1, -1)
    y = x.flip((-1,))
    expected_x, expected_y = depth.infer(x), depth.infer(y)
    assert not torch.equal(expected_x, expected_y)
    cadence = Quest4KForward(depth.model, device)
    depth.forward_override = cadence
    cadence.enabled = True
    try:
        first = depth.infer(x)
        second = depth.infer(y)
        assert cadence.used
        assert torch.equal(first, expected_x) and torch.equal(second, expected_y)
        cadence.used = False
        square = x[:, :, :2160]
        other_shape = depth.infer(square)
        assert not cadence.used
        depth.forward_override = None
        assert torch.equal(other_shape, depth.infer(square))
    finally:
        depth.forward_override = None
        cadence.close()


def test_graph_prep_uses_loaded_size_and_optional_auto_switch(monkeypatch):
    import sys
    from threading import Event

    from iw3.depth_models.depth_anything_model import DepthAnythingModel
    from iw3.desktop import quest_session

    if sys.platform != "win32":
        pytest.skip("4K Quest Graph uses Windows CUDA")
    depth = DepthAnythingModel("Any_V2_S")
    depth.model = NS(prep_lower_bound=392)
    args = NS(
        state={"depth_model": depth, "device": NS(type="cuda"), "stop_event": Event()},
        stream_fps=60,
        depth_model="Any_V2_S",
        resolution=None,
        method="row_flow_v3_sym",
        disable_amp=False,
        tta=False,
        compile=False,
        screenshot="pil",
    )
    session = quest_session.QuestSession(args)
    session.server = NS(_cuda_prepare_event=None)
    graph = Mock()
    monkeypatch.setattr(quest_session, "Quest4KForward", graph)
    assert session._prepare_4k() is None
    args.quest_auto_browser = True
    assert session._prepare_4k() is graph.return_value
    graph.assert_called_once_with(depth.model, args.state["device"])
    depth.model.prep_lower_bound = 518
    assert session._prepare_4k() is None
