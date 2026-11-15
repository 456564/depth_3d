import threading
from argparse import Namespace

import pytest

from iw3.desktop.quest_session import QuestSession, SessionInterrupted


def make_session():
    args = Namespace(
        divergence=1.0,
        convergence=0.5,
        foreground_scale=0.0,
        edge_dilation=2,
        quest_audio_gain=1.0,
        depth_model="Any_V2_S",
        method="row_flow_v3_sym",
        screenshot="wc_mp",
        window_name="Browser",
        quest_control=False,
        pad_mode=None,
        gpu=[0],
        stream_height=1080,
        stream_fps=30,
        quest_bitrate=35,
        quest=True,
        state={"stop_event": threading.Event(), "fps_event": None},
    )
    return QuestSession(args)


def test_restart_request_interrupts_waiting_capture_without_stopping_management():
    session = make_session()
    session.submit("restart_stream", {})
    capture = Namespace(frame_set_event=threading.Event())
    with pytest.raises(SessionInterrupted):
        session.read_frame(capture)
    assert not session.stop_event.is_set()


def test_settings_requests_are_validated_and_bounded():
    session = make_session()
    with pytest.raises(ValueError):
        session.submit("set", {"quest_audio_gain": float("nan")})
    with pytest.raises(ValueError):
        session.submit("unknown", {})
    for index in range(100):
        session.submit("set", {"quest_audio_gain": index / 100})
    assert session.take_live_patch() == {"quest_audio_gain": 0.99}
    assert session.take_live_patch() == {}


def test_snapshot_is_owned_and_pending_configuration_is_not_claimed_applied():
    session = make_session()
    session.submit("apply", {"stream_fps": 60})
    snapshot = session.snapshot()
    assert snapshot["values"]["stream_fps"] == 30
    snapshot["values"]["gpu"].append(9)
    assert session.snapshot()["values"]["gpu"] == [0]


def test_frame_wait_checks_external_stop():
    session = make_session()
    session.stop_event.set()
    with pytest.raises(SessionInterrupted):
        session.read_frame(Namespace(frame_set_event=threading.Event()))


def test_failed_configuration_rolls_back_once_without_restarting_management(monkeypatch):
    from iw3.desktop import quest_server, utils

    session = make_session()
    session.args.bind_addr = "127.0.0.1"
    session.args.quest_bitrate = 35
    events = []

    class Server:
        def __init__(self, *args, **kwargs):
            events.append("create")

        def set_menu(self, menu):
            pass

        def start(self):
            events.append("start")

        def pause_media(self):
            events.append("pause")

        def set_controller_factory(self, factory):
            pass

        def stop(self):
            events.append("stop")

        def take_media_error(self):
            return None

    monkeypatch.setattr(quest_server, "QuestServer", Server)
    monkeypatch.setattr(session, "refresh_choices", lambda: None)

    def prepare(args, **kwargs):
        args.state = dict(stop_event=kwargs["stop_event"], depth_model=None, convergence_model=None)

    monkeypatch.setattr(utils, "set_state_args", prepare)
    seen = []

    def pipeline(args, **kwargs):
        seen.append(args.stream_fps)
        if len(seen) == 1:
            session.media_ready()
            session.submit("apply", {"stream_fps": 60})
        elif len(seen) == 2:
            raise RuntimeError("new configuration failed")
        else:
            session.media_ready()
            session.stop_event.set()

    session.run(pipeline, False)
    assert seen == [30, 60, 30]
    assert events.count("create") == events.count("start") == events.count("stop") == 1
    assert session.snapshot()["values"]["stream_fps"] == 30


def test_refresh_and_recenter_do_not_wait_for_media_restart(monkeypatch):
    session = make_session()
    seen = []
    session.server = Namespace(recenter=lambda: seen.append("recenter"))
    monkeypatch.setattr(session, "refresh_choices", lambda: seen.append("refresh"))
    session.submit("refresh_windows", {})
    session.apply_live()
    session.submit("recenter", {})
    session.apply_live()
    assert seen == ["refresh", "recenter"]


def test_refresh_does_not_replace_a_pending_apply_request(monkeypatch):
    session = make_session()
    monkeypatch.setattr(session, "refresh_choices", lambda: None)
    session.submit("apply", {"stream_fps": 60})
    session.submit("refresh_windows", {})
    session.apply_live()
    assert session._request == ("apply", {"stream_fps": 60})


def test_audio_gain_updates_the_live_source_without_a_capture_frame():
    session = make_session()
    gains = []
    session.server = Namespace(set_audio_gain=gains.append)
    session.submit("set", {"quest_audio_gain": 2.0})
    session.apply_live()
    assert gains == [2.0]
    assert session.snapshot()["values"]["quest_audio_gain"] == 2.0


def test_live_edit_during_loading_does_not_publish_pending_configuration():
    session = make_session()
    session.args.stream_fps = 60
    session.args.depth_model = "DepthPro"
    session.submit("set", {"convergence": 0.7})
    session.apply_live()
    values = session.snapshot()["values"]
    assert values["convergence"] == 0.7
    assert values["stream_fps"] == 30
    assert values["depth_model"] == "Any_V2_S"


def test_frame_wait_reports_media_errors_without_a_capture_frame():
    session = make_session()

    def check_error():
        raise RuntimeError("audio source failed")

    session.server = Namespace(check_error=check_error)
    capture = Namespace(frame_set_event=threading.Event(), is_alive=lambda: False)
    with pytest.raises(RuntimeError, match="audio source failed"):
        session.read_frame(capture)


def test_stats_reach_the_menu_without_entering_restorable_settings():
    session = make_session()
    session.set_stats(output_size=(3840, 1080), estimated_fps=30.0, bind_addr="127.0.0.1")
    values = session.snapshot()["values"]
    assert values["output_size"] == (3840, 1080)
    assert values["estimated_fps"] == 30.0
    assert values["bind_addr"] == "127.0.0.1"
    assert "output_size" not in session._values
    assert "estimated_fps" not in session._last_good


def test_rebuild_uses_the_requested_mapper_instead_of_the_previous_resolved_mapper(monkeypatch):
    from iw3.desktop import quest_server, utils
    from iw3.mapper import resolve_mapper_name

    session = make_session()
    session.args.bind_addr = "127.0.0.1"
    session.args.mapper = None
    monkeypatch.setattr(session, "refresh_choices", lambda: None)

    class Server:
        def __init__(self, *args, **kwargs):
            pass

        def set_menu(self, menu):
            pass

        def start(self):
            pass

        def pause_media(self):
            pass

        def set_controller_factory(self, factory):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(quest_server, "QuestServer", Server)

    def prepare(args, **kwargs):
        args.state = dict(stop_event=kwargs["stop_event"], depth_model=None, convergence_model=None)

    monkeypatch.setattr(utils, "set_state_args", prepare)
    seen = []

    def pipeline(args, **kwargs):
        metric = args.depth_model == "DepthPro"
        args.mapper = resolve_mapper_name(args.mapper, args.foreground_scale, metric)
        seen.append(args.mapper)
        session.media_ready()
        if len(seen) == 1:
            session.submit("apply", {"depth_model": "DepthPro"})
        else:
            session.stop_event.set()

    session.run(pipeline, False)
    assert seen == ["none", "div_6"]


def test_capture_choices_exclude_uninstalled_windows_backends(monkeypatch):
    import importlib.util

    from iw3.desktop import utils

    session = make_session()
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name in {"windows_capture", "wc_cuda"} else original_find_spec(name, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(utils, "get_monitor_size_list", lambda: [(1920, 1080)])
    monkeypatch.setattr(utils, "enum_window_names", lambda: [])
    session.refresh_choices()
    choices = session.snapshot()["choices"]["screenshot"]
    assert "pil" in choices
    assert "wc_mp" not in choices
    assert "wc_cuda" not in choices


def test_live_settings_are_only_restorable_after_a_successful_media_frame():
    session = make_session()
    session.media_ready()
    session.submit("set", {"convergence": 0.7})
    session.apply_live()
    assert session._last_good["convergence"] == 0.5
    session.media_ready()
    assert session._last_good["convergence"] == 0.7


@pytest.mark.parametrize("patch", [{"divergence": 5.0}, {"foreground_scale": 1.0}])
def test_failed_live_model_switch_restores_successful_settings_and_clears_controller(monkeypatch, patch):
    from iw3.desktop import quest_server, utils

    session = make_session()
    session.args.bind_addr = "127.0.0.1"
    session.args.mapper = "div_6"
    session = QuestSession(session.args)
    monkeypatch.setattr(session, "refresh_choices", lambda: None)
    controller_factories = []

    class Server:
        def __init__(self, *args, **kwargs):
            pass

        def set_menu(self, menu):
            pass

        def start(self):
            pass

        def pause_media(self):
            pass

        def set_controller_factory(self, factory):
            controller_factories.append(factory)

        def stop(self):
            pass

        def take_media_error(self):
            return None

    monkeypatch.setattr(quest_server, "QuestServer", Server)

    def prepare(args, **kwargs):
        args.state = dict(stop_event=kwargs["stop_event"], depth_model=None, convergence_model=None)

    monkeypatch.setattr(utils, "set_state_args", prepare)
    seen = []

    def pipeline(args, **kwargs):
        seen.append((args.divergence, args.foreground_scale, args.mapper))
        session.media_ready()
        if len(seen) == 1:
            session.submit("set", patch)
            session.apply_live()
            raise RuntimeError("new MLBW level failed")
        session.stop_event.set()

    def error_status(status, message):
        if status == "error" and session.values(session.args) == session._last_good:
            session.stop_event.set()

    monkeypatch.setattr(session, "status", error_status)
    session.run(pipeline, False)
    assert seen == [(1.0, 0.0, "div_6"), (1.0, 0.0, "div_6")]
    assert controller_factories == [None, None]


def test_stream_restart_reports_loading_and_two_successes_on_the_same_server(monkeypatch):
    from iw3.desktop import quest_server, utils

    session = make_session()
    session.args.bind_addr = "127.0.0.1"
    events = []
    server = Namespace(
        set_menu=lambda menu: None,
        start=lambda: events.append("start"),
        pause_media=lambda: None,
        set_controller_factory=lambda factory: None,
        stop=lambda: events.append("stop"),
    )

    def create_server(*args, **kwargs):
        events.append("create")
        return server

    monkeypatch.setattr(quest_server, "QuestServer", create_server)
    monkeypatch.setattr(session, "refresh_choices", lambda: None)
    monkeypatch.setattr(utils, "set_state_args", lambda *args, **kwargs: None)
    messages = []
    loading_states = []

    def pipeline(args, **kwargs):
        loading_states.append(session.snapshot())
        session.media_ready()
        session.media_ready()
        messages.append(session.snapshot()["message"])
        if len(messages) < 3:
            session.submit("restart_stream", {})
        else:
            session.stop_event.set()

    session.run(pipeline, False)
    assert messages[0] == "设置已在电脑端应用；请在头显确认实际画面。"
    for loading in loading_states[1:]:
        assert loading["status"] == "loading"
        assert "正在重启串流" in loading["message"]
        assert "串流已重启" not in loading["message"]
    assert "串流已重启（第 1 次）" in messages[1]
    assert "串流已重启（第 2 次）" in messages[2]
    assert events == ["create", "start", "stop"]


@pytest.mark.parametrize("rollback", [False, True])
def test_failed_stream_restart_does_not_count_recovery_or_a_later_apply_as_a_restart(monkeypatch, rollback):
    from iw3.desktop import quest_server, utils

    session = make_session()
    session.args.bind_addr = "127.0.0.1"
    server = Namespace(
        set_menu=lambda menu: None,
        start=lambda: None,
        pause_media=lambda: None,
        set_controller_factory=lambda factory: None,
        stop=lambda: None,
        take_media_error=lambda: None,
    )
    monkeypatch.setattr(quest_server, "QuestServer", lambda *args, **kwargs: server)
    monkeypatch.setattr(session, "refresh_choices", lambda: None)
    monkeypatch.setattr(utils, "set_state_args", lambda *args, **kwargs: None)
    original_status = session.status

    def status(state, message):
        original_status(state, message)
        if state == "error" and not rollback:
            session.submit("apply", {"stream_fps": 24})

    monkeypatch.setattr(session, "status", status)
    attempts = []
    messages = []

    def pipeline(args, **kwargs):
        attempts.append(args.stream_fps)
        if len(attempts) == 2:
            if rollback:
                args.stream_fps = 60
            raise RuntimeError("restart failed before first frame")
        session.media_ready()
        messages.append(session.snapshot()["message"])
        if len(attempts) == 3:
            assert "串流已重启" not in messages[-1]
            if rollback:
                assert "已恢复上次成功配置" in messages[-1]
            else:
                assert messages[-1] == "设置已在电脑端应用；请在头显确认实际画面。"
        if len(attempts) < 4:
            session.submit("restart_stream", {})
        else:
            session.stop_event.set()

    session.run(pipeline, False)
    assert len(attempts) == 4
    assert "串流已重启（第 1 次）" in messages[-1]
