import importlib.util
import threading
from types import SimpleNamespace

import pytest


class Capture:
    def __init__(self, kind, log):
        self.kind, self.log = kind, log
        self.frame_set_event = threading.Event()
        self.frame_set_event.set()
        self.generation = "video-1"
        self.alive = False

    def start(self):
        assert not any(item.alive for item in self.log)
        self.alive = True
        self.log.append(self)

    def stop(self):
        self.alive = False

    def is_alive(self):
        return self.alive

    def get_frame(self):
        self.frame_set_event.clear()
        return self.kind

    def get_frame_generation(self):
        return self.generation

    def is_current_generation(self, generation):
        return self.alive and generation == self.generation

    def get_content_rect(self):
        return (0, 0, 9, 16)

    def get_frame_mapping(self):
        return {"capture": id(self)}

    def get_fps(self):
        return 30

    def get_status(self):
        return {"status": "running", "decoder": "NVDEC", "hardware_active": True, "source_size": (9, 16)}


def make_router():
    assert importlib.util.find_spec("iw3.desktop.source_router") is not None
    from iw3.desktop import source_router

    state, log, barriers = {"fresh": False, "fullscreen": False}, [], []
    bridge = SimpleNamespace(status=lambda: dict(state))
    router = source_router.SourceRouter(
        bridge,
        lambda: Capture("desktop", log),
        lambda: Capture("browser", log),
        lambda kind, epoch: barriers.append((kind, epoch)),
    )
    return router, state, log, barriers


def test_router_module_provides_auto_fullscreen_capture():
    assert importlib.util.find_spec("iw3.desktop.source_router") is not None


def test_switch_stops_previous_source_keeps_waiting_on_ttl_and_only_explicit_exit_returns_desktop():
    router, state, log, barriers = make_router()
    router.start()
    assert router.source_kind == "desktop" and len(log) == 1
    state.update(fresh=True, fullscreen=True)
    router.poll_route()
    assert router.source_kind == "browser" and not log[0].alive and log[1].alive
    state.update(fresh=False, fullscreen=False)
    router.poll_route()
    assert router.source_kind == "browser" and len(log) == 2
    assert router.get_status()["status"] == "waiting"
    assert router.get_frame() is None
    state.update(fresh=True, fullscreen=False)
    router.poll_route()
    assert router.source_kind == "desktop" and len(log) == 3 and not log[1].alive
    assert router.get_frame_mapping() == {"capture": id(log[-1])}
    assert barriers == [("desktop", 1), ("browser", 2), ("desktop", 3)]
    router.stop()
    assert not any(item.alive for item in log)


def test_route_and_media_generation_tokens_reject_ai_results_after_switch_or_clip_change():
    router, state, log, _ = make_router()
    router.start()
    assert router.get_frame() == "desktop"
    desktop_token = router.get_frame_generation()
    state.update(fresh=True, fullscreen=True)
    assert not router.is_current_generation(desktop_token)
    assert router.get_frame() == "browser"
    token = router.get_frame_generation()
    assert router.is_current_generation(token)
    log[-1].generation = "video-2"
    assert not router.is_current_generation(token)
    state.update(fresh=False)
    assert not router.is_current_generation(router.get_frame_generation())
    router.stop()


def test_status_reports_real_source_and_browser_failure_never_falls_back():
    router, state, log, _ = make_router()
    router.start()
    desktop = router.get_status()
    assert desktop["source_kind"] == "desktop"
    assert desktop["decoder"] == "屏幕捕获" and not desktop["hardware_active"]
    state.update(fresh=True, fullscreen=True)
    router.poll_route()
    log[-1].get_status = lambda: {"status": "error", "message": "NVDEC unavailable", "hardware_active": False}
    status = router.get_status()
    assert status["source_kind"] == "browser" and status["status"] == "error"
    assert router.get_fps() == 0
    assert len(log) == 2
    router.stop()


def test_ttl_invalidates_pending_event_instead_of_busy_polling_stale_browser_frame():
    router, state, log, _ = make_router()
    router.start()
    state.update(fresh=True, fullscreen=True)
    router.poll_route()
    state.update(fresh=False)
    assert log[-1].frame_set_event.is_set()
    assert router.get_frame() is None
    assert not router.frame_set_event.is_set()
    router.stop()


def test_paused_browser_keeps_last_valid_token_without_redecoding():
    router, state, _, _ = make_router()
    router.start()
    state.update(fresh=True, fullscreen=True)
    assert router.get_frame() == "browser"
    token = router.get_frame_generation()
    assert router.get_frame() is None
    assert router.get_frame_generation() == token and router.is_current_generation(token)
    router.stop()


def test_stop_failure_retains_old_capture_for_finally_cleanup_and_does_not_start_second_producer():
    router, state, log, _ = make_router()
    router.start()
    old = log[-1]
    attempts = []

    def stop():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("stop timed out")
        old.alive = False

    old.stop = stop
    state.update(fresh=True, fullscreen=True)
    with pytest.raises(RuntimeError, match="timed out"):
        router.poll_route()
    assert len(log) == 1
    router.stop()
    assert attempts == [True, True] and not old.alive
