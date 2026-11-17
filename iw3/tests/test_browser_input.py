import importlib

import pytest


class Bridge:
    def __init__(self):
        self.generation = "video-a"
        self.fresh = True
        self.commands = []

    def status(self):
        return {"generation": self.generation, "fresh": self.fresh}

    def queue_command(self, name, generation=None):
        if not self.fresh or generation != self.generation:
            return False
        self.commands.append((name, generation))
        return True


def backend():
    bridge = Bridge()
    return importlib.import_module("iw3.desktop.browser_input").BrowserInput(bridge), bridge


def test_media_click_is_one_semantic_pause_without_desktop_input():
    control, bridge = backend()
    assert control.move(0.2, 0.8)
    assert control.mouse_down() and control.mouse_down()
    control.mouse_up()
    assert bridge.commands == [("pause_toggle", "video-a")]
    assert control.mouse_down()
    control.mouse_up()
    assert len(bridge.commands) == 2


@pytest.mark.parametrize("key,action", [("up", "previous"), ("down", "next"), ("escape", "exit_fullscreen")])
def test_keys_only_queue_whitelisted_media_actions(key, action):
    control, bridge = backend()
    assert control.move(0.5, 0.5) and control.tap_key(key)
    assert bridge.commands == [(action, "video-a")]
    with pytest.raises(ValueError):
        control.tap_key("enter")


@pytest.mark.parametrize("changes", [{"fresh": False}, {"generation": "video-b"}])
def test_changed_or_expired_media_cannot_receive_a_previous_target_action(changes):
    control, bridge = backend()
    assert control.move(0.5, 0.5)
    for key, value in changes.items():
        setattr(bridge, key, value)
    assert not control.mouse_down() and not control.tap_key("down")
    control.mouse_up()
    control.release_keys()
    assert not bridge.commands


@pytest.mark.parametrize("uv", [(float("nan"), 0), (-0.1, 0.5), (0.5, 1.1), (True, 0.5)])
def test_invalid_pointer_uv_is_rejected(uv):
    control, bridge = backend()
    assert not control.move(*uv)
    assert not control.mouse_down()
    assert not bridge.commands


def test_reset_removes_media_action_target_without_toggling_playback():
    control, bridge = backend()
    assert control.move(0.5, 0.5)
    control.release_keys()
    assert not control.tap_key("escape")
    control.mouse_up()
    assert not bridge.commands
