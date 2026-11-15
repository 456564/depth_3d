import struct
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> tuple[SimpleNamespace, MagicMock]:
    manager = MagicMock()
    manager.get_default_wasapi_loopback.return_value = {
        "index": 42,
        "isLoopbackDevice": True,
        "defaultSampleRate": 48000.0,
        "maxInputChannels": 2,
    }
    manager.is_format_supported.return_value = True
    manager.open.return_value.is_active.return_value = True
    module = SimpleNamespace(
        PyAudio=MagicMock(return_value=manager), paInt16=8, paContinue=0, paAbort=2, paInputOverflow=2
    )
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", module)
    return module, manager


def test_start_opens_default_speaker_loopback_only(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    module, manager = backend
    audio = QuestAudio()
    module.PyAudio.assert_not_called()
    audio.start()
    assert audio.sample_rate == 48000
    manager.get_default_wasapi_loopback.assert_called_once_with()
    manager.is_format_supported.assert_called_once_with(48000, input_device=42, input_channels=2, input_format=8)
    options = manager.open.call_args.kwargs
    assert options["input"] is True
    assert options.get("output", False) is False
    assert options["input_device_index"] == 42
    assert options["channels"] == 2
    assert options["rate"] == 48000
    assert options["format"] == 8
    assert options["frames_per_buffer"] == 480
    audio.stop()


def test_process_source_uses_independent_capture_and_applies_quest_gain(monkeypatch, backend) -> None:
    from iw3.desktop import quest_process_audio
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    native = MagicMock()
    native.sample_rate = 48000
    native.get_chunks.return_value = [b"\xe8\x03\x18\xfc"]
    monkeypatch.setattr(quest_process_audio, "ProcessAudio", lambda: native)
    audio = QuestAudio(gain=2, source="process")
    audio.start()
    audio.start()
    native.start.assert_called_once()
    manager.open.assert_not_called()
    assert audio.sample_rate == 48000
    assert audio.get_chunks() == [b"\xd0\x07\x30\xf8"]
    audio.check_error()
    native.check_error.assert_called_once()
    audio.stop()
    audio.stop()
    native.stop.assert_called_once()


def test_invalid_source_fails_without_opening_audio(backend) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    with pytest.raises(ValueError):
        QuestAudio(source="unknown")
    manager.open.assert_not_called()


def test_process_stop_failure_still_clears_queued_pcm_and_can_retry(monkeypatch) -> None:
    from iw3.desktop import quest_process_audio
    from iw3.desktop.quest_audio import QuestAudio

    native = MagicMock()
    native.sample_rate = 48000
    native.stop.side_effect = [RuntimeError("process stop failed"), None]
    monkeypatch.setattr(quest_process_audio, "ProcessAudio", lambda: native)
    audio = QuestAudio(source="process")
    audio.start()
    audio._chunks.append(b"\0" * 4)
    with pytest.raises(RuntimeError, match="process stop failed"):
        audio.stop()
    assert not audio._chunks
    audio.stop()
    audio.stop()
    assert audio.get_chunks() == []
    assert native.stop.call_count == 2


def test_audio_backlog_discards_oldest_and_drains_in_order(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    audio = QuestAudio()
    audio.start()
    callback = manager.open.call_args.kwargs["stream_callback"]
    for index in range(12):
        assert callback(bytes([index]) * 1920, 480, {}, 0) == (None, 0)
    assert audio.get_chunks() == [bytes([index]) * 1920 for index in range(2, 12)]
    assert audio.get_chunks() == []
    audio.stop()


def test_stop_releases_stream_and_manager_once(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    audio = QuestAudio()
    audio.stop()
    audio.start()
    manager.open.call_args.kwargs["stream_callback"](b"\0" * 1920, 480, {}, 0)
    audio.stop()
    audio.stop()
    assert audio.get_chunks() == []
    manager.open.return_value.stop_stream.assert_called_once_with()
    manager.open.return_value.close.assert_called_once_with()
    manager.terminate.assert_called_once_with()


def test_callback_failure_is_returned_to_sender(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    audio = QuestAudio()
    audio.start()
    callback = manager.open.call_args.kwargs["stream_callback"]
    assert callback(b"x", 480, {}, 0) == (None, 2)
    with pytest.raises(RuntimeError, match="音频|声音") as raised:
        audio.check_error()
    assert isinstance(raised.value.__cause__, ValueError)
    audio.stop()


def test_inactive_stream_is_reported(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    audio = QuestAudio()
    audio.start()
    manager.open.return_value.is_active.return_value = False
    with pytest.raises(RuntimeError, match="音频|声音"):
        audio.check_error()
    audio.stop()


def test_overflow_continues_but_other_callback_status_aborts(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    audio = QuestAudio()
    audio.start()
    callback = manager.open.call_args.kwargs["stream_callback"]
    assert callback(b"\0" * 1920, 480, {}, 2) == (None, 0)
    audio.check_error()
    assert callback(b"\0" * 1920, 480, {}, 4) == (None, 2)
    with pytest.raises(RuntimeError, match="音频|声音"):
        audio.check_error()
    audio.stop()


def test_native_sample_rate_is_used(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    manager.get_default_wasapi_loopback.return_value["defaultSampleRate"] = 44100.0
    audio = QuestAudio()
    audio.start()
    assert audio.sample_rate == 44100
    assert manager.open.call_args.kwargs["rate"] == 44100
    assert manager.open.call_args.kwargs["frames_per_buffer"] == 441
    audio.stop()


def test_nonloopback_device_is_never_opened(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    manager.get_default_wasapi_loopback.return_value["isLoopbackDevice"] = False
    with pytest.raises(RuntimeError, match="扬声器"):
        QuestAudio().start()
    manager.open.assert_not_called()
    manager.terminate.assert_called_once_with()


def test_unsupported_stereo_cleans_up_without_opening(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    _, manager = backend
    manager.is_format_supported.side_effect = ValueError("unsupported")
    audio = QuestAudio()
    with pytest.raises(RuntimeError, match="扬声器"):
        audio.start()
    manager.open.assert_not_called()
    manager.terminate.assert_called_once_with()


def test_host_initialization_error_has_speaker_context(backend: tuple[SimpleNamespace, MagicMock]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    module, _ = backend
    module.PyAudio.side_effect = OSError("host unavailable")
    with pytest.raises(RuntimeError, match="扬声器"):
        QuestAudio().start()


def test_missing_optional_package_gives_install_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    monkeypatch.setitem(sys.modules, "pyaudiowpatch", None)
    with pytest.raises(RuntimeError, match="requirements-quest.txt"):
        QuestAudio().start()


@pytest.mark.parametrize(
    ("gain", "expected"),
    [
        (0, [0, 0, 0, 0, 0, 0, 0, 0]),
        (1, [1000, -2000, 20000, -20000, 32767, -32768, 1, -1]),
        (2, [2000, -4000, 32767, -32768, 32767, -32768, 2, -2]),
        (4, [4000, -8000, 32767, -32768, 32767, -32768, 4, -4]),
    ],
)
def test_gain_scales_interleaved_pcm_without_overflow_or_source_mutation(gain: float, expected: list[int]) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    original = struct.pack("<8h", 1000, -2000, 20000, -20000, 32767, -32768, 1, -1)
    audio = QuestAudio(gain=gain)
    audio._chunks.append(original)
    chunks = audio.get_chunks()
    assert len(chunks) == 1
    assert len(chunks[0]) == len(original)
    assert list(struct.unpack("<8h", chunks[0])) == expected
    assert struct.unpack("<8h", original) == (1000, -2000, 20000, -20000, 32767, -32768, 1, -1)
    assert audio.get_chunks() == []
    if gain == 1:
        assert chunks[0] is original


def test_set_gain_applies_to_already_queued_pcm() -> None:
    from iw3.desktop.quest_audio import QuestAudio

    audio = QuestAudio()
    chunk = struct.pack("<2h", 1200, -1400)
    audio._chunks.extend([chunk, chunk])
    audio.set_gain(2.0)
    assert audio.get_chunks() == [struct.pack("<2h", 2400, -2800)] * 2
    audio._chunks.append(chunk)
    audio.set_gain(1.0)
    assert audio.get_chunks()[0] is chunk


@pytest.mark.parametrize(
    "gain", [True, False, None, "2", -0.01, 4.01, float("inf"), -float("inf"), float("nan"), 10**400]
)
def test_invalid_gain_is_rejected_without_changing_current_gain(gain: Any) -> None:
    from iw3.desktop.quest_audio import QuestAudio

    with pytest.raises(ValueError):
        QuestAudio(gain=gain)
    audio = QuestAudio(gain=2.0)
    with pytest.raises(ValueError):
        audio.set_gain(gain)
    audio._chunks.append(struct.pack("<2h", 1000, -1000))
    assert audio.get_chunks() == [struct.pack("<2h", 2000, -2000)]
