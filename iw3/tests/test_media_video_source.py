from dataclasses import replace
from fractions import Fraction
from io import BytesIO
from types import SimpleNamespace

import av
import pytest
import torch


def snapshot(**values):
    from iw3.desktop.media_video_source import MediaSnapshot

    return MediaSnapshot(
        **{
            "generation": "first",
            "data": b"compressed media",
            "revision": 1,
            "media_time": 2.0,
            "paused": False,
            "playback_rate": 1.0,
            "received_at": 10.0,
            "duration": 12.0,
            **values,
        }
    )


@pytest.mark.parametrize(
    "values,now,expected",
    [
        ({}, 10.25, 2.25),
        ({"paused": True}, 10.25, 2.0),
        ({"playback_rate": 2.0}, 10.25, 2.5),
        ({"media_time": 11.9}, 10.25, 12.0),
        ({}, 11.01, None),
        ({}, 9.9, None),
        ({"media_time": float("nan")}, 10.0, None),
        ({"playback_rate": 0.0}, 10.0, None),
        ({"data": b""}, 10.0, None),
    ],
)
def test_clock_follows_browser_pause_speed_and_rejects_stale_state(values, now, expected):
    actual = snapshot(**values).position_at(now)
    assert actual == expected


class FakeReader:
    instances = []

    def __init__(self, data, device):
        self.data = data
        self.device = device
        self.closed = False
        self.width, self.height, self.fps = 4, 6, 30.0
        self.calls = []
        FakeReader.instances.append(self)

    def frame_at(self, target):
        self.calls.append(target)
        timestamp = int(target * 30) / 30
        return torch.full((3, 6, 4), len(self.data) / 100), timestamp

    def close(self):
        self.closed = True


def make_source(monkeypatch, current):
    from iw3.desktop import media_video_source as module

    FakeReader.instances.clear()
    monkeypatch.setattr(module, "_NVDECReader", FakeReader)
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)
    return module.MediaVideoSource(lambda: current[0], torch.device("cuda:0"), fps=30)


def test_latest_generation_cannot_consume_pending_previous_video(monkeypatch):
    current = [snapshot(paused=True)]
    source = make_source(monkeypatch, current)
    source._update(current[0], 10.0)
    assert source.frame_set_event.is_set()
    current[0] = replace(current[0], generation="second", data=b"new")
    assert source.get_frame() is None
    assert not source.frame_set_event.is_set()
    source._update(current[0], 10.0)
    frame = source.get_frame()
    assert frame is not None and frame.shape == (3, 6, 4)
    assert float(frame.mean()) == pytest.approx(0.03)
    assert FakeReader.instances[0].closed
    assert source.get_content_rect() == (0, 0, 4, 6)
    assert source.get_status()["decoder"] == "NVDEC"
    assert source.get_status()["hardware_active"] is True
    assert source.get_status()["source_size"] == (4, 6)
    source.stop()


def test_pause_emits_once_and_seek_emits_new_time(monkeypatch):
    current = [snapshot(paused=True)]
    source = make_source(monkeypatch, current)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    source._update(current[0], 10.01)
    assert not source.frame_set_event.is_set()
    current[0] = replace(current[0], media_time=8.0)
    source._update(current[0], 10.01)
    assert source.get_frame() is not None
    assert FakeReader.instances[0].calls[-1] == 8.0
    source.stop()


def test_stale_or_missing_browser_drops_slot_and_content_geometry(monkeypatch):
    current = [snapshot(paused=True)]
    source = make_source(monkeypatch, current)
    source._update(current[0], 10.0)
    current[0] = None
    assert source.get_frame() is None
    assert source.get_content_rect() is None
    source._update(None, 10.1)
    assert source.get_status()["status"] == "waiting"
    assert source.get_status()["hardware_active"] is False
    assert source.get_status()["source_size"] is None
    current[0] = snapshot(received_at=8.0)
    source._update(current[0], 10.0)
    assert not source.frame_set_event.is_set()
    assert source.get_status()["status"] == "waiting"
    source.stop()


def test_partial_media_error_is_visible_and_new_bytes_can_retry(monkeypatch):
    from iw3.desktop import media_video_source as module

    current = [snapshot(data=b"partial")]
    source = make_source(monkeypatch, current)

    class IncompleteReader(FakeReader):
        def frame_at(self, target):
            if self.data == b"partial":
                raise ValueError("a signed URL must never appear in public status")
            return super().frame_at(target)

    monkeypatch.setattr(module, "_NVDECReader", IncompleteReader)
    source._update(current[0], 10.0)
    assert not source.frame_set_event.is_set()
    assert source.get_status()["status"] == "error"
    assert "signed URL" not in source.get_status()["message"]
    source._update(current[0], 10.01)
    assert len(FakeReader.instances) == 1
    current[0] = replace(current[0], data=b"complete", revision=2)
    source._update(current[0], 10.02)
    assert source.get_frame() is not None
    assert source.get_status()["status"] == "running"
    source.stop()


def test_buffer_revision_retains_reader_until_old_bytes_are_exhausted(monkeypatch):
    current = [snapshot()]
    source = make_source(monkeypatch, current)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    current[0] = replace(current[0], revision=2, data=b"more complete media")
    source._update(current[0], 10.1)
    assert len(FakeReader.instances) == 1
    source.stop()


def test_cpu_decode_is_never_silently_selected():
    from iw3.desktop.media_video_source import MediaVideoSource

    with pytest.raises(ValueError, match="CUDA"):
        MediaVideoSource(lambda: None, torch.device("cpu"), fps=30)


def test_consumed_generation_can_guard_ai_and_send_after_video_switch(monkeypatch):
    current = [snapshot(paused=True)]
    source = make_source(monkeypatch, current)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    consumed = source.get_frame_generation()
    assert consumed == "first" and source.is_current_generation(consumed)
    current[0] = replace(current[0], generation="second")
    assert not source.is_current_generation(consumed)
    source.stop()
    assert not source.is_current_generation("second")


def test_stale_transition_wakes_consumer_once_and_paused_recovery_resends(monkeypatch):
    current = [snapshot(paused=True)]
    source = make_source(monkeypatch, current)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    current[0] = None
    source._update(None, 10.0)
    assert source.frame_set_event.is_set()
    assert source.get_frame() is None
    source._update(None, 10.0)
    assert not source.frame_set_event.is_set()
    current[0] = snapshot(paused=True)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    source.stop()


def test_oversized_media_is_rejected_before_decoding_and_container_closes(monkeypatch):
    from iw3.desktop import media_video_source as module

    container = SimpleNamespace(
        streams=SimpleNamespace(
            video=[SimpleNamespace(codec_context=SimpleNamespace(width=20000, height=100), average_rate=30)]
        ),
        closed=False,
    )
    container.close = lambda: setattr(container, "closed", True)
    monkeypatch.setattr(module, "create_hwaccel", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.av, "open", lambda *args, **kwargs: container)
    with pytest.raises(ValueError, match="dimensions"):
        module._NVDECReader(b"external compressed media", torch.device("cuda:0"))
    assert container.closed


def test_decode_error_invalidates_consumed_frame_and_seek_retries_same_bytes(monkeypatch):
    from iw3.desktop import media_video_source as module

    current = [snapshot(paused=True)]
    source = make_source(monkeypatch, current)

    class PartiallyDecodableReader(FakeReader):
        def frame_at(self, target):
            if target >= 8:
                raise ValueError("incomplete future fragment")
            return super().frame_at(target)

    monkeypatch.setattr(module, "_NVDECReader", PartiallyDecodableReader)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    consumed = source.get_frame_generation()
    assert source.is_current_generation(consumed)
    current[0] = replace(current[0], media_time=8)
    source._update(current[0], 10.0)
    assert not source.is_current_generation(consumed)
    assert source.get_status()["status"] == "error"
    current[0] = replace(current[0], media_time=2)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    assert source.get_status()["status"] == "running"
    current[0] = replace(current[0], media_time=7.8)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    source.stop()


def test_same_failed_clock_position_is_not_retried_each_pump(monkeypatch):
    from iw3.desktop import media_video_source as module

    current = [snapshot(media_time=8)]
    source = make_source(monkeypatch, current)

    class UndecodableReader(FakeReader):
        def frame_at(self, target):
            raise ValueError("incomplete fragment")

    monkeypatch.setattr(module, "_NVDECReader", UndecodableReader)
    source._update(current[0], 10.0)
    source._update(current[0], 10.03)
    source._update(current[0], 10.3)
    assert len(FakeReader.instances) == 1
    source._update(current[0], 10.6)
    assert len(FakeReader.instances) == 2
    source.stop()


@pytest.mark.hwaccel
@pytest.mark.parametrize("fragmented", [False, True])
def test_real_nvdec_bytes_keep_rgb_shape_timestamps_and_backward_seek(fragmented):
    from iw3.desktop.media_video_source import _NVDECReader
    from nunif.utils.video import pyav_init_cuda_primary_context

    pyav_init_cuda_primary_context(max_devices=1)
    output = BytesIO()
    options = {"movflags": "empty_moov+frag_keyframe+default_base_moof"} if fragmented else None
    with av.open(output, mode="w", format="mp4", options=options) as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.options = {"crf": "12", "g": "2", "bf": "0"}
        for index in range(12):
            frame = av.VideoFrame(96, 64, "rgb24")
            pixels = bytes((index * 15, 50, 180)) * (96 * 64)
            frame.planes[0].update(pixels)
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reader = _NVDECReader(output.getvalue(), torch.device("cuda:0"))
    try:
        assert (reader.width, reader.height, reader.fps) == (96, 64, 30.0)
        with torch.inference_mode():
            for target, red in [(0.0, 0), (0.2, 90), (0.1, 45), (0.3, 135)]:
                result = reader.frame_at(target)
                assert result is not None
                rgb, timestamp = result
                assert timestamp == pytest.approx(target, abs=1 / 30)
                assert rgb.shape == (3, 64, 96) and rgb.dtype == torch.float32 and rgb.device.type == "cuda"
                assert torch.isfinite(rgb).all() and 0 <= rgb.min() <= rgb.max() <= 1
                assert float(rgb[0].mean()) * 255 == pytest.approx(red, abs=8)
            assert reader.frame_at(4.0) is None
    finally:
        reader.close()


def remote_descriptor(kind="vod"):
    return {
        "kind": kind,
        "id": "current:one",
        "renditions": [
            {
                "id": "low",
                "width": 720,
                "height": 1280,
                "bitrate": 1000,
                "codec": "h264",
                "urls": ["https://media.douyinvod.com/low"],
            },
            {
                "id": "high",
                "width": 1440,
                "height": 2560,
                "bitrate": 4000,
                "codec": "hevc",
                "urls": ["https://media.douyinvod.com/bad", "https://media.douyinvod.com/good"],
            },
        ],
    }


class RemoteReader(FakeReader):
    attempts = []
    conversions = []

    def __init__(self, data, device, *, live=False):
        RemoteReader.attempts.append(data.url)
        if "bad" in data.url:
            raise ValueError("private URL must never appear")
        super().__init__(data, device)
        self.width, self.height, self.codec = 1440, 2560, "hevc"
        if data.url.endswith("/low"):
            self.width, self.height, self.codec = 720, 1280, "h264"
        self.live = live

    def frame_at(self, target):
        return torch.full((3, 6, 4), 0.25), target

    def next_frame(self, *, convert=True):
        RemoteReader.conversions.append(convert)
        return (torch.full((3, 6, 4), 0.25) if convert else None), float(len(RemoteReader.conversions))


def make_remote_source(monkeypatch, current, quality="auto"):
    from iw3.desktop import media_video_source as module

    RemoteReader.attempts.clear()
    RemoteReader.conversions.clear()
    handles = []

    def network(url, cancelled, *, live=False):
        handle = SimpleNamespace(url=url, cancelled=cancelled, live=live, closed=False)
        handle.close = lambda: setattr(handle, "closed", True)
        handles.append(handle)
        return handle

    monkeypatch.setattr(module, "MediaHTTPIO", network)
    monkeypatch.setattr(module, "_NVDECReader", RemoteReader)
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)
    return module.MediaVideoSource(lambda: current[0], torch.device("cuda:0"), fps=30, quality=quality), handles


def test_descriptor_clock_accepts_no_buffer_but_keeps_ttl():
    current = snapshot(data=b"", source=remote_descriptor())
    assert current.position_at(10.25) == 2.25
    assert current.position_at(11.1) is None


def test_remote_vod_tries_same_quality_urls_before_lower_quality(monkeypatch):
    current = [snapshot(data=b"", source=remote_descriptor())]
    source, handles = make_remote_source(monkeypatch, current)
    source._update(current[0], 10.0)
    assert RemoteReader.attempts == ["https://media.douyinvod.com/bad", "https://media.douyinvod.com/good"]
    assert handles[0].closed and not handles[1].closed
    assert source.get_frame() is not None
    status = source.get_status()
    assert status["selected_quality"] == "1440p" and status["available_quality"] == [1440, 720]
    assert status["codec"] == "hevc" and not status["fallback_required"]
    assert "private" not in str(status) and "https" not in str(status)
    source.stop()
    assert all(handle.closed for handle in handles)


def test_quality_limit_skips_larger_candidates_and_full_failure_requires_capture(monkeypatch):
    selected = remote_descriptor()
    selected["renditions"][0]["urls"] = ["https://media.douyinvod.com/bad-low"]
    current = [snapshot(data=b"", source=selected)]
    source, _ = make_remote_source(monkeypatch, current, quality="1080")
    source._update(current[0], 10.0)
    assert RemoteReader.attempts == ["https://media.douyinvod.com/bad-low"]
    status = source.get_status()
    assert status["fallback_required"] and status["fallback_reason"]
    assert "https" not in str(status)
    source.stop()


def test_only_after_every_same_quality_url_fails_selects_next_rendition(monkeypatch):
    description = remote_descriptor()
    description["renditions"][1]["urls"] = [
        "https://media.douyinvod.com/bad-one",
        "https://media.douyinvod.com/bad-two",
    ]
    current = [snapshot(data=b"", source=description)]
    source, handles = make_remote_source(monkeypatch, current)
    source._update(current[0], 10)
    assert RemoteReader.attempts == [
        "https://media.douyinvod.com/bad-one",
        "https://media.douyinvod.com/bad-two",
        "https://media.douyinvod.com/low",
    ]
    assert source.get_frame() is not None
    assert source.get_status()["selected_quality"] == "720p"
    assert source.get_status()["fallback_reason"] and not source.get_status()["fallback_required"]
    assert all(handle.closed for handle in handles[:-1])
    source.stop()


def test_missing_source_gets_finite_wait_then_explicit_fallback(monkeypatch):
    current = [None]
    source = make_source(monkeypatch, current)
    source._update(None, 10)
    assert not source.get_status().get("fallback_required", False)
    source._update(None, 14)
    assert source.get_status()["fallback_required"]
    source.stop()


def test_missing_source_failure_belongs_only_to_stable_current_operation(monkeypatch):
    from iw3.desktop import media_video_source as module

    operation = {"generation": "first", "fresh": True}
    source = module.MediaVideoSource(lambda: None, torch.device("cuda:0"), state=lambda: dict(operation))
    source._update(None, 10)
    source._update(None, 14)
    assert source.get_status()["failure_generation"] == "first"
    operation["generation"] = "second"
    source._update(None, 14.1)
    assert not source.get_status()["fallback_required"]
    assert source.get_status()["failure_generation"] is None
    source._update(None, 18)
    assert source.get_status()["failure_generation"] == "second"
    operation["fresh"] = False
    source._update(None, 22)
    assert not source.get_status()["fallback_required"]
    source.stop()


def test_remote_failure_tags_current_generation_and_success_clears_failure(monkeypatch):
    description = remote_descriptor()
    description["renditions"] = [description["renditions"][1]]
    description["renditions"][0]["urls"] = ["https://media.douyinvod.com/bad"]
    current = [snapshot(data=b"", source=description)]
    source, _ = make_remote_source(monkeypatch, current)
    source._update(current[0], 10)
    assert source.get_status()["failure_generation"] == "first"
    current[0] = replace(current[0], generation="second", source=remote_descriptor())
    source._update(current[0], 10)
    assert not source.get_status()["fallback_required"] and source.get_status()["failure_generation"] is None
    source.stop()


def test_same_operation_rendition_change_invalidates_old_ai_result(monkeypatch):
    from iw3.desktop import media_video_source as module

    current = [snapshot(data=b"", source=remote_descriptor())]
    source, _ = make_remote_source(monkeypatch, current)

    class FailingHighReader(RemoteReader):
        def frame_at(self, target):
            if self.data.url.endswith("/good") and target >= 8:
                raise ValueError("source connection interrupted")
            return super().frame_at(target)

    monkeypatch.setattr(module, "_NVDECReader", FailingHighReader)
    source._update(current[0], 10)
    assert source.get_frame() is not None
    previous = source.get_frame_generation()
    assert source.is_current_generation(previous)
    current[0] = replace(current[0], media_time=8)
    source._update(current[0], 10)
    assert source.get_frame() is not None
    assert source.get_status()["selected_quality"] == "720p"
    assert source.get_frame_generation() != previous
    assert not source.is_current_generation(previous)
    source.stop()


def test_live_drains_all_frames_but_limits_rgb_publication(monkeypatch):
    description = remote_descriptor("live")
    description["renditions"] = [description["renditions"][1]]
    current = [snapshot(data=b"", source=description)]
    source, handles = make_remote_source(monkeypatch, current)
    source._update(current[0], 10.0)
    assert source.get_frame() is not None
    source._update(current[0], 10.001)
    source._update(current[0], 10.002)
    assert RemoteReader.conversions == [True, False, False]
    assert not source.frame_set_event.is_set()
    current[0] = replace(current[0], paused=True)
    source._update(current[0], 10.1)
    assert RemoteReader.conversions[-1] is False
    current[0] = replace(current[0], generation="new")
    assert handles[-1].cancelled()
    assert not source.is_current_generation("first")
    source.stop()


def test_stale_remote_source_closes_network_and_decoder(monkeypatch):
    current = [snapshot(data=b"", source=remote_descriptor())]
    source, handles = make_remote_source(monkeypatch, current)
    source._update(current[0], 10)
    current[0] = None
    source._update(None, 10.1)
    assert all(handle.closed for handle in handles)
    assert source.get_frame() is None
    source.stop()


def test_live_unknown_size_is_checked_against_quality_after_decode_open(monkeypatch):
    description = remote_descriptor("live")
    description["renditions"][1].update(width=0, height=0)
    description["renditions"] = [description["renditions"][1]]
    current = [snapshot(data=b"", source=description)]
    source, handles = make_remote_source(monkeypatch, current, quality="720")
    source._update(current[0], 10)
    assert source.get_status()["fallback_required"]
    assert all(handle.closed for handle in handles)
    source.stop()


def test_skipped_live_frame_still_requires_hardware_output():
    from iw3.desktop.media_video_source import _NVDECReader

    reader = _NVDECReader.__new__(_NVDECReader)
    frame = av.VideoFrame(4, 6, "rgb24")
    frame.pts, frame.time_base = 0, Fraction(1, 30)
    reader._frames = iter([frame])
    reader.stream = SimpleNamespace(codec_context=SimpleNamespace(is_hwaccel=False))
    with pytest.raises(RuntimeError, match="NVDEC"):
        reader.next_frame(convert=False)


def test_decoder_forbids_ffmpeg_opening_side_resources_outside_checked_reader(monkeypatch):
    from iw3.desktop import media_video_source as module

    captured = {}
    container = SimpleNamespace(
        streams=SimpleNamespace(
            video=[SimpleNamespace(codec_context=SimpleNamespace(width=96, height=64, name="h264"), average_rate=30)]
        ),
        close=lambda: None,
    )

    def open_media(*args, **kwargs):
        captured.update(kwargs)
        return container

    monkeypatch.setattr(module.av, "open", open_media)
    monkeypatch.setattr(module, "create_hwaccel", lambda *args, **kwargs: None)
    reader = module._NVDECReader(b"fixed input", torch.device("cuda:0"))
    assert callable(captured.get("io_open"))
    with pytest.raises(ValueError) as error:
        captured["io_open"]("http://127.0.0.1/private", 1, {})
    assert "private" not in str(error.value)
    reader.close()


@pytest.mark.parametrize("mode", ["decode_error", "missing_position"])
def test_legacy_mse_growing_but_unreadable_media_falls_back_after_grace_and_recovers(monkeypatch, mode):
    from iw3.desktop import media_video_source as module

    current = [snapshot(data=b"partial", paused=True)]
    source = make_source(monkeypatch, current)

    class IncompleteReader(FakeReader):
        def frame_at(self, target):
            if self.data.startswith(b"partial"):
                if mode == "decode_error":
                    raise ValueError("not complete yet")
                return None
            return super().frame_at(target)

    monkeypatch.setattr(module, "_NVDECReader", IncompleteReader)
    source._update(current[0], 10)
    assert not source.get_status()["fallback_required"]
    for index in range(1, 5):
        current[0] = replace(current[0], data=current[0].data + b"x", revision=index + 1, received_at=10 + index)
        source._update(current[0], 10 + index)
    assert source.get_status()["fallback_required"]
    assert source.get_status()["failure_generation"] == "first"
    current[0] = replace(current[0], data=b"complete", revision=9, received_at=14)
    source._update(current[0], 14)
    assert source.get_status()["status"] == "running"
    assert not source.get_status()["fallback_required"]
    assert source.get_status()["failure_generation"] is None
    current[0] = replace(current[0], generation="second", data=b"partial", received_at=15)
    source._update(current[0], 15)
    assert not source.get_status()["fallback_required"]
    source.stop()


def test_legacy_same_failed_paused_position_eventually_requests_fallback(monkeypatch):
    from iw3.desktop import media_video_source as module

    current = [snapshot(paused=True)]
    source = make_source(monkeypatch, current)

    class AlwaysFails(FakeReader):
        def frame_at(self, target):
            raise ValueError("still incomplete")

    monkeypatch.setattr(module, "_NVDECReader", AlwaysFails)
    source._update(current[0], 10)
    current[0] = replace(current[0], received_at=14)
    source._update(current[0], 14)
    assert len(FakeReader.instances) == 1
    assert source.get_status()["fallback_required"] and source.get_status()["failure_generation"] == "first"
    source.stop()
