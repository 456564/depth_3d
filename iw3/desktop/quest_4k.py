"""固定输入的网页 4K 深度前向和单槽节拍，仅供 Quest 媒体线程。"""

import time
from typing import TYPE_CHECKING

import torch

from ..depth_models.depth_anything_model import _forward
from .media_clock import now as media_now

if TYPE_CHECKING:
    from .quest_session import QuestSession
    from .source_router import SourceRouter


FrameToken = tuple[int, str | None] | None
INPUT_SHAPE = (1, 3, 392, 686)


def playing_4k(capture: "SourceRouter | None", token: FrameToken) -> bool:
    if capture is None or capture.source_kind != "browser" or token is None:
        return False
    source = capture._source
    reader = getattr(source, "_reader", None)
    if (
        reader is None
        or reader.live
        or not reader._uses_fused_color
        or (reader.width, reader.height, reader.fps) != (3840, 2160, 60)
    ):
        return False
    state = source.snapshot()
    return (
        state is not None
        and state.source is not None
        and state.source.get("kind") == "vod"
        and not state.paused
        and state.playback_rate == 1.0
        and state.position_at(media_now()) is not None
        and capture.is_current_generation(token)
    )


class Quest4KForward:
    @torch.inference_mode()
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.enabled = self.used = False
        with torch.cuda.device(device):
            self.stream = torch.cuda.current_stream(device)
            helper = torch.cuda.Stream(device)
            helper.wait_stream(self.stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(helper):
                generator = torch.Generator(device=device).manual_seed(392)
                self.input = torch.randn(INPUT_SHAPE, device=device, dtype=torch.float32, generator=generator)
                for _ in range(8):
                    expected = _forward(model, self.input, True)
                helper.synchronize()
                with torch.cuda.graph(self.graph, stream=helper):
                    self.output = _forward(model, self.input, True)
            self.stream.wait_stream(helper)
            self.graph.replay()
            if not torch.equal(expected, self.output):
                raise RuntimeError("CUDA Graph depth output differs from eager")

    def __call__(self, model: torch.nn.Module, x: torch.Tensor, enable_amp: bool) -> torch.Tensor:
        if not (
            self.enabled
            and model is self.model
            and enable_amp
            and x.is_cuda
            and x.device == self.input.device
            and x.dtype == torch.float32
            and x.shape == INPUT_SHAPE
            and x.is_contiguous()
            and torch.cuda.current_stream(x.device) == self.stream
        ):
            return _forward(model, x, enable_amp)
        self.input.copy_(x)
        self.graph.replay()
        self.used = True
        # 逐帧深度可能留在后处理；不能交出下一次重放会写入的地址。
        return self.output.clone()

    def close(self) -> None:
        self.stream.synchronize()


def wait_for_slot(session: "QuestSession", capture: "SourceRouter") -> None:
    server = session.server
    if server is None or session.capture is not capture:
        return
    with server._condition:
        if (
            not server._video_enabled
            or server._suspended
            or server._connection is None
            or server._pending is None
            or (server.width, server.height) != (7680, 2160)
        ):
            return
        connection, generation, slot = server._connection, server._generation, id(server._pending)
    token = capture.get_frame_generation()
    deadline = time.perf_counter() + 0.010
    while not (session.stop_event.is_set() or session.run_stop.is_set() or server._shutdown.is_set()):
        if not playing_4k(capture, token):
            return
        with server._condition:
            if connection is not server._connection or generation != server._generation or slot != id(server._pending):
                return
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(0.00075, remaining))
