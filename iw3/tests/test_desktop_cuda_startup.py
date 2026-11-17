import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.hwaccel
def test_gui_startup_allows_nvdec_after_device_query_and_ai_allocation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # 独立进程避免其他硬件测试提前建立上下文，掩盖桌面 GUI 的启动顺序。
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import io
import sys
from fractions import Fraction

import av
import torch

from iw3.desktop import gui
from iw3.desktop.media_video_source import _NVDECReader
from nunif.utils.video import pyav_init_cuda_primary_context

class App:
    quest_restart_path = None

    def __init__(self, resume_values):
        torch.cuda.get_device_properties(0)

    def MainLoop(self):
        pyav_init_cuda_primary_context()
        self.ai_tensor = torch.zeros((3, 64, 96), device="cuda:0")
        data = io.BytesIO()
        with av.open(data, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
            frame = av.VideoFrame(96, 64, "rgb24")
            frame.planes[0].update(bytes((40, 90, 170)) * (96 * 64))
            frame.pts, frame.time_base = 0, Fraction(1, 30)
            for packet in (*stream.encode(frame), *stream.encode()):
                container.mux(packet)
        reader = _NVDECReader(data.getvalue(), torch.device("cuda:0"))
        try:
            with torch.inference_mode():
                rgb, timestamp = reader.frame_at(0.0)
                assert rgb.shape == self.ai_tensor.shape == (3, 64, 96)
                assert rgb.device.type == "cuda" and rgb.dtype == torch.float32
                assert torch.isfinite(rgb).all() and 0 <= rgb.min() <= rgb.max() <= 1
                assert timestamp == 0 and reader.stream.codec_context.is_hwaccel
                assert reader._selected.format.name == "cuda"
        finally:
            reader.close()

sys.argv = ["iw3.desktop.gui"]
gui.load_language_setting = lambda _: None
gui.IW3DesktopApp = App
gui.main()
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
