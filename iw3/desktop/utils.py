import ipaddress
import math
import socket
import sys
import threading
import time
from argparse import Namespace
from collections import deque
from os import path

import torch
import wx  # for mouse pointer
from packaging.version import Version
from torchvision.io import encode_jpeg

from nunif.device import create_device
from nunif.initializer import gc_collect
from nunif.models import compile_model
from nunif.models.data_parallel import DeviceSwitchInference
from nunif.utils.video import pyav_init_cuda_primary_context

from .. import models  # noqa
from .. import utils as IW3U
from ..stereo_model_factory import get_mlbw_divergence_level
from .quest_settings import AMBIENT_STYLES, BROWSER_QUALITY_LABELS
from .screenshot_process import (  # noqa
    ScreenshotProcess,
    enum_window_names,
    get_monitor_size_list,
    get_window_rect_by_title,
    is_mss_supported,
)
from .screenshot_process_obs import (  # noqa
    ScreenshotProcessOBS,
    is_obs_supported,
    is_wayland,
    obs_unsupported_reason,
)
from .screenshot_thread_cuda import ScreenshotThreadWCCUDA
from .screenshot_thread_kwcapture import (  # noqa
    ScreenshotThreadKWCapture,
    is_kwcapture_supported,
    kwcapture_unsupported_reason,
)
from .screenshot_thread_pil import ScreenshotThreadPIL
from .streaming_server import StreamingServer

WAYLAND_CAPTURE_MISSING_MESSAGE = (
    "Wayland capture backend is not available ('pil' uses XWayland, which is usually black). "
    "Install obs-studio/obs-plugins for 'obs', or 'pip install kwcapture' for 'kwcapture' (KDE Plasma)."
)

LocalViewer: object | None
try:
    from .local_viewer import LocalViewer
except ImportError:
    LocalViewer = None


TORCH_VERSION = Version(torch.__version__)
ENABLE_GPU_JPEG = (TORCH_VERSION.major, TORCH_VERSION.minor) >= (2, 7)
TORCH_NUM_THREADS = torch.get_num_threads()

# Thread-safe JPEG encoding lock
_jpeg_encode_lock = threading.Lock()


def init_win32():
    if sys.platform == "win32":
        import ctypes

        try:
            # Fix mouse position when Display Scaling is not 100%
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except:  # noqa
            pass

        if sys.version_info < (3, 11):
            try:
                # Change timer/sleep precision for 3.10
                # NOTE: python 3.11 or later has high precision sleep.
                ctypes.windll.winmm.timeBeginPeriod(1)
            except:  # noqa
                pass


def init_num_threads(device_id):
    if device_id < 0:
        # cpu
        torch.set_num_threads(TORCH_NUM_THREADS)
    else:
        torch.set_num_threads(1)


def get_local_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 8.8.8.8 is Google DNS address
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:  # noqa
        return "127.0.0.1"  # unknown


def is_private_address(ip):
    try:
        ip = ipaddress.ip_address(ip)
        return not ip.is_unspecified and ip.is_private
    except ValueError:
        return False


def is_loopback_address(ip):
    try:
        ip = ipaddress.ip_address(ip)
        return ip.is_loopback
    except ValueError:
        return False


def to_uint8(x):
    return x.mul(255).round_().to(torch.uint8)


def fps_sleep(start_time, fps, resolution=2e-4):
    # currently not in used
    end_diff_time = (1 / fps) - resolution
    while time.perf_counter() - start_time < end_diff_time:
        time.sleep(resolution)


def to_jpeg_data(frame, quality, tick, gpu_jpeg=True):
    # Thread-safe JPEG encoding with lock
    with _jpeg_encode_lock:
        if ENABLE_GPU_JPEG and gpu_jpeg and frame.device.type == "cuda":
            jpeg_data = encode_jpeg(to_uint8(frame), quality=quality).cpu()
        else:
            jpeg_data = encode_jpeg(to_uint8(frame).cpu(), quality=quality)

    jpeg_data = jpeg_data.numpy().tobytes()
    # debug_jpeg_data(frame, jpeg_data)
    return (jpeg_data, tick)


def debug_jpeg_data(frame, jpeg_data):
    from torchvision.io import decode_jpeg

    jpeg_data = torch.tensor(list(jpeg_data), dtype=torch.uint8)
    try:
        decodec_frame = decode_jpeg(jpeg_data).cpu() / 255.0
        diff = (frame.cpu() - decodec_frame).abs().mean()
        print(diff)
    except RuntimeError as e:
        print(e)


def create_parser():
    local_address = get_local_address()
    parser = IW3U.create_parser(required_true=False)
    viewer = parser.add_mutually_exclusive_group()
    viewer.add_argument("--local-viewer", action="store_true", help="Use Local Viewer instead of Web Streaming")
    viewer.add_argument("--quest", action="store_true", help="HEVC LAN streaming to the Pop3D Quest app")
    parser.add_argument(
        "--quest-auto-browser", action="store_true", help="Automatically use compressed webpage video during fullscreen"
    )
    parser.add_argument("--quest-bitrate", type=int, default=35, help="Quest HEVC bitrate in Mbps (5-100)")
    parser.add_argument(
        "--browser-quality",
        choices=tuple(BROWSER_QUALITY_LABELS),
        default="auto",
        help="Maximum source short-edge resolution; auto selects highest readable quality",
    )
    parser.add_argument("--quest-no-audio", action="store_true", help="Disable PC speaker audio in Quest mode")
    parser.add_argument("--quest-audio-source", choices=["process", "speaker"], default="process")
    parser.add_argument("--quest-audio-gain", type=float, default=1.0, help="Quest-only audio gain (0-4)")
    parser.add_argument(
        "--quest-control", action="store_true", help="Control the captured Windows window or monitor with Quest"
    )
    parser.add_argument(
        "--quest-screen-curvature",
        type=float,
        default=0.0,
        help="Quest horizontal screen arc in degrees (0-120; 0=flat)",
    )
    parser.add_argument("--port", type=int, default=1303, help="HTTP listen port")
    parser.add_argument("--quest-ambient", action="store_true", help="Enable ambient light around the Quest screen")
    parser.add_argument(
        "--quest-ambient-style", choices=AMBIENT_STYLES, default="edge", help="Quest ambient light style"
    )
    parser.add_argument("--quest-ambient-brightness", type=float, default=0.25, help="Ambient brightness (0-1)")
    parser.add_argument("--quest-ambient-spread", type=float, default=2.0, help="Ambient coverage multiplier (1.2-4)")
    parser.add_argument(
        "--quest-ambient-no-fill-borders",
        action="store_false",
        dest="quest_ambient_fill_borders",
        help="Keep video black bars instead of filling them with ambient light",
    )
    parser.add_argument("--quest-ambient-feather", type=float, default=0.03, help="Video edge feather width (0-0.1)")
    parser.add_argument(
        "--quest-ambient-front-only",
        action="store_false",
        dest="quest_ambient_360",
        help="Disable full-sphere ambient coverage",
    )
    parser.add_argument(
        "--quest-ambient-no-motion",
        action="store_false",
        dest="quest_ambient_motion",
        help="Disable slow ambient animation",
    )
    parser.add_argument("--bind-addr", type=str, default=local_address, help="HTTP listen address")
    parser.add_argument("--user", type=str, help="HTTP Basic Authentication username")
    parser.add_argument("--password", type=str, help="HTTP Basic Authentication password")
    parser.add_argument("--stream-fps", type=int, default=30, help="Streaming FPS")
    parser.add_argument(
        "--uncap-fps",
        action="store_true",
        help="Allows LocalViewer to render at frame rates beyond the display's refresh rate",
    )
    parser.add_argument(
        "--fit-window",
        action="store_true",
        help="Stretch image to fit the Local Viewer window ignoring aspect ratio",
    )
    parser.add_argument("--stream-height", type=int, default=1080, help="Streaming screen resolution")
    parser.add_argument("--stream-quality", type=int, default=90, help="Streaming JPEG quality")
    parser.add_argument("--full-sbs", action="store_true", help="Use Full SBS for Pico4")
    parser.add_argument(
        "--screenshot",
        type=str,
        default="pil",
        choices=["pil", "mss", "wc_mp", "wc_cuda", "kwcapture", "obs", "browser_video"],
        help="Input source; browser_video uses compressed webpage video and NVDEC",
    )
    parser.add_argument("--gpu-jpeg", action="store_true", help="Use GPU JPEG Encoder")
    parser.add_argument("--monitor-index", type=int, default=0, help="monitor_index for wc_mp. 0 origin. 0 = monitor 1")
    parser.add_argument(
        "--window-name",
        type=str,
        help=("target window name for wc_mp. When this is specified, --monitor-index is ignored"),
    )
    parser.add_argument("--crop-top", type=int, default=0, help="Crop pixels from top when using --window-name")
    parser.add_argument("--crop-left", type=int, default=0, help="Crop pixels from left when using --window-name")
    parser.add_argument("--crop-right", type=int, default=0, help="Crop pixels from right when using --window-name")
    parser.add_argument("--crop-bottom", type=int, default=0, help="Crop pixels from bottom when using --window-name")
    parser.add_argument(
        "--disable-draw-cursor", action="store_true", help="Disables drawing the cursor on the output stream"
    )

    parser.set_defaults(
        input="dummy",
        output="dummy",
        depth_model="Any_V2_S",
        divergence=1.0,
        convergence=1.0,
        ema_normalize=True,
    )
    return parser


def prepare_quest_args(args: Namespace) -> None:
    browser_video = args.screenshot == "browser_video"
    auto_browser = getattr(args, "quest_auto_browser", False)
    if auto_browser:
        if not args.quest:
            raise ValueError("抖音全屏自动视频流需要 Quest 模式")
        if (
            args.pad is not None
            or args.pad_mode not in {None, "tblr"}
            or args.ipd_offset != 0
            or args.rotate_left
            or args.rotate_right
        ):
            raise ValueError("抖音全屏自动视频流需要关闭补边、IPD偏移和旋转；桌面窗口裁剪仍可使用")
    if browser_video:
        if not args.quest:
            raise ValueError("Browser video source requires Quest mode")
        if args.pad is not None or args.pad_mode not in {None, "tblr"}:
            raise ValueError("Browser video source requires no padding / 补边")
        if any((args.crop_top, args.crop_left, args.crop_right, args.crop_bottom)):
            raise ValueError("Browser video source requires zero crop values")
    if args.quest_control:
        if not args.quest or sys.platform != "win32":
            raise ValueError("Quest PC control requires Quest mode and Windows")
        args.window_name = args.window_name if args.window_name and args.window_name.strip() else None
        if args.screenshot not in {"wc_mp", "mss", "wc_cuda", "browser_video"}:
            raise ValueError("Quest PC control requires wc_mp, mss or wc_cuda capture")
        if args.window_name is None and any((args.crop_top, args.crop_left, args.crop_right, args.crop_bottom)):
            raise ValueError("Quest PC control of a monitor requires zero crop values")
        if (
            args.pad is not None
            or args.pad_mode not in {None, "tblr"}
            or args.ipd_offset != 0
            or args.rotate_left
            or args.rotate_right
        ):
            raise ValueError("Quest PC control requires no padding, IPD offset or rotation")
    if not args.quest:
        return
    from .quest_settings import validate_patch

    validate_patch(
        {
            key: getattr(args, key)
            for key in (
                "quest_screen_curvature",
                "quest_ambient",
                "quest_ambient_style",
                "quest_ambient_360",
                "quest_ambient_motion",
                "browser_quality",
                "quest_ambient_brightness",
                "quest_ambient_spread",
                "quest_ambient_fill_borders",
                "quest_ambient_feather",
            )
        }
    )
    args.method = {"row_flow": "row_flow_v3", "row_flow_sym": "row_flow_v3_sym"}.get(args.method, args.method)
    if isinstance(args.edge_dilation, list) and len(args.edge_dilation) == 1:
        args.edge_dilation = args.edge_dilation[0]
    if not math.isfinite(args.quest_audio_gain) or not 0 <= args.quest_audio_gain <= 4:
        raise ValueError("Quest audio gain must be finite and between 0 and 4")
    if args.local_viewer or args.gpu[0] < 0:
        raise ValueError("Quest output requires an NVIDIA GPU and cannot use Local Viewer")
    if not 5 <= args.quest_bitrate <= 100 or args.stream_fps <= 0:
        raise ValueError("Quest requires 5-100 Mbps and a positive streaming FPS")
    args.full_sbs = True
    # 原生客户端分别取全宽双眼；不沿用网页的半宽或其他排版。
    for name in ("half_sbs", "tb", "half_tb", "cross_eyed", "rgbd", "half_rgbd", "vr180"):
        setattr(args, name, False)
    args.anaglyph = None


def set_state_args(args, args_lock=None, stop_event=None, fps_event=None, depth_model=None, quest_owner=False):
    prepare_quest_args(args)
    if args.quest:
        pyav_init_cuda_primary_context()
    if args.quest and not quest_owner:
        # 先建立管理菜单；自动会聚等可能下载/失败的模型初始化留给媒体owner。
        args.state = {"stop_event": stop_event, "depth_model": None, "convergence_model": None}
    else:
        IW3U.set_state_args(args, stop_event=stop_event, depth_model=depth_model)
    args.state["fps_event"] = fps_event
    args.state["args_lock"] = args_lock if args_lock is not None else threading.Lock()
    if args.edge_dilation is None or (isinstance(args.edge_dilation, list) and len(args.edge_dilation) == 0):
        args.edge_dilation = 2


def test_output_size(size, args, depth_model, side_model):
    frame = torch.zeros((3, *size), dtype=torch.float32).to(args.state["device"])
    sbs = IW3U.process_image(frame, args, depth_model, side_model, autocrop_uncrop=True)
    depth_model.reset()
    return sbs.shape[-2:]


def try_switch_mlbw_model(old_divergence, new_divergence, old_side_model, args):
    assert args.method in {"mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s"}

    scale = 2.0 if args.synthetic_view in {"right", "left"} else 1.0
    old_level = get_mlbw_divergence_level(old_divergence * scale)
    new_level = get_mlbw_divergence_level(new_divergence * scale)

    # print("switch_mlbw_model", old_level, new_level, old_divergence, new_divergence)

    if old_level == new_level:
        return old_side_model

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    new_side_model = IW3U.create_stereo_model(
        args.method,
        divergence=new_divergence * scale,
        device_id=args.gpu[0],
    )
    if args.compile and not isinstance(new_side_model, DeviceSwitchInference):
        new_side_model = compile_model(new_side_model, args.state["device"])
        gc_collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return new_side_model


def iw3_desktop_main(args, init_wxapp=True):
    if args.quest and not args.state.get("quest_session"):
        from .quest_session import QuestSession

        return QuestSession(args).run(_iw3_desktop_main_once, init_wxapp)
    return _iw3_desktop_main_once(args, init_wxapp)


def _iw3_desktop_main_once(args, init_wxapp=True):
    init_num_threads(args.gpu[0])
    session = args.state.get("quest_session")

    if not any([args.full_sbs, args.tb, args.half_tb, args.cross_eyed, args.rgbd, args.half_rgbd, args.anaglyph]):
        args.half_sbs = True

    if args.user or args.password:
        user = args.user or ""
        password = args.password or ""
        auth = (user, password)
    else:
        auth = None

    if not args.local_viewer:
        if args.bind_addr is None:
            args.bind_addr = get_local_address()
            if not is_private_address(args.bind_addr):
                raise RuntimeError(
                    f"Detected IP address({args.bind_addr}) is not Local Area Network Address. "
                    " Specify --bind-addr option"
                )

        if is_loopback_address(args.bind_addr):
            print(
                (
                    f"Warning: {args.bind_addr} is only accessible from this PC. "
                    "If this option is not explicitly enabled, the network connection might be unavailable."
                ),
                file=sys.stderr,
            )
        if not is_private_address(args.bind_addr) and auth is None:
            raise RuntimeError(
                f"({args.bind_addr}) is not Local Area Network Address. Specify --username/--password option."
            )

    browser_video = args.screenshot == "browser_video"
    fixed_browser = browser_video
    auto_browser = bool(args.quest and getattr(args, "quest_auto_browser", False) and not browser_video)
    versioned_source = browser_video or auto_browser
    if versioned_source:
        from .browser_video_bridge import BrowserVideoBridge
        from .media_video_source import MediaVideoSource
    if browser_video:

        def screenshot_factory(*values, **settings):
            return ScreenshotProcess(*values, **settings, backend="mss")
    elif args.screenshot == "pil":
        screenshot_factory = ScreenshotThreadPIL
    elif args.screenshot == "mss":
        if not is_mss_supported():
            raise ValueError("mss is not supported on Wayland")

        def screenshot_factory(*args, **kwargs):
            return ScreenshotProcess(*args, **kwargs, backend="mss")
    elif args.screenshot == "wc_mp":

        def screenshot_factory(*args, **kwargs):
            return ScreenshotProcess(*args, **kwargs, backend="windows_capture")
    elif args.screenshot == "wc_cuda":

        def screenshot_factory(*args, **kwargs):
            return ScreenshotThreadWCCUDA(*args, **kwargs)
    elif args.screenshot == "kwcapture":
        reason = kwcapture_unsupported_reason()
        if reason is not None:
            raise ValueError(f"--screenshot kwcapture: {reason}")
        screenshot_factory = ScreenshotThreadKWCapture
    elif args.screenshot == "obs":
        reason = obs_unsupported_reason()
        if reason is not None:
            raise ValueError(f"--screenshot obs: {reason}")
        screenshot_factory = ScreenshotProcessOBS

    if args.screenshot == "pil" and is_wayland():
        if is_obs_supported() or is_kwcapture_supported():
            print(
                "Warning: 'pil' screenshot on Wayland captures the XWayland screen (usually "
                "black). Use --screenshot obs or --screenshot kwcapture.",
                file=sys.stderr,
            )
        else:
            print(f"Warning: {WAYLAND_CAPTURE_MISSING_MESSAGE}", file=sys.stderr)

    if args.screenshot in {"pil", "obs"} and args.monitor_index != 0:
        raise RuntimeError(f"{args.screenshot} does not support monitor_index={args.monitor_index}")
    if args.screenshot in {"pil", "obs"} and args.window_name:
        raise RuntimeError(f"{args.screenshot} does not support --window-name option")

    device = create_device(args.gpu)
    depth_model = args.state["depth_model"]

    screenshot_thread = None
    browser_bridge = None
    server = None
    try:
        if versioned_source:
            browser_bridge = BrowserVideoBridge()
            browser_bridge.start()
        if browser_video:
            screen_width, screen_height = 1920, 1080
        elif args.screenshot == "obs":
            screenshot_thread = screenshot_factory(
                fps=args.stream_fps,
                frame_width=None,
                frame_height=None,
                monitor_index=args.monitor_index,
                window_name=args.window_name,
                device=device,
                crop_top=args.crop_top,
                crop_left=args.crop_left,
                crop_right=args.crop_right,
                crop_bottom=args.crop_bottom,
                draw_cursor_enabled=not (args.disable_draw_cursor or (args.quest and args.quest_control)),
            )
            screen_width, screen_height = screenshot_thread.init_capture(stop_event=args.state["stop_event"])
        elif args.window_name:
            rect = get_window_rect_by_title(args.window_name)
            if rect is None:
                raise RuntimeError(f"window_name={args.window_name} not found")
            screen_width = rect["width"] - args.crop_left - args.crop_right
            screen_height = rect["height"] - args.crop_top - args.crop_bottom
        else:
            size_list = get_monitor_size_list()
            if args.monitor_index >= len(size_list):
                raise RuntimeError(f"monitor_index={args.monitor_index} not found")
            screen_width, screen_height = size_list[args.monitor_index]

        if not depth_model.loaded():
            depth_model.load(gpu=args.gpu, resolution=args.resolution, limit_resolution=args.limit_resolution)

        # Use Flicker Reduction to prevent 3D sickness
        if args.ema_normalize:
            depth_model.enable_ema(args.ema_decay, buffer_size=1)
        else:
            depth_model.disable_ema()
        if args.state["convergence_model"] is not None:
            args.state["convergence_model"].reset(enable_ema=True, decay=0.98)

        args.mapper = IW3U.resolve_mapper_name(
            mapper=args.mapper, foreground_scale=args.foreground_scale, metric_depth=depth_model.is_metric()
        )

        side_model = IW3U.create_stereo_model(
            args.method,
            divergence=args.divergence * (2.0 if args.synthetic_view in {"right", "left"} else 1.0),
            device_id=args.gpu[0],
        )

        screen_size = (screen_width, screen_height)
        if screen_height > args.stream_height:
            frame_height = args.stream_height
            frame_width = math.ceil((args.stream_height / screen_height) * screen_width)
            frame_width -= frame_width % 2
        else:
            frame_height = screen_height
            frame_width = screen_width

        if args.quest:
            frame_height -= frame_height % 2
            frame_width -= frame_width % 2
        desktop_frame_size = (frame_width, frame_height)
        desktop_screen_size = screen_size

        if screenshot_thread is not None and not browser_video:
            screenshot_thread.set_frame_size(frame_width, frame_height)

        if browser_video:
            output_frame_height, output_frame_width = frame_height, frame_width * 2
        else:
            output_frame_height, output_frame_width = test_output_size(
                (frame_height, frame_width), args, depth_model, side_model
            )
        if session is not None:
            session.set_stats(output_size=(output_frame_width, output_frame_height), bind_addr=args.bind_addr)

        lock = threading.Lock()
        if init_wxapp:
            empty_app = wx.App()  # noqa: F841 this is needed to initialize wx.GetMousePosition()

        if args.quest:
            from .quest_server import QuestServer

            if args.quest_control:
                from .quest_input import QuestController

                if browser_video:
                    from .browser_input import BrowserInput

                    backend = BrowserInput(browser_bridge)
                else:
                    from .quest_windows import WindowsInput

                    backend = WindowsInput(
                        args.window_name,
                        args.screenshot,
                        (args.crop_top, args.crop_left, args.crop_right, args.crop_bottom),
                        frame_mapping=lambda: (
                            screenshot_thread.get_frame_mapping() if screenshot_thread is not None else None
                        ),
                        monitor_index=args.monitor_index,
                    )
            server_factory = session.configure if session is not None else QuestServer
            server = server_factory(
                host=args.bind_addr,
                width=output_frame_width,
                height=output_frame_height,
                fps=args.stream_fps,
                bitrate=args.quest_bitrate,
                audio=not args.quest_no_audio,
                audio_source=args.quest_audio_source,
                audio_gain=args.quest_audio_gain,
                controller_factory=(lambda: QuestController(backend)) if args.quest_control else None,
                screen_curvature=args.quest_screen_curvature,
                ambient_enabled=args.quest_ambient,
                ambient_brightness=args.quest_ambient_brightness,
                ambient_spread=args.quest_ambient_spread,
                ambient_fill_borders=args.quest_ambient_fill_borders,
                ambient_feather=args.quest_ambient_feather,
                ambient_style=args.quest_ambient_style,
                ambient_360=args.quest_ambient_360,
                ambient_motion=args.quest_ambient_motion,
                reference_aspect=16 / 9 if browser_video else None,
            )
        elif not args.local_viewer:
            # Web Streaming
            with open(path.join(path.dirname(__file__), "views", "index.html.tpl"), mode="r", encoding="utf-8") as f:
                index_template = f.read()

            server = StreamingServer(
                host=args.bind_addr,
                port=args.port,
                lock=lock,
                frame_width=output_frame_width,
                frame_height=output_frame_height,
                fps=args.stream_fps,
                index_template=index_template,
                stream_uri="/stream.jpg",
                stream_content_type="image/jpeg",
                auth=auth,
            )
        else:
            # Local Viewer
            if LocalViewer is None:
                raise RuntimeError("Local Viewer is not available")
            IS_ROCM = getattr(torch.version, "hip", None) is not None
            USE_CUDA = torch.cuda.is_available() and not IS_ROCM
            server = LocalViewer(
                lock=lock,
                width=output_frame_width,
                height=output_frame_height,
                use_cuda=USE_CUDA,
                uncap_fps=args.uncap_fps,
                fit_window=args.fit_window,
            )

        desktop_backend = backend if args.quest_control and not browser_video else None
        initial_capture = screenshot_thread if not browser_video else None

        def create_desktop_capture():
            nonlocal \
                initial_capture, \
                desktop_backend, \
                desktop_frame_size, \
                desktop_screen_size, \
                frame_width, \
                frame_height, \
                screen_size
            if fixed_browser:
                # 纯视频失败时才查询桌面，避免正常取源依赖显示器或先启动录屏。
                sizes = get_monitor_size_list()
                if args.monitor_index >= len(sizes):
                    raise ValueError("Fallback monitor is unavailable")
                desktop_screen_size = sizes[args.monitor_index]
                width, height = desktop_screen_size
                frame_height = min(height, args.stream_height) // 2 * 2
                frame_width = int(width * frame_height / height) // 2 * 2
                desktop_frame_size = (frame_width, frame_height)
                screen_size = desktop_screen_size
                if args.quest_control:
                    from .quest_input import QuestController
                    from .quest_windows import WindowsInput

                    desktop_backend = WindowsInput(
                        None,
                        "mss",
                        (0, 0, 0, 0),
                        frame_mapping=lambda: screenshot_thread.get_frame_mapping(),
                        monitor_index=args.monitor_index,
                    )
                    server.set_controller_factory(lambda: QuestController(desktop_backend))
            if initial_capture is not None:
                capture, initial_capture = initial_capture, None
                return capture
            capture = screenshot_factory(
                fps=args.stream_fps,
                frame_width=desktop_frame_size[0],
                frame_height=desktop_frame_size[1],
                monitor_index=args.monitor_index,
                window_name=None if fixed_browser else args.window_name,
                device=device,
                crop_top=args.crop_top,
                crop_left=args.crop_left,
                crop_right=args.crop_right,
                crop_bottom=args.crop_bottom,
                draw_cursor_enabled=not (args.disable_draw_cursor or (args.quest and args.quest_control)),
                **(
                    {
                        "window_handle": desktop_backend.hwnd,
                        "monitor_target": desktop_backend.monitor_target,
                        "fit_window": True,
                    }
                    if args.quest_control
                    else {}
                ),
            )
            if args.screenshot == "obs":
                try:
                    capture.init_capture(stop_event=args.state["stop_event"])
                    capture.set_frame_size(*desktop_frame_size)
                except Exception:
                    capture.stop()
                    raise
            return capture

        count = last_status_time = 0
        fps_counter = deque(maxlen=120)
        prev_divergence = args.divergence
        browser_blanked = False
        processed_source_generation = None

        def blank_browser_frame():
            nonlocal browser_blanked
            if not browser_blanked:
                server.discard_video_frames()
                black = torch.zeros((3, output_frame_height, output_frame_width), device=device)
                server.set_frame_data((black, time.perf_counter()))
                browser_blanked = True

        def switch_source(kind, epoch):
            nonlocal browser_video, browser_blanked, frame_width, frame_height, screen_size
            browser_video = kind == "browser"
            # 同一连接/音频 owner 保持；先释放旧输入，再屏障旧 SBS 与环境光。
            server.set_controller_factory(None)
            browser_blanked = False
            blank_browser_frame()
            server.set_reference_aspect(16 / 9 if browser_video else None)
            if not browser_video:
                frame_width, frame_height = desktop_frame_size
                screen_size = desktop_screen_size
            if args.quest_control:
                from .browser_input import BrowserInput
                from .quest_input import QuestController

                selected_backend = BrowserInput(browser_bridge) if browser_video else desktop_backend
                server.set_controller_factory(
                    (lambda: QuestController(selected_backend)) if selected_backend is not None else None
                )
            if session is not None:
                session.set_stats(source_kind=kind)
            elif args.state["fps_event"] is not None and hasattr(args.state["fps_event"], "set_source_kind"):
                args.state["fps_event"].set_source_kind(kind)

        if versioned_source:
            from .source_router import SourceRouter

            screenshot_thread = SourceRouter(
                browser_bridge,
                create_desktop_capture,
                lambda: MediaVideoSource(
                    browser_bridge.snapshot,
                    device,
                    fps=args.stream_fps,
                    quality=args.browser_quality,
                    state=browser_bridge.status,
                ),
                switch_source,
                fixed_browser=fixed_browser,
            )
        elif screenshot_thread is None:
            screenshot_thread = create_desktop_capture()

        if args.compile:
            depth_model.compile()
            if side_model is not None and not isinstance(side_model, DeviceSwitchInference):
                side_model = compile_model(side_model, device=device)
        gc_collect()

        # main loop
        server.start()
        screenshot_thread.start()
        if session is not None:
            session.capture = screenshot_thread

        if args.quest:
            print("Open Pop3D on Quest (close the Pop3D PC app first). UDP 52300 / TCP 52301")
            if args.state["fps_event"] is not None:
                args.state["fps_event"].set_url(None)
        elif not args.local_viewer:
            if args.state["fps_event"] is not None:
                args.state["fps_event"].set_url(f"http://{args.bind_addr}:{args.port}")
            else:
                print(f"Open http://{args.bind_addr}:{args.port}")
        else:
            if args.state["fps_event"] is not None:
                args.state["fps_event"].set_url(None)

        if browser_video and not auto_browser:
            blank_browser_frame()

        while True:
            if args.state["stop_event"] and args.state["stop_event"].is_set():
                break
            if versioned_source:
                screenshot_thread.poll_route()
            if versioned_source and session is None and not screenshot_thread.frame_set_event.wait(0.1):
                frame = None
            else:
                frame = session.read_frame(screenshot_thread) if session is not None else screenshot_thread.get_frame()
            if versioned_source:
                browser_video = screenshot_thread.source_kind == "browser"
            if frame is None:
                if versioned_source and screenshot_thread.is_alive():
                    generation = screenshot_thread.get_frame_generation()
                    if not screenshot_thread.is_current_generation(generation):
                        blank_browser_frame()
                    else:
                        server.refresh_ambient_frame()
                    continue
                break
            source_generation = screenshot_thread.get_frame_generation() if versioned_source else None
            if versioned_source:
                if not screenshot_thread.is_current_generation(source_generation):
                    blank_browser_frame()
                    continue
            if browser_video:
                # 源为CUDA float32[3,H,W]；奇数边只复制末像素对齐NV12，不裁内容、不套桌面高度。
                screen_height, screen_width = frame.shape[-2:]
                screen_size = (screen_width, screen_height)
                frame_height = screen_height + screen_height % 2
                frame_width = screen_width + screen_width % 2
                if min(frame_height, frame_width) < 2:
                    raise ValueError("Browser video dimensions are too small")
                if frame.shape[-2:] != (frame_height, frame_width):
                    frame = torch.nn.functional.pad(
                        frame, (0, screen_width % 2, 0, screen_height % 2), mode="replicate"
                    )
            tick = time.perf_counter()
            with args.state["args_lock"]:
                ambient_enabled = args.quest and args.quest_ambient
            content_rect = (
                (0, 0, frame_width, frame_height)
                if browser_video
                else screenshot_thread.get_content_rect()
                if ambient_enabled and hasattr(screenshot_thread, "get_content_rect")
                else None
            )
            with args.state["args_lock"]:
                if args.quest:
                    server.check_error()
                    server.set_audio_gain(args.quest_audio_gain)
                if prev_divergence != args.divergence:
                    if args.method in {"mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s"}:
                        side_model = try_switch_mlbw_model(prev_divergence, args.divergence, side_model, args)
                    prev_divergence = args.divergence
                if versioned_source and source_generation != processed_source_generation:
                    blank_browser_frame()
                    # 换条清空深度/会聚时序，不重载权重；reset 保留现有 EMA decay 和 buffer_size。
                    depth_model.reset()
                    if args.state["convergence_model"] is not None:
                        args.state["convergence_model"].reset(enable_ema=True, decay=0.98)
                    processed_source_generation = source_generation
                sbs = IW3U.process_image(frame, args, depth_model, side_model, autocrop_uncrop=True)

            if versioned_source:
                if not screenshot_thread.is_current_generation(source_generation):
                    blank_browser_frame()
                    continue
                height, width = sbs.shape[-2:]
                if (height, width) != (output_frame_height, output_frame_width):
                    server.configure_video(width, height, args.stream_fps, args.quest_bitrate)
                    output_frame_height, output_frame_width = height, width
                    if session is not None:
                        session.set_stats(output_size=(width, height))
                if not screenshot_thread.is_current_generation(source_generation):
                    blank_browser_frame()
                    continue
                browser_blanked = False
            if ambient_enabled or browser_video:
                # 原单目[3,H,W]负责取色，最终FullSBS[3,H2,2W2]负责遮罩，同一包提交且不持设置锁。
                server.set_ambient_frame(frame, content_rect=content_rect, sbs=sbs)
            with args.state["args_lock"]:
                if args.quest:
                    if versioned_source and not screenshot_thread.is_current_generation(source_generation):
                        blank_browser_frame()
                        continue
                    server.set_frame_data((sbs, tick))
                    if session is not None and server.get_fps() > 0:
                        session.media_ready()
                elif not args.local_viewer:
                    if args.gpu_jpeg:
                        server.set_frame_data(
                            to_jpeg_data(sbs, quality=args.stream_quality, tick=tick, gpu_jpeg=args.gpu_jpeg)
                        )
                    else:
                        with torch.no_grad():
                            sbs_gpu_clone = sbs.detach().clone().contiguous()
                        server.set_frame_data(
                            lambda sbs=sbs_gpu_clone: to_jpeg_data(
                                sbs, quality=args.stream_quality, tick=tick, gpu_jpeg=False
                            )
                        )
                else:
                    server.set_frame_data((sbs, tick))

                if count > 1 and tick - last_status_time > 1:
                    last_status_time = tick
                    mean_processing_time = sum(fps_counter) / len(fps_counter)
                    estimated_fps = 1.0 / mean_processing_time
                    screen_size_tuple = (screen_size, (frame_width, frame_height))
                    if session is not None:
                        session.set_stats(
                            estimated_fps=estimated_fps,
                            screenshot_fps=screenshot_thread.get_fps(),
                            streaming_fps=server.get_fps(),
                            screen_surface="曲面配置已发送"
                            if server.actual_cylinder
                            else "平面配置（曲面须接收端支持）",
                        )
                    if args.state["fps_event"] is not None:
                        args.state["fps_event"].update(
                            estimated_fps, screenshot_thread.get_fps(), server.get_fps(), screen_size_tuple
                        )
                    else:
                        print(
                            f"\rEstimated FPS = {estimated_fps:.02f}, "
                            f"{'NVDEC' if browser_video else 'Screenshot'} FPS = {screenshot_thread.get_fps():.02f}, "
                            f"Streaming FPS = {server.get_fps():.02f}, "
                            f"Screen Size = {screen_size_tuple}",
                            end="",
                        )

            if args.local_viewer:
                # Platform-specific behavior for local viewer
                if sys.platform != "win32":
                    # Linux: Use wx.Yield() (original working behavior)
                    if not wx.GetApp().IsMainLoopRunning():
                        wx.Yield()
                # Check if window was closed
                if server.is_closed():
                    break

            process_time = time.perf_counter() - tick
            wait_time = max((1 / (args.stream_fps)) - process_time, 0)
            if session is not None:
                session.run_stop.wait(wait_time)
            else:
                time.sleep(wait_time)
            fps_counter.append(process_time)
            count += 1
            if args.state["stop_event"] and args.state["stop_event"].is_set():
                break
    finally:
        try:
            if server is not None and session is None:
                server.stop()
        finally:
            try:
                if screenshot_thread is not None:
                    screenshot_thread.stop()
            finally:
                try:
                    if browser_bridge is not None:
                        browser_bridge.stop()
                finally:
                    depth_model.clear_compiled_model()
                    depth_model.reset()
                    gc_collect()

    if args.state["stop_event"] and args.state["stop_event"].is_set():
        args.state["stop_event"].clear()

    return args
