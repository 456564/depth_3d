import ctypes as c
import logging
import os
import sys
from functools import cache
from pathlib import Path
from threading import Lock

import torch

from nunif.utils.video.color_transform import ColorRange, Colorspace, ColorTransform, OutputTransform
from nunif.utils.video.metadata import COLORSPACE_UNSPECIFIED

_PTR = c.c_void_p
_LOG = logging.getLogger(__name__)
_LOCK = Lock()


def _bind(dll, name, *args):
    func = getattr(dll, name)
    func.argtypes = args
    func.restype = c.c_int
    return func


def _check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed ({status})")


@cache
def _compile_ptx():
    lib = Path(torch.__file__).parent / "lib"
    with os.add_dll_directory(str(lib)):
        builtins = c.WinDLL(str(lib / "nvrtc-builtins64_132.dll"))
        nvrtc = c.WinDLL(str(lib / "nvrtc64_130_0.dll"))
        program = _PTR()
        create = _bind(nvrtc, "nvrtcCreateProgram", _PTR, c.c_char_p, c.c_char_p, c.c_int, _PTR, _PTR)
        source = Path(__file__).with_suffix(".cu").read_bytes()
        _check(create(c.byref(program), source, b"quest_nv12.cu", 0, None, None), "nvrtcCreateProgram")
        try:
            flags = (c.c_char_p * 3)(b"--gpu-architecture=compute_89", b"--fmad=false", b"--ftz=false")
            compile_program = _bind(nvrtc, "nvrtcCompileProgram", _PTR, c.c_int, _PTR)
            _check(compile_program(program, len(flags), flags), "nvrtcCompileProgram")
            size = c.c_size_t()
            _check(_bind(nvrtc, "nvrtcGetPTXSize", _PTR, _PTR)(program, c.byref(size)), "nvrtcGetPTXSize")
            ptx = c.create_string_buffer(size.value)
            _check(_bind(nvrtc, "nvrtcGetPTX", _PTR, _PTR)(program, ptx), "nvrtcGetPTX")
            return ptx
        finally:
            _bind(nvrtc, "nvrtcDestroyProgram", _PTR)(c.byref(program))
            del builtins


@cache
def _driver():
    try:
        dll = c.WinDLL(str(Path(os.environ["SystemRoot"]) / "System32" / "nvcuda.dll"))
        _bind(dll, "cuCtxGetCurrent", _PTR)
        _bind(dll, "cuCtxGetId", _PTR, _PTR)
        _bind(dll, "cuModuleLoadData", _PTR, _PTR)
        _bind(dll, "cuModuleUnload", _PTR)
        _bind(dll, "cuModuleGetFunction", _PTR, _PTR, c.c_char_p)
        _bind(dll, "cuLaunchKernel", _PTR, *([c.c_uint] * 7), _PTR, _PTR, _PTR)
        return dll
    except (OSError, AttributeError, KeyError) as error:
        _LOG.warning("Quest 4K CUDA conversion unavailable: %s", error)
        return None


@cache
def _kernel(driver, context_id, name=b"pack_nv12"):
    del context_id  # CUDA 在调用线程的当前 context 加载模块；缓存区分 context 生命周期。
    try:
        ptx = _compile_ptx()
    except (OSError, AttributeError, RuntimeError) as error:
        _LOG.warning("Quest 4K CUDA conversion unavailable: %s", error)
        return None
    module, function = _PTR(), _PTR()
    status = driver.cuModuleLoadData(c.byref(module), ptx)
    if status in (200, 209, 218, 221, 222, 223):  # 镜像、架构或 PTX 不受支持。
        _LOG.warning("Quest 4K NV12 kernel unavailable: cuModuleLoadData failed (%s)", status)
        return None
    _check(status, "cuModuleLoadData")
    status = driver.cuModuleGetFunction(c.byref(function), module, name)
    if status == 500:  # 查找不到入口；没有任务曾用过这个模块。
        _check(driver.cuModuleUnload(module), "cuModuleUnload")
        _LOG.warning("Quest 4K NV12 kernel entry %r is missing", name)
        return None
    _check(status, "cuModuleGetFunction")
    return function  # 模块留到 context 销毁，避免尚在流中的 kernel 被卸载。


@cache
def _supported(device):
    return (
        sys.platform == "win32"
        and torch.__version__ == "2.14.1+cu132"
        and torch.backends.cudnn.version() == 92400
        and torch.cuda.get_device_capability(device) == (8, 9)
    )


def _compatible(device: torch.device) -> bool:
    return (
        _supported(device)
        and not torch.cuda.is_current_stream_capturing()
        and not torch.is_autocast_enabled("cuda")
        and torch.backends.cudnn.enabled
        and torch.backends.cudnn.conv.fp32_precision == "tf32"
        and not torch.backends.cudnn.benchmark
        and not torch.backends.cudnn.deterministic
        and not torch.are_deterministic_algorithms_enabled()
    )


def _current_kernel(name):
    driver = _driver()
    if driver is None:
        return None
    context, context_id = _PTR(), c.c_ulonglong()
    _check(driver.cuCtxGetCurrent(c.byref(context)), "cuCtxGetCurrent")
    if not context:
        return None
    _check(driver.cuCtxGetId(context, c.byref(context_id)), "cuCtxGetId")
    with _LOCK:
        kernel = _kernel(driver, context_id.value, name)
    return (driver, kernel) if kernel is not None else None


def _launch(entry, grid, block, stream, args):
    driver, kernel = entry
    params = (_PTR * len(args))(*(c.addressof(arg) for arg in args))
    status = driver.cuLaunchKernel(kernel, *grid, *block, 0, _PTR(stream.cuda_stream), params, None)
    _check(status, "cuLaunchKernel")


def try_pack_nv12(
    frame: torch.Tensor, transform: OutputTransform, stream: torch.cuda.Stream
) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    # shortcut: 只用在逐字节核对过的 4K/Ada/cuDNN 配置；扩展前需重测卷积舍入。
    if (
        frame.shape not in {(3, 2160, 7680), (3, 3840, 4320)}
        or frame.dtype != torch.float32
        or not frame.is_contiguous()
        or transform.dst_pix_fmt not in {"nv12", "yuv420p", "yuvj420p"}
        or transform.dst_colorspace != Colorspace.ITU709
        or transform.dst_color_range != ColorRange.MPEG
        or os.environ.get("IW3_DISABLE_QUEST_FUSED_NV12") == "1"
        or not _compatible(frame.device)
    ):
        return None
    entry = _current_kernel(b"pack_nv12")
    if entry is None:
        return None

    # CUDA FP32 [3,H,2W] → 连续 U8 [H*3/2,2W]；NVENC 从同一基址读 Y/UV。
    height, width = frame.shape[-2:]
    packed = torch.empty((height + height // 2, width), dtype=torch.uint8, device=frame.device)
    args = _PTR(frame.data_ptr()), _PTR(packed.data_ptr()), c.c_int(height), c.c_int(width)
    blocks = (height * width // 4 + 255) // 256
    _launch(entry, (blocks, 1, 1), (256, 1, 1), stream, args)
    return packed[:height], packed[height:].view(height // 2, width // 2, 2), "nv12"


def try_unpack_nv12(
    y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, colorspace: Colorspace | int, color_range: ColorRange | int
) -> torch.Tensor | None:
    if (
        y.shape not in {(1, 1, 2160, 3840), (1, 1, 3840, 2160)}
        or u.shape != (1, 1, y.shape[-2] // 2, y.shape[-1] // 2)
        or u.shape != v.shape
        or not y.is_cuda
        or y.device != u.device
        or y.device != v.device
        or y.dtype != torch.float32
        or y.dtype != u.dtype
        or y.dtype != v.dtype
        or y.stride(-1) != 1
        or u.stride()[-2:] != v.stride()[-2:]
        or colorspace not in {Colorspace.ITU709, COLORSPACE_UNSPECIFIED}
        or color_range not in {ColorRange.UNSPECIFIED, ColorRange.MPEG}
        or os.environ.get("IW3_DISABLE_BROWSER_FUSED_NV12") == "1"
        or not _compatible(y.device)
    ):
        return None
    entry = _current_kernel(b"unpack_nv12")
    if entry is None:
        return None

    # Torch 已拥有 FP32 平面；不让新 CUDA 核借用可被 NVDEC 重用的解码表面。
    matrix = ColorTransform._get_matrix(colorspace, y.device, y.dtype, "yuv2rgb", color_range, 255.0)
    assert isinstance(matrix, tuple)
    weight, bias = matrix
    height, width = y.shape[-2:]
    rgb = y.new_empty((1, 3, height, width))
    args = tuple(_PTR(t.data_ptr()) for t in (y, u, v, weight, bias, rgb)) + (
        c.c_int(height),
        c.c_int(width),
        c.c_int(y.stride(-2)),
        c.c_int(u.stride(-2)),
        c.c_int(u.stride(-1)),
    )
    _launch(entry, ((width + 31) // 32, (height + 7) // 8, 1), (32, 8, 1), torch.cuda.current_stream(y.device), args)
    return rgb


def try_warp_sbs(rgb: torch.Tensor, grid: torch.Tensor, delta: torch.Tensor, scale: float) -> torch.Tensor | None:
    # shortcut: 仅对已逐位验证的 4K/392/Ada FP32 布局启用，其他布局走 PyTorch。
    layout = (tuple(rgb.shape), tuple(grid.shape))
    if layout == ((1, 3, 2160, 3840), (1, 2, 392, 686)):
        name = b"warp_sbs_landscape"
    elif layout == ((1, 3, 3840, 2160), (1, 2, 686, 392)):
        name = b"warp_sbs_portrait"
    else:
        return None
    if (
        not rgb.is_cuda
        or not rgb.is_contiguous()
        or rgb.dtype != torch.float32
        or grid.dtype != rgb.dtype or delta.dtype != rgb.dtype
        or grid.device != rgb.device or delta.device != rgb.device
        or delta.shape != grid.shape
        or os.environ.get("IW3_DISABLE_QUEST_FUSED_STEREO") == "1"
        or not _compatible(rgb.device)
    ):
        return None
    with torch.cuda.device(rgb.device):
        entry = _current_kernel(name)
        if entry is None:
            return None
        # [1,3,H,W] + 两眼 [1,2,GH,GW] → 独立存储的 FP32 [3,H,2W]。
        left = (grid + delta * scale).contiguous()
        right = (grid + (-delta) * scale).contiguous()
        height, width = rgb.shape[-2:]
        packed = rgb.new_empty((3, height, width * 2))
        params = tuple(_PTR(t.data_ptr()) for t in (rgb, left, right, packed))
        blocks = (2 * height * width + 255) // 256
        _launch(entry, (blocks, 1, 1), (256, 1, 1), torch.cuda.current_stream(rgb.device), params)
        return packed
