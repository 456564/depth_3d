"""Optional local TensorRT-RTX forward used by the 4K60 experiment launcher."""

import logging
import os
from pathlib import Path
from typing import Any

import torch

from ..depth_models.depth_anything_model import _forward

INPUT_SHAPE = (1, 3, 392, 686)
OUTPUT_SHAPE = (1, 1, 392, 686)
LOGGER = logging.getLogger(__name__)


class QuestTensorRTForward:
    @torch.inference_mode()
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        if os.getenv("IW3_4K60_EXPERIMENT") != "1":
            raise RuntimeError("4K60 experiment mode is disabled")
        plan_name = os.getenv("IW3_TRT_PLAN")
        site = os.getenv("IW3_TRT_SITE")
        if not plan_name or not site:
            raise RuntimeError("TensorRT-RTX experiment paths are not configured")
        import sys

        if site not in sys.path:
            sys.path.insert(0, site)
        try:
            import tensorrt_rtx_bindings as trt
        except Exception as exc:
            raise RuntimeError(f"TensorRT-RTX import failed: {exc}") from exc
        plan = Path(plan_name)
        if not plan.is_file():
            raise RuntimeError(f"TensorRT plan is missing: {plan}")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(plan.read_bytes())
        if engine is None:
            raise RuntimeError("TensorRT plan is incompatible with this GPU")
        self.model = model
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.owner_stream = torch.cuda.current_stream(device)
        self.stream.wait_stream(self.owner_stream)
        self.runtime = runtime
        self.engine = engine
        self.config = engine.create_runtime_config()
        self.config.cuda_graph_strategy = trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
        self.cache = None
        cache_name = os.getenv("IW3_TRT_CACHE")
        if cache_name and Path(cache_name).is_file():
            self.cache = self.config.create_runtime_cache()
            if not self.cache.deserialize(Path(cache_name).read_bytes()):
                raise RuntimeError("TensorRT runtime cache is incompatible")
            if not self.config.set_runtime_cache(self.cache):
                raise RuntimeError("TensorRT runtime cache could not be attached")
        self.context = engine.create_execution_context(self.config)
        if self.context is None:
            raise RuntimeError("TensorRT execution context creation failed")
        self.input = torch.empty(INPUT_SHAPE, device=device, dtype=torch.float32)
        self.output = torch.empty(OUTPUT_SHAPE, device=device, dtype=torch.float32)
        if tuple(engine.get_tensor_shape("rgb")) != INPUT_SHAPE or tuple(engine.get_tensor_shape("depth")) != OUTPUT_SHAPE:
            raise RuntimeError("TensorRT plan shape does not match Any_V2_S")
        if not self.context.set_tensor_address("rgb", self.input.data_ptr()):
            raise RuntimeError("TensorRT input binding failed")
        if not self.context.set_tensor_address("depth", self.output.data_ptr()):
            raise RuntimeError("TensorRT output binding failed")
        self.enabled = False
        self.used = False
        self._warmup()

    def _warmup(self) -> None:
        with torch.cuda.stream(self.stream):
            self.input.zero_()
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT warmup failed")
        self.owner_stream.wait_stream(self.stream)

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
            and torch.cuda.current_stream(x.device) == self.owner_stream
        ):
            return _forward(model, x, enable_amp)
        self.stream.wait_stream(self.owner_stream)
        try:
            with torch.cuda.stream(self.stream):
                self.input.copy_(x)
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT execution failed")
                result = self.output.clone()
                result.record_stream(self.owner_stream)
        finally:
            self.owner_stream.wait_stream(self.stream)
        self.used = True
        return result

    def close(self) -> None:
        self.owner_stream.synchronize()
        self.stream.synchronize()


def try_create(model: torch.nn.Module, device: torch.device) -> Any | None:
    try:
        return QuestTensorRTForward(model, device)
    except Exception as exc:
        LOGGER.warning("4K60 TensorRT disabled; using Torch Graph: %s", exc)
        return None
