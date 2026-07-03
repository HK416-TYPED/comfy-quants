"""NVFP4 (FP4 E2M1 microscaling) reusable format declaration.

Storage format for the OFFLINE producer of **stock-ComfyUI-native** NVFP4
checkpoints: FP4-E2M1 weights packed 2-per-byte, with two-level scaling — a
per-block-16 FP8-E4M3 ``weight_scale`` (cuBLAS ``to_blocked`` swizzle) and a
per-tensor float32 ``weight_scale_2`` — plus a per-layer ``comfy_quant`` marker.
Block math/pack/swizzle live in :mod:`comfy_quants.formats.nvfp4_blocked`.

The consumer is **stock ComfyUI** via ``QUANT_ALGOS["nvfp4"]`` + ``TensorCoreNVFP4Layout``
(same per-layer ``comfy_quant`` handshake as FP8/MXFP8). The nvfp4 tensor-core matmul
is gated at runtime on Blackwell (SM>=10) + comfy_kitchen; on other hardware ComfyUI
silently dequantizes to the compute dtype (loads & correct, no speedup). No
``input_scale`` is stored (activations are quantized dynamically at runtime).

ConvRot (EXPERIMENTAL, runtime-pending): the marker optionally carries ``convrot`` /
``convrot_groupsize`` keys following the ``int8_tensorwise`` marker convention
(ConvRot paper, arXiv 2512.03673: group-wise regular-Hadamard rotation before 4-bit
quantization). **No released runtime honors these keys on nvfp4 yet** — today's stock
ComfyUI loader ignores unknown marker keys on the nvfp4 branch, so a rotated
checkpoint would load and silently produce wrong outputs. Rotated exports are for
runtime bring-up only until the comfy-kitchen/ComfyUI support lands.
"""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.formats.convrot import CONVROT_GROUP_SIZE
from comfy_quants.formats.nvfp4_blocked import BLOCK_SIZE
from comfy_quants.registry.global_registry import registry

__all__ = ["NVFP4_FORMAT_NAME", "NVFP4_FORMAT", "nvfp4_checkpoint_quant_config"]

NVFP4_FORMAT_NAME = "nvfp4"


def nvfp4_checkpoint_quant_config(
    *, convrot: bool = False, convrot_groupsize: int = CONVROT_GROUP_SIZE
) -> dict[str, str | bool | int]:
    """Return the per-layer ``comfy_quant`` marker payload.

    ComfyUI's loader reads only ``format`` (``"nvfp4"`` is the exact ``QUANT_ALGOS``
    key); we do not set ``full_precision_matrix_mult`` (we want the nvfp4 kernel; on
    unsupported hardware ComfyUI auto-disables and dequantizes anyway).

    Keys/insertion-order follow stock ComfyUI's ``int8_tensorwise`` save path
    (``ops.py:_quantized_weight_state_dict``): ``format`` first, then — only when the
    layer was actually rotated — ``convrot`` and ``convrot_groupsize``.
    """
    conf: dict[str, str | bool | int] = {"format": NVFP4_FORMAT_NAME}
    if convrot:
        conf["convrot"] = True
        conf["convrot_groupsize"] = int(convrot_groupsize)
    return conf


NVFP4_FORMAT = QuantFormatSpec(
    name=NVFP4_FORMAT_NAME,
    storage_dtype="uint8",  # abstract bit-container; on disk: weight is packed fp4 (uint8), block scale is fp8
    bits=4,
    category="floating_point_block_scaled",
    scale_required=True,
    default_scale_granularity="block",
    compatible_families=("qwen_image", "qwen_image_edit", "qwen_image_layered", "anima", "anima_14b", "flux", "flux2", "ltxv", "ltx2", "ideogram4"),
    notes=(
        "NVFP4: FP4-E2M1 weights packed 2/byte (uint8) + per-block-16 fp8 scale + per-tensor fp32 scale.",
        "weight_scale (block) stored as float8_e4m3fn in the cuBLAS to_blocked swizzle.",
        "weight_scale_2 (per-tensor) stored as a float32 scalar = amax(|W|)/(448*6).",
        "Loaded by stock ComfyUI QUANT_ALGOS[nvfp4] / TensorCoreNVFP4Layout (Blackwell SM>=10).",
        "No input_scale: activations quantized dynamically at runtime.",
        "Optional offline ConvRot (regular Hadamard, group 256) weight rotation — EXPERIMENTAL, no released runtime honors the marker keys yet.",
    ),
    metadata={
        "block_size": BLOCK_SIZE,
        "weight_tensor": "weight",
        "weight_torch_dtype": "uint8",  # fp4_e2m1 packed 2/byte, shape [out, in//2]
        "scale_tensor": "weight_scale",
        "scale_dtype": "float8_e4m3fn",
        "scale_layout": "to_blocked_swizzled",
        "per_tensor_scale_tensor": "weight_scale_2",
        "per_tensor_scale_dtype": "float32",
        "marker_tensor": "comfy_quant",
        "marker_format": NVFP4_FORMAT_NAME,
        "convrot_group_size": CONVROT_GROUP_SIZE,  # optional rotation; marker keys only when rotated
        "no_input_scale": True,
        "downstream_loader": "stock ComfyUI QUANT_ALGOS[nvfp4] (Blackwell)",
    },
)


registry.register_format(NVFP4_FORMAT)
