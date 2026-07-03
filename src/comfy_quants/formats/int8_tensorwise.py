"""Stock-ComfyUI INT8 (+ optional ConvRot) reusable format declaration.

Storage format for the OFFLINE producer of checkpoints loadable by **stock ComfyUI**'s
built-in ``QUANT_ALGOS["int8_tensorwise"]`` path (ComfyUI >= v0.27.0, comfy-kitchen
``TensorWiseINT8Layout``, SM >= 7.5 / Turing+): native ``torch.int8`` weights with a
symmetric per-output-channel ``float32`` scale, plus a per-layer ``comfy_quant`` marker
that — unlike the retired ComfyUI-INT8-Fast contract in :mod:`formats.int8_w8a8` —
**must carry a** ``"format"`` **key** (stock's loader raises ``ValueError`` without it)
and has no ``per_row`` key. Activations are quantized dynamically inside the
comfy-kitchen kernel (W8A8), so NO ``input_scale`` is stored.

The two int8 formats share the storage tensors (int8 weight + fp32 [out,1] scale +
identical regular-Hadamard ConvRot); they differ in the marker payload and in the
bit-faithfulness oracle (this format mirrors comfy-kitchen's division-based
``quantize_int8_rowwise``; int8_w8a8 mirrors INT8-Fast's reciprocal-multiply).
"""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.formats.convrot import CONVROT_GROUP_SIZE
from comfy_quants.registry.global_registry import registry

__all__ = [
    "INT8_TENSORWISE_FORMAT_NAME",
    "INT8_TENSORWISE_FORMAT",
    "int8_tensorwise_checkpoint_quant_config",
]

INT8_TENSORWISE_FORMAT_NAME = "int8_tensorwise"


def int8_tensorwise_checkpoint_quant_config(
    *, convrot: bool, convrot_groupsize: int = CONVROT_GROUP_SIZE
) -> dict[str, str | bool | int]:
    """Return the per-layer ``comfy_quant`` marker payload.

    Keys/insertion-order match stock ComfyUI's save path
    (``ops.py:_quantized_weight_state_dict``): ``format`` first, then — only when the
    layer was actually rotated — ``convrot`` and ``convrot_groupsize``.
    ``full_precision_matrix_mult`` is deliberately omitted (mxfp8/nvfp4 precedent).
    """
    conf: dict[str, str | bool | int] = {"format": INT8_TENSORWISE_FORMAT_NAME}
    if convrot:
        conf["convrot"] = True
        conf["convrot_groupsize"] = int(convrot_groupsize)
    return conf


INT8_TENSORWISE_FORMAT = QuantFormatSpec(
    name=INT8_TENSORWISE_FORMAT_NAME,
    storage_dtype="int8",
    bits=8,
    category="integer_weight_activation",
    scale_required=True,
    default_scale_granularity="per_channel",  # per output channel (axis = out_features)
    compatible_families=(
        "qwen_image",
        "qwen_image_edit",
        "qwen_image_layered",
        "anima",
        "anima_14b",
        "flux",
        "flux2",
        "ltxv",
        "ltx2",
        "ideogram4",
    ),
    notes=(
        "Stock-ComfyUI int8: symmetric per-row int8 weights + dynamic int8 activations (runtime).",
        "Optional offline ConvRot (regular Hadamard) weight rotation; group size 256.",
        "Loaded by stock ComfyUI QUANT_ALGOS[int8_tensorwise] / TensorWiseINT8Layout (SM>=7.5, ComfyUI>=0.27).",
        "Marker MUST carry format=int8_tensorwise; contrast the retired INT8-Fast int8_w8a8 marker.",
        "No input_scale: activations quantized dynamically at runtime.",
    ),
    metadata={
        "weight_tensor": "weight",
        "scale_tensor": "weight_scale",
        "marker_tensor": "comfy_quant",
        "weight_scale_shape": "per_row_2d_out_1",
        "convrot_group_size": CONVROT_GROUP_SIZE,
        "symmetric": True,
        "quant_min": -128,
        "quant_max": 127,
        "no_input_scale": True,
        "downstream_loader": "stock ComfyUI QUANT_ALGOS[int8_tensorwise] (comfy-kitchen TensorWiseINT8Layout)",
    },
)


registry.register_format(INT8_TENSORWISE_FORMAT)
