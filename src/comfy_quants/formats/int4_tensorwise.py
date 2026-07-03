"""Stock-ComfyUI-track INT4 (int4_tensorwise, + optional ConvRot) format declaration.

Storage format for the OFFLINE producer of W4A4 checkpoints targeting the
comfy-kitchen ``TensorWiseINT4Layout`` runtime (kitchen branch
``feat/int4-tensorwise-convrot``; the matching stock-ComfyUI ``QUANT_ALGOS``
entry is in flight — mirror of the ``int8_tensorwise`` rollout sequence).

The format is the **4-bit downward extension of** :mod:`formats.int8_tensorwise`:
signed INT4 weights packed 2-per-byte (LOW nibble first, ``int8`` container,
shape ``[out, in//2]``) with a symmetric per-output-channel ``float32`` scale
``[out, 1]`` and the same optional regular-Hadamard ConvRot rotation. The
per-layer ``comfy_quant`` marker carries a **required** ``format`` key plus
``convrot``/``convrot_groupsize`` only when the layer was actually rotated.
Activations are rotated online and quantized dynamically per-token inside the
comfy-kitchen ``int4_linear`` kernel (W4A4) — NO ``input_scale`` is stored.

Quant math is bit-faithful to comfy-kitchen eager ``quantize_int4_rowwise`` /
``quantize_int4_convrot_weight``: rotation and DIVISION at the source weight
dtype, fp32 scale = ``amax/7`` clamped to >= 1e-30, symmetric emission clamped
to ``[-7, 7]`` (the kitchen int4 emission contract — -8 is never emitted).

Mixed-precision note: the ConvRot paper's quality recipe keeps ~20% of layers
(functionally sensitive, not outlier-driven) at W8A8. The int4_tensorwise
exporter therefore supports per-layer INT8 fallback — those layers are written
with the :mod:`formats.int8_tensorwise` tensors and marker in the same
checkpoint (ComfyUI resolves ``QUANT_ALGOS`` per layer from each marker).
"""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.formats.convrot import CONVROT_GROUP_SIZE
from comfy_quants.registry.global_registry import registry

__all__ = [
    "INT4_TENSORWISE_FORMAT_NAME",
    "INT4_TENSORWISE_FORMAT",
    "int4_tensorwise_checkpoint_quant_config",
]

INT4_TENSORWISE_FORMAT_NAME = "int4_tensorwise"


def int4_tensorwise_checkpoint_quant_config(
    *, convrot: bool, convrot_groupsize: int = CONVROT_GROUP_SIZE
) -> dict[str, str | bool | int]:
    """Return the per-layer ``comfy_quant`` marker payload.

    Keys/insertion-order mirror the ``int8_tensorwise`` marker (ComfyUI
    ``ops.py:_quantized_weight_state_dict`` save order): ``format`` first, then —
    only when the layer was actually rotated — ``convrot`` and
    ``convrot_groupsize``.
    """
    conf: dict[str, str | bool | int] = {"format": INT4_TENSORWISE_FORMAT_NAME}
    if convrot:
        conf["convrot"] = True
        conf["convrot_groupsize"] = int(convrot_groupsize)
    return conf


INT4_TENSORWISE_FORMAT = QuantFormatSpec(
    name=INT4_TENSORWISE_FORMAT_NAME,
    storage_dtype="int8",  # container: two signed int4 codes per byte, low nibble first
    bits=4,
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
        "W4A4: symmetric per-row packed INT4 weights + dynamic INT4 activations (runtime).",
        "4-bit downward extension of int8_tensorwise: same marker convention, same ConvRot.",
        "Emission contract [-7, 7], scale = absmax/7 (kitchen int4 contract; -8 never emitted).",
        "Pack order LOW nibble first (kitchen/svdquant convention; NVFP4 is high-first — do not mix).",
        "Runtime: comfy-kitchen TensorWiseINT4Layout int4_linear; native INT4 tensor cores on SM 7.5-8.9 only.",
        "Supports per-layer INT8 (int8_tensorwise) fallback for the mixed-precision quality recipe.",
        "No input_scale: activations quantized dynamically at runtime.",
    ),
    metadata={
        "weight_tensor": "weight",
        "weight_packing": "int4_pairs_low_nibble_first",
        "scale_tensor": "weight_scale",
        "marker_tensor": "comfy_quant",
        "weight_scale_shape": "per_row_2d_out_1",
        "convrot_group_size": CONVROT_GROUP_SIZE,
        "symmetric": True,
        "quant_min": -7,
        "quant_max": 7,
        "no_input_scale": True,
        "mixed_int8_fallback": "per-layer int8_tensorwise markers in the same checkpoint",
        "downstream_loader": "comfy-kitchen TensorWiseINT4Layout (stock ComfyUI QUANT_ALGOS entry in flight)",
    },
)


registry.register_format(INT4_TENSORWISE_FORMAT)
