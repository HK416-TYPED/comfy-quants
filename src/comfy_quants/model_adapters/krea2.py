"""Krea 2 model adapter (K2 single-stream DiT).

Authored from ComfyUI ``comfy/ldm/krea2/model.py`` (``image_model="krea2"``,
detection key ``txtfusion.projector.weight``) and validated against the real
checkpoint headers (``krea/Krea-2-Turbo`` lineage; bare ``blocks.`` keys).
Krea 2 is a single-stream DiT: ``blocks.N`` each carry a GQA self-attention
(``attn.wq/wk/wv`` with 48 query / 12 kv heads of 128, an ``attn.gate`` output
gating projection, ``attn.wo``) and a SwiGLU MLP (``mlp.gate/up/down``).
Per-block modulation (``mod.lin``) is a learned bias *parameter*, not a Linear —
it is copied verbatim like every undeclared tensor.

Config: features=6144, kv_dim=1536, mlp=16384, 28 blocks. Every quantizable
``in_features`` (6144/16384) is divisible by 256, so ConvRot applies to all 224
block Linears. The text-fusion transformer (``txtfusion.*``), timestep/text MLPs
(``tmlp``/``txtmlp``), the shared modulation projector (``tproj`` — modulation
producing, never 4-bit), ``first`` and ``last`` stay high precision by default.
"""

from __future__ import annotations

from comfy_quants.comfy.stock_dit_contract import stock_dit_artifact_contract_metadata
from comfy_quants.core.policy import QuantPolicy
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.stock_dit_contract import (
    BlockGroup,
    StockDitContract,
    build_stock_dit_graph,
    kept_component,
    linear,
    summarize_stock_dit_graph,
)

CONTRACT_SCHEMA_VERSION = "krea2_static_contract.v1"

# Krea 2 (Turbo/Raw) config (comfy/ldm/krea2/model.py + real checkpoint header).
_FEATURES = 6144   # 48 heads x 128
_KV = 1536         # 12 kv heads x 128 (GQA)
_MLP = 16384
_BLOCKS = 28


def _dims() -> dict[str, int]:
    return {
        "F": _FEATURES,
        "KV": _KV,
        "MLP": _MLP,
    }


def _block_modules() -> tuple:
    p = "blocks.{block}"
    return (
        # GQA self-attention
        linear(f"{p}.attn.wq", "F", "F"),
        linear(f"{p}.attn.wk", "KV", "F"),
        linear(f"{p}.attn.wv", "KV", "F"),
        # output gating projection — modulation-class (int8 fallback in W4A4 recipes)
        linear(f"{p}.attn.gate", "F", "F", module_type="GateLinear"),
        linear(f"{p}.attn.wo", "F", "F"),
        # SwiGLU MLP: down(silu(gate) * up)
        linear(f"{p}.mlp.gate", "MLP", "F", module_type="SwiGLULinear"),
        linear(f"{p}.mlp.up", "MLP", "F", module_type="SwiGLULinear"),
        linear(f"{p}.mlp.down", "F", "MLP"),
    )


def _extra_components() -> tuple:
    return (
        kept_component("first", "Linear", "transformer", "patchified latent input projection kept high precision"),
        kept_component("tmlp", "Sequential", "transformer", "timestep MLP kept high precision"),
        kept_component("txtmlp", "Sequential", "transformer", "pooled-text MLP kept high precision"),
        kept_component("tproj", "Sequential", "transformer", "shared modulation projector kept high precision (modulation must never be 4-bit)"),
        kept_component("txtfusion", "TextFusionTransformer", "transformer", "text-fusion transformer kept high precision"),
        kept_component("last", "LastLayer", "transformer", "final layer kept high precision"),
    )


def build_krea2_static_contract() -> StockDitContract:
    return StockDitContract(
        family="krea2",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="fp8_e4m3",
        dims=_dims(),
        block_groups=(BlockGroup(prefix="blocks", count=_BLOCKS, modules=_block_modules()),),
        extra_components=_extra_components(),
        metadata={
            "export_name": "Krea-2",
            "architecture": "krea2_single_stream_dit",
            "features": _FEATURES,
            "num_heads": 48,
            "num_kv_heads": 12,
            "head_dim": 128,
            "mlp_dim": _MLP,
            "num_layers": _BLOCKS,
            "text_encoder": "krea2_te (hunyuan-detected)",
            "supported_model_ids": ("krea/Krea-2-Turbo", "krea/Krea-2-Raw", "Comfy-Org/Krea-2"),
        },
    )


class Krea2Adapter:
    """Adapter for the Krea 2 (K2) single-stream DiT."""

    family = "krea2"
    supported_model_ids = ["krea/Krea-2-Turbo", "krea/Krea-2-Raw", "Comfy-Org/Krea-2"]

    def inspect(self, source: ModelSource):
        contract = build_krea2_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("krea2"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        return QuantPolicy(
            name="krea2_default",
            algorithm="fp8_static",
            target_dtype=target_dtype,
            include=["blocks.*"],
            exclude=[],
            keep_components=["text_encoder", "vae"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(Krea2Adapter())
