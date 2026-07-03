"""LTX-2 model adapter (Lightricks LTX-2.3 audio-video DiT, stock-ComfyUI ``image_model="ltxav"``).

Authored from ComfyUI ``comfy/ldm/lightricks/av_model.py`` and verified against the real
``Lightricks/LTX-2.3`` checkpoint header. LTX-2.3 (22B) is a dual-stream audio+video DiT:
``transformer_blocks.N`` each carry a video stream (``attn1`` self / ``attn2`` text-cross,
GELU ``ff``), a parallel audio stream (``audio_attn1`` / ``audio_attn2`` / ``audio_ff``),
and bidirectional audio<->video cross-attention (``audio_to_video_attn`` /
``video_to_audio_attn``). Every attention additionally has a gated-attention head
projection ``to_gate_logits`` (new in 2.3) — 34 Linears/block. Video inner_dim=4096
(32 heads x 128), audio inner_dim=2048 (32 heads x 64), 48 blocks, ff mult=4. Bare
architecture differs from the older ``ltxv`` family (LTXV 0.9.x 2B, video-only); an
LTX-2.0 checkpoint also does NOT match this contract (no gate logits, 2-layer external
embeddings connectors).

Default policy mirrors the official Lightricks FP8/NVFP4 releases: quantize blocks
2..45 and keep the first two and last two blocks in bf16 (1,496 quantized Linears,
layer-for-layer identical to ``Lightricks/LTX-2.3-fp8``/``-nvfp4``). Kept high precision:
the 8-layer video/audio embeddings connectors, all adaln stacks (incl. the 2.3
``prompt_adaln_single`` / ``av_ca_*`` heads), patchify/output projections, and the
``scale_shift_table`` parameters. The released single-file bundle stores diffusion
tensors under ``model.diffusion_model.`` and also carries ``vae.``, ``audio_vae.``,
``vocoder.`` and ``text_embedding_projection.`` sections, which the exporter copies
verbatim. Stock-ComfyUI-native target (fp8_e4m3/fp8_e5m2/mxfp8/nvfp4).
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

CONTRACT_SCHEMA_VERSION = "ltx2_static_contract.v1"

# LTX-2.3 transformer config (checkpoint __metadata__.config.transformer:
# AVTransformer3DModel, num_layers=48, 32 heads x head_dim 128 -> video inner 4096,
# audio 32 x 64 -> inner 2048, cross_attention_dim=4096, audio_cross_attention_dim=2048).
_INNER = 4096
_AUDIO = 2048
_BLOCKS = 48
_GATE_HEADS = 32
# Like ltxv (and unlike flux/ideogram), the released single-file bundles
# (Lightricks/LTX-2.3 ltx-2.3-22b-{dev,distilled,distilled-1.1}.safetensors) prefix
# every diffusion tensor with ``model.diffusion_model.`` (verified from the real header).
_PFX = "model.diffusion_model."

# The official Lightricks FP8/NVFP4 releases keep the first two and last two blocks
# in bf16; the default policy reproduces that recipe exactly.
OFFICIAL_KEPT_BLOCKS = (0, 1, 46, 47)


def official_kept_block_globs() -> list[str]:
    return [f"{_PFX}transformer_blocks.{block}.*" for block in OFFICIAL_KEPT_BLOCKS]


def _dims() -> dict[str, int]:
    return {
        "D": _INNER,  # 4096 video stream width
        "A": _AUDIO,  # 2048 audio stream width
        "FF": 4 * _INNER,  # 16384
        "AFF": 4 * _AUDIO,  # 8192
        "G": _GATE_HEADS,  # 32 gated-attention logits rows
    }


def _attention(prefix: str, q_in: str, kv_in: str, attn_dim: str, out_dim: str, gate_in: str) -> tuple:
    """One LTX-2.3 attention: q/k/v/out Linears + the gated-attention logits Linear."""
    return (
        linear(f"{prefix}.to_q", attn_dim, q_in),
        linear(f"{prefix}.to_k", attn_dim, kv_in),
        linear(f"{prefix}.to_v", attn_dim, kv_in),
        linear(f"{prefix}.to_out.0", out_dim, attn_dim),
        linear(f"{prefix}.to_gate_logits", "G", gate_in, module_type="GateLogitsLinear"),
    )


def _block_modules() -> tuple:
    p = _PFX + "transformer_blocks.{block}"
    return (
        # video self-attention / text cross-attention (cross_attention_dim == D)
        *_attention(f"{p}.attn1", "D", "D", "D", "D", "D"),
        *_attention(f"{p}.attn2", "D", "D", "D", "D", "D"),
        # audio self-attention / text cross-attention (audio_cross_attention_dim == A)
        *_attention(f"{p}.audio_attn1", "A", "A", "A", "A", "A"),
        *_attention(f"{p}.audio_attn2", "A", "A", "A", "A", "A"),
        # audio->video cross-attention: Q from video (D), KV from audio, back to video
        *_attention(f"{p}.audio_to_video_attn", "D", "A", "A", "D", "D"),
        # video->audio cross-attention: Q from audio, KV from video (D), back to audio
        *_attention(f"{p}.video_to_audio_attn", "A", "D", "A", "A", "A"),
        # feed-forward (GELU_approx project_in -> Linear out), video + audio streams
        linear(f"{p}.ff.net.0.proj", "FF", "D", module_type="GELULinear"),
        linear(f"{p}.ff.net.2", "D", "FF"),
        linear(f"{p}.audio_ff.net.0.proj", "AFF", "A", module_type="GELULinear"),
        linear(f"{p}.audio_ff.net.2", "A", "AFF"),
    )


def _extra_components() -> tuple:
    return (
        kept_component(_PFX + "patchify_proj", "Linear", "transformer", "video latent patch projection kept high precision"),
        kept_component(_PFX + "audio_patchify_proj", "Linear", "transformer", "audio latent patch projection kept high precision"),
        kept_component(_PFX + "adaln_single", "AdaLayerNormSingle", "transformer", "video timestep adaln kept high precision"),
        kept_component(_PFX + "audio_adaln_single", "AdaLayerNormSingle", "transformer", "audio timestep adaln kept high precision"),
        kept_component(_PFX + "prompt_adaln_single", "AdaLayerNormSingle", "transformer", "video cross-attention adaln kept high precision"),
        kept_component(_PFX + "audio_prompt_adaln_single", "AdaLayerNormSingle", "transformer", "audio cross-attention adaln kept high precision"),
        kept_component(_PFX + "av_ca_video_scale_shift_adaln_single", "AdaLayerNormSingle", "transformer", "a/v cross-attention video scale-shift adaln kept high precision"),
        kept_component(_PFX + "av_ca_a2v_gate_adaln_single", "AdaLayerNormSingle", "transformer", "audio->video gate adaln kept high precision"),
        kept_component(_PFX + "av_ca_audio_scale_shift_adaln_single", "AdaLayerNormSingle", "transformer", "a/v cross-attention audio scale-shift adaln kept high precision"),
        kept_component(_PFX + "av_ca_v2a_gate_adaln_single", "AdaLayerNormSingle", "transformer", "video->audio gate adaln kept high precision"),
        kept_component(_PFX + "video_embeddings_connector", "Embeddings1DConnector", "transformer", "8-layer video text-embeddings connector kept high precision"),
        kept_component(_PFX + "audio_embeddings_connector", "Embeddings1DConnector", "transformer", "8-layer audio text-embeddings connector kept high precision"),
        kept_component(_PFX + "proj_out", "Linear", "transformer", "video output projection kept high precision"),
        kept_component(_PFX + "audio_proj_out", "Linear", "transformer", "audio output projection kept high precision"),
        kept_component(_PFX + "scale_shift_table", "Parameter", "transformer", "output-head modulation table kept high precision"),
        kept_component(_PFX + "audio_scale_shift_table", "Parameter", "transformer", "audio output-head modulation table kept high precision"),
    )


def build_ltx2_static_contract() -> StockDitContract:
    return StockDitContract(
        family="ltx2",
        schema_version=CONTRACT_SCHEMA_VERSION,
        preferred_format="fp8_e4m3",
        dims=_dims(),
        block_groups=(BlockGroup(prefix=_PFX + "transformer_blocks", count=_BLOCKS, modules=_block_modules()),),
        extra_components=_extra_components(),
        metadata={
            "export_name": "LTX-2",
            "architecture": "ltxav_dit",
            "model_version": "2.3",
            "inner_dim": _INNER,
            "num_heads": 32,
            "head_dim": 128,
            "audio_inner_dim": _AUDIO,
            "audio_num_heads": 32,
            "audio_head_dim": 64,
            "cross_attention_dim": _INNER,
            "audio_cross_attention_dim": _AUDIO,
            "ff_mult": 4,
            "num_blocks": _BLOCKS,
            "caption_channels": 3840,
            "official_kept_blocks": OFFICIAL_KEPT_BLOCKS,
            "supported_model_ids": ("Lightricks/LTX-2.3",),
        },
    )


class Ltx2Adapter:
    """Adapter for LTX-2.3 (22B) audio-video DiT."""

    family = "ltx2"
    supported_model_ids = ["Lightricks/LTX-2.3"]

    def inspect(self, source: ModelSource):
        contract = build_ltx2_static_contract()
        graph = build_stock_dit_graph(
            contract,
            source,
            artifact_metadata=stock_dit_artifact_contract_metadata("ltx2"),
        )
        return summarize_stock_dit_graph(graph, self.__class__.__name__), graph

    def default_policy(self, target_dtype: str = "fp8_e4m3") -> QuantPolicy:
        return QuantPolicy(
            name="ltx2_default",
            algorithm="fp8_static",
            target_dtype=target_dtype,
            include=[_PFX + "transformer_blocks.*"],
            exclude=official_kept_block_globs(),
            keep_components=["text_encoder", "vae"],
        )


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_adapter(Ltx2Adapter())
