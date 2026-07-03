# LTX-2 (LTX-2.3) family export (FP8 / MXFP8 / NVFP4 / INT8 W8A8 / INT8 tensorwise)

Quantize the **LTX-2.3** diffusion model (Lightricks 22B audio-video DiT, ComfyUI
`image_model="ltxav"`) to the stock-ComfyUI-native formats. This is a separate family
from [`ltxv`](ltxv.md) (the LTXV 0.9.x 2B video-only DiT) — the architectures do not
share a contract, and an LTX-2.0 checkpoint does not match this contract either
(no gated attention, external 2-layer connectors).

## Supported formats & configs

| Format | Command | Config | Loads in |
| --- | --- | --- | --- |
| FP8 E4M3 | `export-model` | `configs/ltx2_fp8.yaml` | stock ComfyUI (any GPU) |
| FP8 E5M2 | `export-model` | `configs/ltx2_fp8_e5m2.yaml` | stock ComfyUI (any GPU) |
| MXFP8 | `export-model-mxfp8` | `configs/ltx2_mxfp8.yaml` | stock ComfyUI (Blackwell SM≥10) |
| NVFP4 | `export-model-nvfp4` | `configs/ltx2_nvfp4.yaml` | stock ComfyUI (Blackwell SM≥10) |
| INT8 W8A8 (+ConvRot) | `export-model-w8a8` | `configs/ltx2_int8_w8a8.yaml` | ComfyUI-INT8-Fast custom node (retired) |
| INT8 tensorwise (+ConvRot) | `export-model-int8-tensorwise` | `configs/ltx2_int8_tensorwise.yaml` | stock ComfyUI (>= 0.27, SM ≥ 7.5) |

## Architecture

LTX-2.3 (22B, ComfyUI `comfy/ldm/lightricks/av_model.py`): dual-stream audio+video DiT,
**48** `transformer_blocks.N`, each carrying six attentions and two feed-forwards —
video self/cross (`attn1`/`attn2`, inner_dim=4096, 32 heads × 128), audio self/cross
(`audio_attn1`/`audio_attn2`, inner_dim=2048, 32 heads × 64), bidirectional
audio↔video cross-attention (`audio_to_video_attn` / `video_to_audio_attn`), and GELU
FFs (`ff` 4096→16384, `audio_ff` 2048→8192). Every attention adds a gated-attention
projection `to_gate_logits` (new in 2.3) → **34 Linears/block**. Text encoder is
Gemma-3-12B (caption_channels=3840), projected outside the DiT
(`text_embedding_projection.*`).

> **Key prefix**: the released single-file bundles
> (`Lightricks/LTX-2.3/ltx-2.3-22b-{dev,distilled,distilled-1.1}.safetensors`, 46.1 GB
> bf16) store diffusion tensors under **`model.diffusion_model.`** (like `ltxv`) and also
> bundle `vae.`, `audio_vae.`, `vocoder.` and `text_embedding_projection.` sections. The
> exporter copies those verbatim, so the output is a full checkpoint loadable via
> `CheckpointLoaderSimple` (+ `LTXAVTextEncoderLoader` / `LTXVAudioVAELoader` reading the
> same file).

## Official-recipe selection

The default policy reproduces the official `Lightricks/LTX-2.3-fp8` / `-nvfp4` releases
**layer-for-layer**: quantize blocks 2–45 and keep blocks 0, 1, 46, 47 in bf16 →
**1,496** quantized Linears (44 × 34, gate logits included). The four block excludes are
carried in every `configs/ltx2_*.yaml` under `quant.modules.exclude` (including both
INT8 configs, so all six formats quantize the same 1,496 layers) — keep them when
deriving new configs, since config include/exclude **overrides** the adapter policy.
Every ltx2 `in_features` (4096/2048/16384/8192) is divisible by 256, so ConvRot applies
to all selected layers in the INT8 export.
Kept high precision: the two 8-layer `*_embeddings_connector`s, all `*adaln_single`
stacks, `patchify_proj`/`audio_patchify_proj`, `proj_out`/`audio_proj_out`, and the
`scale_shift_table` parameters (F32 in the source).

Contract tensor names/shapes were authored from ComfyUI `av_model.py` and verified
against the real `ltx-2.3-22b-dev.safetensors` header (5,947 tensors; 1,496/1,496
selection match with the official fp8/nvfp4 releases).

> **Quantization source**: use a dense bf16 bundle from `Lightricks/LTX-2.3`. The
> official `-fp8` / `-nvfp4` repos are already quantized and are **not** valid sources
> (same constraint as the Ideogram 4.0 transcode note in
> [`README.md`](README.md)).

## Quick start (NVFP4)

```bash
comfy-quants export-model-nvfp4 \
  --config configs/ltx2_nvfp4.yaml \
  --source /path/to/ltx-2.3-22b-dev.safetensors \
  --out /path/to/ltx2_nvfp4.safetensors \
  --device cuda:0 --hash-output --json
```

`comfy-quants inspect --family ltx2 --model … --out … --json` validates names/shapes.

## Measured (RTX PRO 6000 Blackwell, exported from `ltx-2.3-22b-dev.safetensors` 46.1 GB)

End-to-end in **stock ComfyUI** (v0.26+, **torch 2.10.0+cu130** — all five formats
native incl. mxfp8; `CheckpointLoaderSimple` on our bundles + `LTXAVTextEncoderLoader`
Gemma-3-12B fp4 + `LTXVAudioVAELoader` from the same file; 512²×49 frames + audio,
20 steps euler cfg 3.0, seed 42; peak VRAM per fresh server, including text
encoder/VAEs/activations):

| Format | Artifact size | Peak VRAM | Sampling speed | Video PSNR vs bf16 | Weight SQNR |
| --- | --- | --- | --- | --- | --- |
| bf16 (baseline) | 46.1 GB | 53.7 GB | 2.37 it/s | — | — |
| FP8 E4M3 | 29.1 GB | 36.7 GB (−32%) | 2.27 it/s (0.96×) | 38.5 dB | 31.5 dB |
| MXFP8 | 29.7 GB | 38.4 GB | 2.28 it/s (0.96×) | 38.3 dB | 31.5 dB |
| NVFP4 | **21.7 GB** | **29.6 GB (−45%)** | 1.56 it/s (0.66×) | 32.1 dB | 20.6 dB |
| INT8 W8A8 (+ConvRot) | 29.2 GB | n/a (custom node, retired) | — | — | 41.3 dB (rotated) |
| **INT8 tensorwise** | 29.2 GB | 38.6 GB (−28%) | **2.73 it/s (1.15×)** | **45.4 dB** | 41.3 dB (rotated) |

All five stock formats run natively (`Native ops: int8_tensorwise, float8_e4m3fn,
float8_e5m2, nvfp4, mxfp8`) and produce 49 frames + FLAC audio. All exports report
1,496 quantized layers, string-identical to the official fp8/nvfp4 releases; the INT8
exports rotate all 1,496 (ConvRot applies to every ltx2 in_features).
**int8_tensorwise is the best pick for LTX-2.3**: best fidelity, the only
faster-than-bf16 format (CUTLASS int8 GEMM, SM >= 7.5), and −28% VRAM.

> **torch version note**: mxfp8 tensor-core compute requires **torch >= 2.10** (the
> 2.9 blocker is fp8 DLPack export, not the GEMM — `BufferError: float8 types are not
> supported by dlpack` inside comfy-kitchen's `quantize_mxfp8`). On torch 2.9 mxfp8
> silently falls back to dequant emulation (measured 1.62 it/s, 0.68×) — and emulation
> *changes the numbers*: emulated PSNR was 41.2 dB (bf16-matmul precision) vs 38.3 dB
> native (true fp8 GEMM). Install torch from the **cu130 index**
> (`--index-url https://download.pytorch.org/whl/cu130`) — the default PyPI cu128 wheel
> bundles a cuBLAS too old for block-scaled GEMM and mxfp8 will not accelerate.
> torch 2.10 also lifted int8_tensorwise from 2.50 → 2.73 it/s on this model.

Note the official nvfp4 was QAD-trained, so its *quality* (not encoding) may exceed a
plain post-training NVFP4 export; NVFP4's flux2 speed win does not materialize at this
size/resolution (small-matmul overhead) but it remains the smallest artifact.
