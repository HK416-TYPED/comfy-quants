# INT4 tensorwise (W4A4 + ConvRot, mixed int8 fallback) export

Use this guide to produce a **W4A4 `int4_tensorwise`** checkpoint — the 4-bit
downward extension of [int8_tensorwise](int8_tensorwise.md), calibration-free
(rotate + round; no calibration data, no GPTQ, no low-rank solve). Storage
contract: [`../formats/int4_tensorwise.md`](../formats/int4_tensorwise.md).

Runtime status: comfy-kitchen `TensorWiseINT4Layout` (kitchen branch
`feat/int4-tensorwise-convrot`); the stock-ComfyUI loader entry is in flight —
same rollout sequence as int8_tensorwise. Native INT4 tensor-core speedup on
**SM 7.5–8.9** (20/30/40-series, L4); on Blackwell prefer nvfp4/int8.

## Supported model-family configs (v1)

| Model family | Config | int8 fallback |
| --- | --- | --- |
| Qwen-Image-Edit-2511 | `configs/qwen_image_edit_2511_int4_tensorwise_mixed.yaml` | attn value/output proj + **adaLN modulation** (359 of 839) — **E2E-validated** |
| LTX-2.3 | `configs/ltx2_int4_tensorwise_mixed.yaml` | all attention `to_v` / `to_out.0` + gate projections (792 of 1,496) |
| FLUX.2 | `configs/flux2_int4_tensorwise_mixed.yaml` | `*_attn.proj` + `single_blocks.*.linear2` (64 of 160) |
| Krea 2 | `configs/krea2_int4_tensorwise_mixed.yaml` | `attn.wv`/`attn.wo` + `attn.gate` (modulation-class) (84 of 224) |

## Measured results (L4 / SM 8.9, qwen-edit-2511, 20 steps, PSNR vs bf16-sentinel)

Quality: mean over 4 edit prompts, all checkpoints through the SAME eager
comfy-kitchen build on the same GPU (single-image PSNR carries ±2 dB trajectory
noise — always compare on multi-prompt means):

| Checkpoint | Layers (int4/int8) | Size | 4-prompt mean PSNR |
| --- | --- | --- | --- |
| int8_tensorwise (baseline) | 0 / 839 | 20.0 GB | 30.05 dB |
| int4 mixed + MLP down-proj at int8 (quality variant) | 360 / 479 | 17.2 GB | 27.19 dB |
| int4 mixed (**shipped default**) | 480 / 359 | 14.9 GB | 26.58 dB |
| int4, paper-style list only (**no modulation fallback**) | 599 / 240 | 11.5 GB | ~9 dB ❌ (single image) |

The quality variant's +0.61 dB mean advantage over the default splits 2:2 across
prompts (within noise), while the default is 2.3 GB smaller and keeps 33% more
layers at int4 (future INT4 tensor-core speed coverage) — hence the default.

Speed baselines (same box, 20-step image, steady state after model load):

| Path | bf16 | int8 | int4 mixed |
| --- | --- | --- | --- |
| eager reference (quality runs above) | 117 s | 288 s | 309–330 s |
| **release form** (kitchen wheel, CUDA kernels) | 114 s | **87 s (1.31×)** | pending CUDA kernel |

The eager path exists for correctness/quality work only; the int8 release-form
line is the speed bar the future int4 CUDA kernel must beat on SM 7.5–8.9.

**⚠️ adaLN modulation layers must never be int4.** With qwen's `img_mod.1` /
`txt_mod.1` at int4 the output shows global saturation drift + grain (the
9.27 dB row) even though every attention value/output projection was already at
int8; promoting the modulation Linears alone recovered +12.4 dB. Families whose
modulation lives outside the block include-globs (FLUX.2, LTX-2's adaln_single)
keep it bf16 automatically; LTX-2's in-block gate projections are
modulation-class and sit in the fallback list.

## Quick start: LTX-2.3

```bash
comfy-quants export-model-int4-tensorwise \
  --config configs/ltx2_int4_tensorwise_mixed.yaml \
  --source /path/to/ltx-2.3-22b-dev.safetensors \
  --out /path/to/ltx2_int4_tensorwise_mixed.safetensors \
  --device cuda:0 \
  --hash-output \
  --json
```

Directory outputs use `diffusion_pytorch_model.int4_tensorwise.safetensors`.

## Inputs

| Input | Argument | Description |
| --- | --- | --- |
| Config | `--config` | YAML selecting family, source, `quant.target_dtype: int4_tensorwise`, and the `quant.modules.int8_fallback` glob list. |
| Source | `--source` | Local dense bf16 transformer `.safetensors` / index JSON / indexed directory. |
| Output | `--out` | Output `.safetensors` path or directory. |
| ConvRot | `--convrot` / `--no-convrot` | Regular-Hadamard rotation (default **on**; per-layer `in % 256 == 0` gate). |
| Group size | `--convrot-groupsize` | Power of four; default 256. |
| INT8 fallback | `--int8-fallback GLOB` (repeatable) | Overrides the config's `int8_fallback` list. |

## Mixed precision (80/20 recipe)

The ConvRot paper's quality result requires ~20% of layers at W8A8: pure W4A4
loses low-frequency detail (FLUX.1 FID 12.32 vs BF16 10.07); with the sensitive
layers at W8A8 it matches SVDQuant (10.03 vs 10.01). Layers matching
`quant.modules.int8_fallback` are written as **int8_tensorwise** (identical
bytes to the dedicated int8 writer) in the same checkpoint; the loader
dispatches per layer from each `comfy_quant` marker. The shipped lists map the
paper's FLUX.1 findings (attention value/output projections) onto each
architecture — treat them as starting points and refine with E2E A/B. The
export report records `int4_tensor_count`, `int8_fallback_tensor_count`, and
the exact `int8_fallback_tensors` list.

## Validation

- Offline: the export report (`*_report.json`) plus the bit-exact parity test
  `tests/unit/test_external_int4_tensorwise_parity.py` (kitchen oracle).
- E2E: pending the ComfyUI loader entry; A/B against int8_tensorwise / nvfp4
  baselines on real prompts (PSNR vs bf16).
