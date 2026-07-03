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

| Model family | Config | int8 fallback (initial recipe) |
| --- | --- | --- |
| LTX-2.3 | `configs/ltx2_int4_tensorwise_mixed.yaml` | all attention `to_v` / `to_out.0` (528 of 1,496 layers) |
| FLUX.2 | `configs/flux2_int4_tensorwise_mixed.yaml` | `*_attn.proj` + `single_blocks.*.linear2` (64 of 160) |

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
