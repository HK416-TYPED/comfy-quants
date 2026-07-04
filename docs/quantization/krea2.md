# Krea 2 family export (INT8 tensorwise / INT4 tensorwise)

Quantize the **Krea 2** single-stream DiT (K2; `krea/Krea-2-Turbo`,
`krea/Krea-2-Raw`, `Comfy-Org/Krea-2` — ComfyUI `image_model="krea2"`).
28 blocks × 8 Linears (GQA attention `wq/wk/wv` + gating `gate` + `wo`, SwiGLU
`mlp.gate/up/down`), features 6144 / kv 1536 / mlp 16384 — **224 quantizable
Linears, every `in_features` divisible by 256** (ConvRot applies everywhere).
Bare `blocks.` keys (no wrapper prefix).

## Supported formats & configs

| Format | Command | Config |
| --- | --- | --- |
| INT8 tensorwise (+ConvRot) | `export-model-int8-tensorwise` | `configs/krea2_int8_tensorwise.yaml` |
| [INT4 tensorwise W4A4 mixed](int4_tensorwise.md) (+ConvRot, int8 fallback) | `export-model-int4-tensorwise` | `configs/krea2_int4_tensorwise_mixed.yaml` |

## Layer selection

Both configs quantize `blocks.*` (224 layers). The INT4 mixed recipe keeps 84
layers at int8 — `attn.wv` / `attn.wo` (the ConvRot paper's sensitive value/
output projections) plus `attn.gate` (**modulation-class**: the qwen L4 E2E
showed modulation-path layers must never be int4). Krea 2's per-block `mod.lin`
is a bias parameter (copied verbatim, nothing to quantize) and the shared
modulation projector `tproj` sits outside the include set (bf16). The
text-fusion transformer (`txtfusion.*`), `tmlp`/`txtmlp`, `first` and `last`
stay high precision.

## Quick start

```bash
comfy-quants export-model-int4-tensorwise \
  --config configs/krea2_int4_tensorwise_mixed.yaml \
  --source /path/to/krea-2-turbo.safetensors \
  --out krea2_turbo_int4_tensorwise_mixed.safetensors \
  --device cuda:0 --hash-output --json
```

INT8 loads in stock ComfyUI ≥ 0.27 (SM ≥ 7.5). INT4 needs the comfy-kitchen
`TensorWiseINT4Layout` runtime (in-flight; see the
[format page](../formats/int4_tensorwise.md)). Contract test:
`tests/unit/test_krea2_contract.py`; tensor names validated against the real
checkpoint header at first export (strict missing-tensor check).
