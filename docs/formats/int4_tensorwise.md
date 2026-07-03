# INT4 tensorwise (W4A4 + ConvRot, mixed-capable) checkpoint format

This page defines the **`int4_tensorwise`** checkpoint format produced by Comfy
Quants — the **4-bit downward extension of
[`int8_tensorwise`](int8_tensorwise.md)**. User commands are in
[`../quantization/int4_tensorwise.md`](../quantization/int4_tensorwise.md).

The consumer is comfy-kitchen's `TensorWiseINT4Layout` runtime (kitchen branch
`feat/int4-tensorwise-convrot`; the stock-ComfyUI `QUANT_ALGOS["int4_tensorwise"]`
entry follows the same rollout sequence as int8_tensorwise). Weights are packed
INT4 offline; activations are rotated online (ConvRot) and dynamically quantized
per-token to INT4 **inside the kernel** — true W4A4. Native INT4 tensor cores
exist on **SM 7.5–8.9 only** (20/30/40-series, incl. L4); newer architectures
execute the same math on the INT8 pipeline (correct, no 4-bit speedup — use
nvfp4/int8 there).

## Relation to `int8_tensorwise`

| | `int4_tensorwise` | `int8_tensorwise` |
| --- | --- | --- |
| Weight | int8 container `[out, in/2]` — 2 signed INT4/byte, **LOW nibble first** | int8 `[out, in]` |
| Emission | symmetric `[-7, 7]`, scale = `amax/7` (kitchen int4 contract; -8 never emitted) | `[-128, 127]`, scale = `amax/127` |
| Scale | fp32 `[out, 1]` | fp32 `[out, 1]` |
| Marker | `{"format": "int4_tensorwise"[, "convrot": true, "convrot_groupsize": 256]}` | same shape, format `int8_tensorwise` |
| ConvRot | identical regular-Hadamard, group 256, per-layer `in % 256 == 0` gate | identical |
| Runtime | comfy-kitchen `int4_linear` (W4A4) | stock ComfyUI ≥ 0.27 |

The nibble order matches kitchen/svdquant and `formats/int4_common.py`; the NVFP4
pack is high-nibble-first — **do not mix the two**.

## Numeric convention (bit-faithful to comfy-kitchen eager)

Mirrors comfy-kitchen `quantize_int4_convrot_weight` / `quantize_int4_rowwise`
(the runtime's requantization path, e.g. on LoRA offload — the int8 lesson from
ComfyUI #14642 applies):

```text
if convrot and in_features % group_size == 0:        # group_size = 256
    H = regular_hadamard(gs, dtype=W.dtype)           # built AT the weight dtype
    w = (W.view(out, in//gs, gs) @ H.T).reshape(out, in)
scale      = clamp(w.abs().amax(dim=-1, keepdim=True).float() / 7, min=1e-30)   # fp32 [out, 1]
scale_math = scale.to(w.dtype); scale_math[scale_math == 0] = tiny(w.dtype)
codes      = round(w / scale_math).clamp(-7, 7)                                  # DIVISION, at w.dtype
weight     = pack_low_nibble_first(codes)                                        # int8 [out, in//2]
```

Enforced bit-exactly by `tests/unit/test_external_int4_tensorwise_parity.py`
against the comfy-kitchen source (`COMFY_QUANTS_COMFY_KITCHEN_SOURCE`; skips on
checkouts predating int4_tensorwise).

## Layer side tensors

For each INT4-quantized Linear `<layer>` (`in_features` must be even; ConvRot
additionally needs `in % 256 == 0`, else the layer stores unrotated with no
convrot marker keys):

```text
<layer>.weight        int8 tensor, shape [out_features, in_features//2]  (packed INT4)
<layer>.weight_scale  float32 tensor, shape [out_features, 1]
<layer>.comfy_quant   uint8 JSON marker
```

No `<layer>.input_scale`; bias is copied through unquantized. Marker bytes use
the stock save-path encoding (default `json.dumps`, `format` first):

```json
{"format": "int4_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

## Mixed precision (the W4A4 quality recipe)

Pure rotation-based W4A4 measurably degrades DiTs (ConvRot paper: FLUX.1 FID
12.32 vs BF16 10.07); ~20% of layers are *functionally* 4-bit-sensitive (not
outlier-driven — better rotation cannot fix them) and must stay W8A8 to match
SVDQuant quality (FID 10.03 vs 10.01). The exporter therefore accepts
**`int8_fallback` module globs**: matching layers are written with the exact
[`int8_tensorwise`](int8_tensorwise.md) tensors + marker (the int8 quant helper
is imported from that writer, guaranteeing byte-identity) in the same
checkpoint. ComfyUI resolves `QUANT_ALGOS` per layer from each `comfy_quant`
marker, so the two formats coexist natively. The shipped family configs carry
fallback lists validated/derived from the L4 E2E sweep — **adaLN modulation
Linears must always be in the fallback set** (at int4 they cause global
saturation drift: 9.27 dB vs 21.67 dB on qwen-edit-2511), followed by the
paper's attention value/output projections. See the workflow page for measured
numbers.

## Compute path (context)

`int4_linear`: online group-256 rotation → per-token INT4 dynamic quantization
(same `amax/7` recipe) → 4-bit codes through integer MMA with exact INT32
accumulation → **rank-1 epilogue dequant** (per-token × per-row scale outer
product; no per-group rescaling inside the K loop — the structural advantage
over group-scaled W4A4 kernels) → bias. Eager fallback runs anywhere via
`torch._int_mm`.
