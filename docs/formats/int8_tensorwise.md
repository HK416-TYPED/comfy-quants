# INT8 tensorwise (stock ComfyUI, + ConvRot) checkpoint format

This page defines the **stock-ComfyUI-native INT8** checkpoint format
(`int8_tensorwise`) produced by Comfy Quants. User commands are in
[`../quantization/int8_tensorwise.md`](../quantization/int8_tensorwise.md).

`int8_tensorwise` is the exact key of stock ComfyUI's built-in
`QUANT_ALGOS["int8_tensorwise"]` entry (**ComfyUI >= v0.27.0**, 2026-06-30; kernels in
comfy-kitchen >= 0.2.12, `TensorWiseINT8Layout`, **SM >= 7.5 / Turing+** — not
Blackwell-gated like MXFP8/NVFP4). Weights are int8 offline; activations are
dynamically row-quantized (and, with ConvRot, online-rotated) **inside the
comfy-kitchen kernel** at runtime — W8A8 without any custom node.

## Relation to `int8_w8a8` (retired ComfyUI-INT8-Fast)

The storage tensors are identical to [`int8_w8a8.md`](int8_w8a8.md) (int8 weight +
fp32 `[out, 1]` scale + the same regular-Hadamard ConvRot, group 256). The two
formats differ in:

| | `int8_tensorwise` (stock) | `int8_w8a8` (INT8-Fast, retired) |
| --- | --- | --- |
| Marker | `{"format": "int8_tensorwise"[, "convrot": true, "convrot_groupsize": 256]}` | `{"convrot": …[, "convrot_groupsize"], "per_row": true}` |
| `format` key | **required** (stock raises `ValueError` without it) | absent |
| Quant math oracle | comfy-kitchen `quantize_int8_rowwise` (division, weight dtype) | INT8-Fast (fp32 reciprocal-multiply) |
| Loader | stock ComfyUI >= 0.27 (SM >= 7.5) | ComfyUI-INT8-Fast custom node |

An `int8_w8a8` artifact does **not** load in stock ComfyUI (its marker lacks the
required `format` key → `ValueError`), and an `int8_tensorwise` artifact is not
understood by the retired INT8-Fast node — do not mix the two marker payloads.

## Numeric convention (bit-faithful to comfy-kitchen)

Mirrors comfy-kitchen eager `quantize_int8_convrot_weight` / `quantize_int8_rowwise`
**as of >= 0.2.15** (the layout shipped in 0.2.11; 0.2.12–0.2.14 divide in fp32
instead, differing by at most ±1 int8 code on bf16/fp16 weights — scales identical):
rotation and division happen at the **source weight dtype** (bf16 for released
checkpoints). Once ComfyUI's comfy-kitchen pin is >= 0.2.15, re-quantization inside
ComfyUI (e.g. on LoRA offload) reproduces the artifact bit-exactly:

```text
if convrot and in_features % group_size == 0:        # group_size = 256
    H = regular_hadamard(gs, dtype=W.dtype)           # built AT the weight dtype
    w = (W.view(out, in//gs, gs) @ H.T).reshape(out, in)
scale      = clamp(w.abs().amax(dim=-1, keepdim=True).float() / 127, min=1e-30)  # fp32 [out, 1]
scale_math = scale.to(w.dtype); scale_math[scale_math == 0] = tiny(w.dtype)
q          = round(w / scale_math).clamp(-128, 127).to(int8)                     # DIVISION, at w.dtype
```

Enforced by `tests/unit/test_external_int8_tensorwise_parity.py` against the
comfy-kitchen source (`COMFY_QUANTS_COMFY_KITCHEN_SOURCE`): the Hadamard/rotation
helpers and — when the checkout is importable as a package — the real eager
`quantize_int8_rowwise` are loaded from the checkout (requires >= 0.2.15 for the
division recipe; falls back to a pinned inline oracle otherwise).

## Layer side tensors

For each quantized Linear `<layer>`:

```text
<layer>.weight        int8 tensor, shape [out_features, in_features]
<layer>.weight_scale  float32 tensor, shape [out_features, 1]
<layer>.comfy_quant   uint8 JSON marker
```

No `<layer>.input_scale`; bias is copied through unquantized. Marker bytes match
stock ComfyUI's own save path (`ops.py:_quantized_weight_state_dict`: default
`json.dumps`, `format` first; `convrot`/`convrot_groupsize` only when the layer was
actually rotated):

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

Non-divisible `in_features` → rotation skipped for that layer (comfy-kitchen would
raise; the writer omits the convrot keys instead): `{"format": "int8_tensorwise"}`.
`full_precision_matrix_mult` is deliberately never emitted (mxfp8/nvfp4 precedent).

## Compute path (context)

The stock kernel picks cuda (CUTLASS/cuBLASLt IMMA, SM >= 7.5) > triton (opt-in) >
eager (`torch._int_mm`, any device); unsupported setups fall back to dequantized
matmul, so the checkpoint loads anywhere, just without the int8 speedup.
