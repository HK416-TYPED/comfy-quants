# INT8 tensorwise export (stock ComfyUI, + ConvRot)

Export a checkpoint whose Linear weights are stored as **stock-ComfyUI-native INT8**
(`int8_tensorwise`, ComfyUI >= v0.27.0, SM >= 7.5). Storage contract:
[`../formats/int8_tensorwise.md`](../formats/int8_tensorwise.md). For the retired
ComfyUI-INT8-Fast custom-node flavor, see [`int8_w8a8.md`](int8_w8a8.md).

## Supported model-family configs (v1)

| Model family | Config |
| --- | --- |
| Qwen-Image | `configs/qwen_image_2512_int8_tensorwise.yaml` |
| Qwen-Image-Edit-2511 | `configs/qwen_image_edit_2511_int8_tensorwise.yaml` |
| Qwen-Image-Layered | `configs/qwen_image_layered_int8_tensorwise.yaml` |
| LTX-2 (2.3) | `configs/ltx2_int8_tensorwise.yaml` (see [`ltx2.md`](ltx2.md)) |

Selection follows each family's official recipe (same include/exclude as the
family's FP8/MXFP8/NVFP4 configs — e.g. ltx2 keeps the first/last two blocks bf16,
1,496 quantized layers), NOT the INT8-Fast exclude list used by `int8_w8a8`.

## Quick start: LTX-2.3

```bash
comfy-quants export-model-int8-tensorwise \
  --config configs/ltx2_int8_tensorwise.yaml \
  --source /path/to/ltx-2.3-22b-dev.safetensors \
  --out /path/to/ltx2_int8_tensorwise.safetensors \
  --device cuda:0 --hash-output --json
```

Flags: `--convrot` / `--no-convrot` (default on; regular Hadamard, group 256),
`--convrot-groupsize` (power of four). The output loads directly in stock ComfyUI
(`CheckpointLoaderSimple` for bundle sources) — no custom node required.

## Notes

- ConvRot applies per layer only when `in_features % 256 == 0`; other layers are
  quantized unrotated and their marker omits the convrot keys.
- Quantization math is bit-faithful to comfy-kitchen >= 0.2.15 (division at the
  source weight dtype). LoRA-offload re-quantization inside ComfyUI reproduces the
  artifact bit-exactly once ComfyUI's comfy-kitchen pin is >= 0.2.15; against the
  0.2.12 pin shipped with ComfyUI v0.27.0 it differs by at most ±1 int8 code on
  bf16/fp16 weights (scales identical).
- No `input_scale` is stored; activations are quantized dynamically at runtime.
