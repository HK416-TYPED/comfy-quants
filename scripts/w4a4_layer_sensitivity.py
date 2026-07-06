"""Per-layer W4A4 sensitivity ranking for Krea-2 (local + leave-one-out).

Loads the bf16 DiT through ComfyUI's model management (lowvram streaming, so a
28 GB model runs on a 24 GB card), replays real captured sampling inputs, and
produces two rankings over all quantizable Linears:

  local  — relative output error of the layer itself under W4A4 simulation
           (eager int4 path, bit-matching the runtime, ConvRot when K%256==0),
           averaged over all captured steps
  global — relative drift of the FINAL model output when ONLY that layer is
           swapped to W4A4 (leave-one-out), on the middle captured step

Usage:
  python krea2_layer_sensitivity.py <dump_dir> <krea2_bf16.safetensors> <out.json>
"""
from __future__ import annotations

import glob
import json
import sys

import torch

import os

sys.path.insert(0, os.environ.get("COMFY_QUANTS_COMFYUI_SOURCE", "../external/ComfyUI"))
sys.path.insert(0, os.environ.get("COMFY_QUANTS_COMFY_KITCHEN_SOURCE", "../external/comfy-kitchen"))

from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    int4_linear,
    quantize_int4_convrot_weight,
    quantize_int4_rowwise,
)

DEVICE = "cuda"


def load_dit(ckpt_path):
    import comfy.model_management
    import comfy.sd
    patcher = comfy.sd.load_diffusion_model(
        ckpt_path, model_options={"weight_dtype": torch.bfloat16}
    )
    # Leave headroom: the eager int4 reference materializes an fp32 dequantized
    # weight per call (~400 MB for the largest layer), so don't let the loader
    # fill the whole card.
    comfy.model_management.load_models_gpu([patcher], memory_required=5 * 1024**3)
    return patcher, patcher.model.diffusion_model.eval()


def quantizable_linears(dit):
    """All block Linears plus the shared modulation projector tproj.1."""
    mods = {}
    for name, mod in dit.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if name.startswith("blocks.") or name == "tproj.1":
            mods[name] = mod
    return mods


class W4A4Sim:
    """W4A4 simulation for one Linear: quantize once (ConvRot when K%256==0),
    cache the int4 buffers on CPU, move to GPU per call."""

    def __init__(self, mod):
        self.mod = mod
        w = mod.weight.data.to(DEVICE, torch.bfloat16, non_blocking=True)
        self.convrot = w.shape[1] % 256 == 0
        if self.convrot:
            wq, ws = quantize_int4_convrot_weight(w, 256)
        else:
            wq, ws = quantize_int4_rowwise(w)
        self.wq, self.ws = wq.cpu(), ws.cpu()
        del w
        self.orig_forward = mod.forward

    def y_quant(self, x):
        bias = self.mod.bias
        if bias is not None:
            bias = bias.data.to(x.device, x.dtype)
        return int4_linear(
            x, self.wq.to(x.device), self.ws.to(x.device), bias,
            x.dtype, self.convrot, 256,
        )

    def install(self):
        self.mod.forward = self.y_quant

    def restore(self):
        self.mod.forward = self.orig_forward


@torch.no_grad()
def main():
    dump_dir, ckpt, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    dumps = sorted(glob.glob(f"{dump_dir}/krea2_inputs_call*.pt"))
    assert dumps, f"no dumps in {dump_dir}"
    inputs = [torch.load(d, weights_only=True) for d in dumps]
    print(f"{len(inputs)} captured steps; loading model...", flush=True)

    _patcher, dit = load_dit(ckpt)
    mods = quantizable_linears(dit)
    print(f"{len(mods)} quantizable linears", flush=True)

    def run(inp):
        return dit(
            inp["x"].to(DEVICE, torch.bfloat16),
            inp["timesteps"].to(DEVICE, torch.bfloat16),
            context=inp["context"].to(DEVICE, torch.bfloat16),
        ).float()

    print("building W4A4 sims...", flush=True)
    sims = {}
    for i, (name, mod) in enumerate(mods.items()):
        sims[name] = W4A4Sim(mod)
        if i % 50 == 0:
            print(f"  quantized {i}/{len(mods)}", flush=True)

    # Pass 1: local per-layer error, one instrumented forward per step.
    local = {name: [] for name in mods}
    hooks = []

    def make_hook(name, sim):
        def hook(_m, args, out):
            ref = out.float()
            yq = sim.y_quant(args[0]).float()
            local[name].append(
                ((yq - ref).norm() / ref.norm().clamp(min=1e-9)).item()
            )
        return hook

    for name, mod in mods.items():
        hooks.append(mod.register_forward_hook(make_hook(name, sims[name])))
    for inp in inputs:
        run(inp)
    for h in hooks:
        h.remove()
    print("local pass done", flush=True)

    # Pass 2: leave-one-out global drift on the middle captured step.
    mid = inputs[len(inputs) // 2]
    ref_mid = run(mid)
    global_err = {}
    for i, name in enumerate(mods):
        sims[name].install()
        y = run(mid)
        sims[name].restore()
        global_err[name] = ((y - ref_mid).norm() / ref_mid.norm().clamp(min=1e-9)).item()
        if i % 20 == 0:
            print(f"  loo {i}/{len(mods)}: {name} {global_err[name]:.5f}", flush=True)

    result = {
        name: {
            "local": sum(local[name]) / max(len(local[name]), 1),
            "global": global_err[name],
            "shape": list(mods[name].weight.shape),
        }
        for name in mods
    }
    json.dump(result, open(out_path, "w"), indent=1)

    ranked = sorted(result.items(), key=lambda kv: -kv[1]["global"])
    print(f"\n{'layer':<42} {'global':>9} {'local':>9}")
    for name, r in ranked[:30]:
        print(f"{name:<42} {r['global']:>9.5f} {r['local']:>9.5f}")
    print("RANKING_DONE")


if __name__ == "__main__":
    main()
