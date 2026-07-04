import fnmatch
import unittest
from pathlib import Path

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.core.config import load_quant_config
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.krea2 import build_krea2_static_contract
from comfy_quants.model_adapters.registry import get_adapter

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _tensors_by_name(graph):
    return {t.name: t for m in graph.modules for t in m.tensors}


def _index(target_dtype, include, exclude):
    adapter = get_adapter("krea2")
    _insp, graph = adapter.inspect(ModelSource(family="krea2", model_id="x"))
    policy = adapter.default_policy(target_dtype)
    policy.include = include
    policy.exclude = exclude
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=target_dtype,
            algorithm_version="0.1.0",
            target_dtype=target_dtype,
            scale_granularity="per_channel",
            scale_axis="out_features",
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
        ),
    )


class TestKrea2Contract(unittest.TestCase):
    def test_contract(self):
        c = build_krea2_static_contract()
        self.assertEqual(c.family, "krea2")
        self.assertEqual(c.schema_version, "krea2_static_contract.v1")
        self.assertEqual([(g.prefix, g.count) for g in c.block_groups], [("blocks", 28)])
        self.assertEqual(c.dimensions()["F"], 6144)
        self.assertEqual(c.dimensions()["KV"], 1536)
        self.assertEqual(c.dimensions()["MLP"], 16384)
        # 5 attention + 3 SwiGLU linears per block
        self.assertEqual(len(c.block_groups[0].modules), 8)

    def test_adapter_registered(self):
        self.assertEqual(get_adapter("krea2").family, "krea2")

    def test_graph_tensor_names_and_shapes(self):
        # Names/shapes match the real Krea-2-Turbo checkpoint header (bare keys).
        adapter = get_adapter("krea2")
        _insp, graph = adapter.inspect(ModelSource(family="krea2", model_id="krea/Krea-2-Turbo"))
        by_name = _tensors_by_name(graph)
        self.assertEqual(by_name["blocks.0.attn.wq.weight"].shape, [6144, 6144])
        self.assertEqual(by_name["blocks.0.attn.wk.weight"].shape, [1536, 6144])
        self.assertEqual(by_name["blocks.0.attn.wv.weight"].shape, [1536, 6144])
        self.assertEqual(by_name["blocks.0.attn.gate.weight"].shape, [6144, 6144])
        self.assertEqual(by_name["blocks.0.attn.wo.weight"].shape, [6144, 6144])
        self.assertEqual(by_name["blocks.0.mlp.gate.weight"].shape, [16384, 6144])
        self.assertEqual(by_name["blocks.0.mlp.up.weight"].shape, [16384, 6144])
        self.assertEqual(by_name["blocks.0.mlp.down.weight"].shape, [6144, 16384])
        self.assertEqual(by_name["blocks.27.mlp.down.weight"].shape, [6144, 16384])
        for name, t in by_name.items():
            if name.startswith("blocks.") and t.role == "weight":
                self.assertEqual(t.shape[1] % 256, 0, name)

    def test_int4_mixed_config(self):
        cfg = load_quant_config(_CONFIG_DIR / "krea2_int4_tensorwise_mixed.yaml")
        self.assertEqual(cfg.model.family, "krea2")
        self.assertEqual(cfg.quant.target_dtype, "int4_tensorwise")
        idx = _index("int4_tensorwise", cfg.quant.modules["include"], cfg.quant.modules["exclude"])
        mods = [r["metadata"]["module_name"] for r in idx["tensors"]]
        self.assertEqual(len(mods), 224)  # 28 blocks x 8 linears
        fallback = cfg.quant.modules["int8_fallback"]
        hits = [m for m in mods if any(fnmatch.fnmatchcase(m, g) for g in fallback)]
        # wv + wo + gate per block: the value/output projections (ConvRot paper)
        # plus the modulation-class gating projection.
        self.assertEqual(len(hits), 84)
        self.assertIn("blocks.0.attn.gate", hits)
        self.assertIn("blocks.27.attn.wv", hits)
        self.assertNotIn("blocks.0.mlp.down", hits)
        # tproj (shared modulation projector) must not be selected at int4.
        self.assertFalse(any(m.startswith("tproj") for m in mods))

    def test_int8_config(self):
        cfg = load_quant_config(_CONFIG_DIR / "krea2_int8_tensorwise.yaml")
        self.assertEqual(cfg.quant.target_dtype, "int8_tensorwise")
        idx = _index("int8_tensorwise", cfg.quant.modules["include"], cfg.quant.modules["exclude"])
        self.assertEqual(len({r["name"] for r in idx["tensors"]}), 224)


if __name__ == "__main__":
    unittest.main()
