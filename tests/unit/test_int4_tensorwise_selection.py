import fnmatch
import unittest
from pathlib import Path

from comfy_quants.algorithms.registry import get_algorithm
from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.core.config import load_quant_config
from comfy_quants.core.dtypes import KNOWN_DTYPES
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter
from comfy_quants.registry.global_registry import registry

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _selected_modules(family: str, include, exclude):
    adapter = get_adapter(family)
    _insp, graph = adapter.inspect(ModelSource(family=family, model_id="x"))
    policy = adapter.default_policy("int4_tensorwise")
    policy.include = include
    policy.exclude = exclude
    idx = build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm="int4_tensorwise",
            algorithm_version="0.1.0",
            target_dtype="int4_tensorwise",
            scale_granularity="per_channel",
            scale_axis="out_features",
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
        ),
    )
    return idx, [row["metadata"]["module_name"] for row in idx["tensors"]]


def _fallback_hits(modules, globs):
    return [m for m in modules if any(fnmatch.fnmatchcase(m, g) for g in globs)]


class TestInt4TensorwiseSelection(unittest.TestCase):
    def test_registered(self):
        self.assertIn("int4_tensorwise", registry.list_formats())
        self.assertIn("int4_tensorwise", registry.list_algorithms())
        self.assertIn("int4_tensorwise", KNOWN_DTYPES)
        self.assertEqual(get_algorithm("int4_tensorwise").name, "int4_tensorwise")
        self.assertEqual(KNOWN_DTYPES["int4_tensorwise"].bits, 4)

    def test_ltx2_mixed_config(self):
        cfg = load_quant_config(_CONFIG_DIR / "ltx2_int4_tensorwise_mixed.yaml")
        self.assertEqual(cfg.model.family, "ltx2")
        self.assertEqual(cfg.quant.target_dtype, "int4_tensorwise")
        # Same 1,496-layer selection as every other ltx2 format (official kept blocks).
        idx, modules = _selected_modules("ltx2", cfg.quant.modules["include"], cfg.quant.modules["exclude"])
        self.assertEqual(len(modules), 1496)
        self.assertEqual(idx["format"]["name"], "int4_tensorwise")
        self.assertEqual(idx["format"]["storage_dtype"], "int8")
        # Paper-shaped fallback: every attention to_v / to_out.0 goes int8 (528 of 1496).
        fallback = cfg.quant.modules["int8_fallback"]
        hits = _fallback_hits(modules, fallback)
        self.assertEqual(len(hits), 528)
        self.assertIn("model.diffusion_model.transformer_blocks.2.attn1.to_v", hits)
        self.assertIn("model.diffusion_model.transformer_blocks.45.audio_attn2.to_out.0", hits)
        # Fallback layers must be a strict subset of the selected set.
        self.assertTrue(set(hits) <= set(modules))
        self.assertLess(len(hits), len(modules))
        # Every selected ltx2 layer is ConvRot-eligible and pack-eligible.
        for row in idx["tensors"]:
            self.assertEqual(row["shape"][1] % 256, 0, row["name"])

    def test_flux2_mixed_config(self):
        cfg = load_quant_config(_CONFIG_DIR / "flux2_int4_tensorwise_mixed.yaml")
        self.assertEqual(cfg.model.family, "flux2")
        idx, modules = _selected_modules("flux2", cfg.quant.modules["include"], cfg.quant.modules["exclude"])
        self.assertEqual(len(modules), 160)
        hits = _fallback_hits(modules, cfg.quant.modules["int8_fallback"])
        self.assertEqual(len(hits), 64)
        self.assertTrue(set(hits) <= set(modules))
        for row in idx["tensors"]:
            self.assertEqual(row["shape"][1] % 256, 0, row["name"])

    def test_selection_matches_int8_tensorwise_membership(self):
        # int4 and int8 tensorwise share the family selection: same include/exclude
        # must produce the same module set for both target dtypes.
        cfg = load_quant_config(_CONFIG_DIR / "ltx2_int4_tensorwise_mixed.yaml")
        _idx4, modules4 = _selected_modules("ltx2", cfg.quant.modules["include"], cfg.quant.modules["exclude"])
        adapter = get_adapter("ltx2")
        _insp, graph = adapter.inspect(ModelSource(family="ltx2", model_id="x"))
        policy = adapter.default_policy("int8_tensorwise")
        policy.include = cfg.quant.modules["include"]
        policy.exclude = cfg.quant.modules["exclude"]
        idx8 = build_quant_tensor_index(
            graph,
            policy,
            TensorIndexOptions(
                algorithm="int8_tensorwise",
                algorithm_version="0.1.0",
                target_dtype="int8_tensorwise",
                scale_granularity="per_channel",
                scale_axis="out_features",
                scale_method="amax",
                rounding="nearest_even",
                compatibility_level="L2",
            ),
        )
        modules8 = [row["metadata"]["module_name"] for row in idx8["tensors"]]
        self.assertEqual(set(modules4), set(modules8))


if __name__ == "__main__":
    unittest.main()
