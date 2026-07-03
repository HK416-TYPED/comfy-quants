import unittest
from pathlib import Path

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.core.config import load_quant_config
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Family-official selection: each int8_tensorwise config must select exactly the
# same membership as the family's mxfp8 (official-recipe) config.
_FAMILIES = [
    ("qwen_image", "qwen_image_2512"),
    ("qwen_image_edit", "qwen_image_edit_2511"),
    ("qwen_image_layered", "qwen_image_layered"),
    ("ltx2", "ltx2"),
]


def _selection_for_config(family, config_name, target_dtype, granularity, axis, block_size=None):
    cfg = load_quant_config(_CONFIG_DIR / config_name)
    adapter = get_adapter(family)
    _insp, graph = adapter.inspect(ModelSource(family=family, model_id="x"))
    policy = adapter.default_policy(target_dtype)
    policy.algorithm = cfg.quant.algorithm
    policy.include = cfg.quant.modules.get("include", policy.include)
    policy.exclude = cfg.quant.modules.get("exclude", policy.exclude)
    index = build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=cfg.quant.algorithm,
            algorithm_version="0.1.0",
            target_dtype=target_dtype,
            scale_granularity=granularity,
            scale_axis=axis,
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
            scale_block_size=block_size,
            scale_dtype="float8_e8m0fnu" if granularity == "block" else "fp32",
        ),
    )
    return index


class TestInt8TensorwiseSelection(unittest.TestCase):
    def test_index_builds_with_int8_tensorwise_dtype(self):
        # Guards the dtype-registration coupling: target_dtype flows into
        # QuantTensorMetadata.__post_init__ -> get_dtype_spec, which KeyErrors
        # unless 'int8_tensorwise' is in KNOWN_DTYPES.
        index = _selection_for_config("qwen_image", "qwen_image_2512_int8_tensorwise.yaml", "int8_tensorwise", "per_channel", "out_features")
        self.assertEqual(index["format"]["name"], "int8_tensorwise")
        self.assertEqual(index["format"]["storage_dtype"], "int8")
        for row in index["tensors"]:
            self.assertEqual(row["quant_dtype"], "int8_tensorwise")
            self.assertEqual(row["storage_dtype"], "int8")

    def test_selection_membership_matches_family_mxfp8_config(self):
        for family, stem in _FAMILIES:
            with self.subTest(family=family):
                int8_idx = _selection_for_config(family, f"{stem}_int8_tensorwise.yaml", "int8_tensorwise", "per_channel", "out_features")
                mxfp8_idx = _selection_for_config(family, f"{stem}_mxfp8.yaml", "mxfp8", "block", "in_features", 32)
                int8_sel = {row["name"] for row in int8_idx["tensors"]}
                mxfp8_sel = {row["name"] for row in mxfp8_idx["tensors"]}
                self.assertGreater(len(int8_sel), 0)
                self.assertEqual(int8_sel, mxfp8_sel)


if __name__ == "__main__":
    unittest.main()
