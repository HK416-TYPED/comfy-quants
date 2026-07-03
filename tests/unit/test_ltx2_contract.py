import unittest
from pathlib import Path

from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.core.config import load_quant_config
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.ltx2 import (
    OFFICIAL_KEPT_BLOCKS,
    build_ltx2_static_contract,
    official_kept_block_globs,
)
from comfy_quants.model_adapters.registry import get_adapter

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _tensors_by_name(graph):
    return {t.name: t for m in graph.modules for t in m.tensors}


def _index(family, target_dtype, granularity, axis, block_size=None):
    adapter = get_adapter(family)
    _insp, graph = adapter.inspect(ModelSource(family=family, model_id="x"))
    policy = adapter.default_policy(target_dtype)
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=policy.algorithm,
            algorithm_version="0.1.0",
            target_dtype=target_dtype,
            scale_granularity=granularity,
            scale_axis=axis,
            scale_method="amax",
            rounding="nearest_even",
            compatibility_level="L2",
            scale_block_size=block_size,
            scale_dtype="float8_e4m3fn" if granularity == "block" else "fp32",
        ),
    )


class TestLtx2Contract(unittest.TestCase):
    def test_contract(self):
        c = build_ltx2_static_contract()
        self.assertEqual(c.family, "ltx2")
        self.assertEqual(c.schema_version, "ltx2_static_contract.v1")
        self.assertEqual(c.preferred_format, "fp8_e4m3")
        self.assertEqual([(g.prefix, g.count) for g in c.block_groups], [("model.diffusion_model.transformer_blocks", 48)])
        self.assertEqual(c.dimensions()["D"], 4096)
        self.assertEqual(c.dimensions()["A"], 2048)
        self.assertEqual(c.dimensions()["FF"], 16384)
        self.assertEqual(c.dimensions()["AFF"], 8192)
        self.assertEqual(c.dimensions()["G"], 32)
        # 6 attentions x 5 linears + 2 ff x 2 linears = 34 quantizable linears per block
        self.assertEqual(len(c.block_groups[0].modules), 34)

    def test_adapter_registered(self):
        self.assertEqual(get_adapter("ltx2").family, "ltx2")

    def test_graph_tensor_names_and_shapes(self):
        adapter = get_adapter("ltx2")
        _insp, graph = adapter.inspect(ModelSource(family="ltx2", model_id="Lightricks/LTX-2.3"))
        by_name = _tensors_by_name(graph)
        pfx = "model.diffusion_model.transformer_blocks"
        # video stream
        self.assertEqual(by_name[f"{pfx}.0.attn1.to_q.weight"].shape, [4096, 4096])
        self.assertEqual(by_name[f"{pfx}.0.attn2.to_k.weight"].shape, [4096, 4096])
        self.assertEqual(by_name[f"{pfx}.0.attn1.to_gate_logits.weight"].shape, [32, 4096])
        self.assertEqual(by_name[f"{pfx}.0.ff.net.0.proj.weight"].shape, [16384, 4096])
        self.assertEqual(by_name[f"{pfx}.0.ff.net.2.weight"].shape, [4096, 16384])
        # audio stream
        self.assertEqual(by_name[f"{pfx}.0.audio_attn1.to_q.weight"].shape, [2048, 2048])
        self.assertEqual(by_name[f"{pfx}.0.audio_attn2.to_v.weight"].shape, [2048, 2048])
        self.assertEqual(by_name[f"{pfx}.0.audio_attn1.to_gate_logits.weight"].shape, [32, 2048])
        self.assertEqual(by_name[f"{pfx}.0.audio_ff.net.0.proj.weight"].shape, [8192, 2048])
        self.assertEqual(by_name[f"{pfx}.0.audio_ff.net.2.weight"].shape, [2048, 8192])
        # audio->video cross-attention: Q from video, KV from audio, out back to video
        self.assertEqual(by_name[f"{pfx}.0.audio_to_video_attn.to_q.weight"].shape, [2048, 4096])
        self.assertEqual(by_name[f"{pfx}.0.audio_to_video_attn.to_k.weight"].shape, [2048, 2048])
        self.assertEqual(by_name[f"{pfx}.0.audio_to_video_attn.to_out.0.weight"].shape, [4096, 2048])
        self.assertEqual(by_name[f"{pfx}.0.audio_to_video_attn.to_gate_logits.weight"].shape, [32, 4096])
        # video->audio cross-attention: Q from audio, KV from video, out back to audio
        self.assertEqual(by_name[f"{pfx}.0.video_to_audio_attn.to_q.weight"].shape, [2048, 2048])
        self.assertEqual(by_name[f"{pfx}.0.video_to_audio_attn.to_k.weight"].shape, [2048, 4096])
        self.assertEqual(by_name[f"{pfx}.0.video_to_audio_attn.to_v.weight"].shape, [2048, 4096])
        self.assertEqual(by_name[f"{pfx}.0.video_to_audio_attn.to_out.0.weight"].shape, [2048, 2048])
        self.assertEqual(by_name[f"{pfx}.0.video_to_audio_attn.to_gate_logits.weight"].shape, [32, 2048])
        # last block exists with the same layout
        self.assertEqual(by_name[f"{pfx}.47.ff.net.2.weight"].shape, [4096, 16384])
        self.assertEqual(by_name[f"{pfx}.47.audio_to_video_attn.to_out.0.weight"].shape, [4096, 2048])
        self.assertEqual(by_name[f"{pfx}.0.attn1.to_q.weight"].scale_axis, "out_features")
        for name, t in by_name.items():
            if name.startswith(f"{pfx}.") and t.role == "weight":
                self.assertEqual(t.shape[1] % 32, 0, name)

    def test_fp8_selection_matches_official_recipe(self):
        idx = _index("ltx2", "fp8_e4m3", "per_tensor", None)
        sel = {row["name"] for row in idx["tensors"]}
        # 44 blocks (2..45) * 34 linears = 1496, layer-for-layer identical to the
        # official Lightricks/LTX-2.3-fp8 and -nvfp4 releases.
        self.assertEqual(len(sel), 1496)
        pfx = "model.diffusion_model.transformer_blocks"
        self.assertIn(f"{pfx}.2.attn1.to_q.weight", sel)
        self.assertIn(f"{pfx}.2.audio_ff.net.0.proj.weight", sel)
        self.assertIn(f"{pfx}.2.attn1.to_gate_logits.weight", sel)
        self.assertIn(f"{pfx}.45.video_to_audio_attn.to_v.weight", sel)
        # official recipe keeps the first two and last two blocks in bf16
        self.assertEqual(OFFICIAL_KEPT_BLOCKS, (0, 1, 46, 47))
        for block in OFFICIAL_KEPT_BLOCKS:
            for name in sel:
                self.assertFalse(name.startswith(f"{pfx}.{block}."), name)
        for name in sel:
            self.assertNotIn("patchify_proj", name)
            self.assertNotIn("adaln_single", name)
            self.assertNotIn("embeddings_connector", name)
            self.assertNotIn("proj_out", name)
            self.assertNotIn("scale_shift_table", name)

    def test_configs_carry_official_recipe_excludes(self):
        # Config include/exclude OVERRIDES the adapter default policy in the CLI export
        # path, so every shipped ltx2 config must carry the official kept-block globs.
        for name in ["ltx2_fp8.yaml", "ltx2_fp8_e5m2.yaml", "ltx2_mxfp8.yaml", "ltx2_nvfp4.yaml", "ltx2_int8_w8a8.yaml", "ltx2_int8_tensorwise.yaml"]:
            with self.subTest(config=name):
                cfg = load_quant_config(_CONFIG_DIR / name)
                self.assertEqual(cfg.model.family, "ltx2")
                self.assertEqual(cfg.quant.modules["include"], ["model.diffusion_model.transformer_blocks.*"])
                self.assertEqual(cfg.quant.modules["exclude"], official_kept_block_globs())

    def test_block_formats_build(self):
        for target_dtype, block_size in [("mxfp8", 32), ("nvfp4", 16)]:
            with self.subTest(target_dtype=target_dtype):
                idx = _index("ltx2", target_dtype, "block", "in_features", block_size)
                self.assertEqual(idx["format"]["name"], target_dtype)
                self.assertEqual(len({row["name"] for row in idx["tensors"]}), 1496)
                for row in idx["tensors"]:
                    self.assertEqual(row["scale"]["block_size"], block_size)

    def test_int8_selection(self):
        for target_dtype in ["int8_w8a8", "int8_tensorwise"]:
            with self.subTest(target_dtype=target_dtype):
                idx = _index("ltx2", target_dtype, "per_channel", "out_features")
                self.assertEqual(idx["format"]["name"], target_dtype)
                self.assertEqual(idx["format"]["storage_dtype"], "int8")
                self.assertEqual(len({row["name"] for row in idx["tensors"]}), 1496)
                for row in idx["tensors"]:
                    self.assertEqual(row["quant_dtype"], target_dtype)
                    # every ltx2 in_features (4096/2048/16384/8192) is ConvRot-eligible (%256==0)
                    self.assertEqual(row["shape"][1] % 256, 0, row["name"])


if __name__ == "__main__":
    unittest.main()
