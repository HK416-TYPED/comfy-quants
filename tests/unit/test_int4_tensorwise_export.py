import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.backends.int4_tensorwise_model_export import (
    _quantize_int4_tensorwise_per_row,
    write_int4_tensorwise_inference_checkpoint_from_safetensors,
)
from comfy_quants.cli.main import main
from comfy_quants.core.artifact_layout import DEFAULT_ARTIFACT_PAYLOAD_LAYOUT
from comfy_quants.formats.int4_tensorwise import int4_tensorwise_checkpoint_quant_config
from comfy_quants.formats.registry import get_format


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    return torch, safe_open, load_file, save_file


def _tensor_row(tensor_name: str, out_features: int, in_features: int) -> dict:
    return {
        "name": tensor_name,
        "source_name": tensor_name,
        "shape": [out_features, in_features],
        "source_dtype": "bf16",
        "quant_dtype": "int4_tensorwise",
        "storage_dtype": "int8",
        "algorithm": "int4_tensorwise",
        "scale": {
            "dtype": "fp32",
            "shape": [out_features, 1],
            "granularity": "per_channel",
            "axis": "out_features",
            "tensor_name": f"{tensor_name}.scale",
        },
        "rounding": "nearest_even",
        "fallback": False,
        "compatibility_level": "L2",
        "metadata": {"module_name": tensor_name.rsplit(".", 1)[0]},
    }


def _tensor_index(rows: list[dict]) -> dict:
    return {
        "schema_version": "quant_tensor_index.v1",
        "artifact_state": "model_export",
        "tensor_payload_state": "written_in_checkpoint",
        "payload_layout": DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.to_dict(),
        "format": {
            "name": "int4_tensorwise",
            "storage_dtype": "int8",
            "scale_granularity": "per_channel",
            "scale_axis": "out_features",
            "scale_method": "amax",
            "rounding": "nearest_even",
        },
        "selection": {
            "algorithm": "int4_tensorwise",
            "algorithm_version": "0.1.0",
            "target_dtype": "int4_tensorwise",
            "quantized_tensor_count": len(rows),
        },
        "tensors": rows,
    }


class TestInt4TensorwiseExport(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.safe_open, self.load_file, self.save_file = deps

    def test_format_registered(self):
        fmt = get_format("int4_tensorwise")
        self.assertEqual(fmt.storage_dtype, "int8")
        self.assertEqual(fmt.bits, 4)
        self.assertEqual(fmt.default_scale_granularity, "per_channel")
        self.assertEqual(fmt.metadata["quant_min"], -7)
        self.assertEqual(fmt.metadata["quant_max"], 7)
        self.assertEqual(fmt.metadata["weight_packing"], "int4_pairs_low_nibble_first")

    def test_marker_builder_bytes(self):
        self.assertEqual(
            json.dumps(int4_tensorwise_checkpoint_quant_config(convrot=True)).encode("utf-8"),
            b'{"format": "int4_tensorwise", "convrot": true, "convrot_groupsize": 256}',
        )
        self.assertEqual(
            json.dumps(int4_tensorwise_checkpoint_quant_config(convrot=False)).encode("utf-8"),
            b'{"format": "int4_tensorwise"}',
        )
        self.assertEqual(
            json.dumps(int4_tensorwise_checkpoint_quant_config(convrot=True, convrot_groupsize=64)).encode("utf-8"),
            b'{"format": "int4_tensorwise", "convrot": true, "convrot_groupsize": 64}',
        )

    def test_quant_math_matches_kitchen_recipe(self):
        # In-repo numeric guard for the kitchen-faithful math (rotation + division at
        # weight dtype, fp32 amax/7 scale, [-7,7], low-nibble-first pack); the external
        # oracle check lives in test_external_int4_tensorwise_parity.py.
        torch = self.torch
        from comfy_quants.formats.convrot import build_hadamard, rotate_weight
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs, unpack_signed_int4_pairs

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                torch.manual_seed(0)
                w = torch.randn(8, 512, dtype=dtype)
                packed, scale, rotated = _quantize_int4_tensorwise_per_row(w, convrot=True, group_size=256)
                self.assertTrue(rotated)
                self.assertEqual(list(packed.shape), [8, 256])
                self.assertEqual(packed.dtype, torch.int8)
                self.assertEqual(scale.dtype, torch.float32)
                self.assertEqual(list(scale.shape), [8, 1])

                h = build_hadamard(256, device="cpu", dtype=w.dtype)
                w_rot = rotate_weight(w, h, 256)
                expected_scale = (w_rot.abs().amax(dim=-1, keepdim=True).float() / 7.0).clamp(min=1e-30)
                self.assertTrue(torch.equal(scale, expected_scale))
                expected_codes = (w_rot / expected_scale.to(dtype)).round().clamp(-7, 7).to(torch.int8)
                self.assertTrue(torch.equal(packed, pack_signed_int4_pairs(expected_codes)))
                codes = unpack_signed_int4_pairs(packed)
                self.assertGreaterEqual(int(codes.min()), -7)
                self.assertLessEqual(int(codes.max()), 7)

    def test_quant_skips_rotation_when_not_divisible(self):
        torch = self.torch
        w = torch.randn(8, 64, dtype=torch.bfloat16)  # 64 % 256 != 0, 64 even
        packed, scale, rotated = _quantize_int4_tensorwise_per_row(w, convrot=True, group_size=256)
        self.assertFalse(rotated)
        self.assertEqual(list(packed.shape), [8, 32])

    def test_quant_rejects_odd_in_features(self):
        torch = self.torch
        with self.assertRaisesRegex(Exception, "even in_features"):
            _quantize_int4_tensorwise_per_row(torch.randn(4, 7), convrot=False, group_size=256)

    def test_writer_mixed_markers_payloads_and_report(self):
        torch = self.torch
        int4_name = "transformer_blocks.0.attn.to_q.weight"
        int8_name = "transformer_blocks.0.attn.to_v.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file(
                {
                    int4_name: torch.randn(8, 512, dtype=torch.float32),
                    int8_name: torch.randn(8, 512, dtype=torch.float32),
                },
                str(source),
            )
            report = write_int4_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(int4_name, 8, 512), _tensor_row(int8_name, 8, 512)]),
                convrot=True,
                int8_fallback=["*.to_v"],
                device="cpu",  # payload oracles below are computed on cpu
            )
            self.assertEqual(report.status, "model_written")
            self.assertEqual(report.quantized_tensor_count, 2)
            self.assertEqual(report.int4_tensor_count, 1)
            self.assertEqual(report.int8_fallback_tensor_count, 1)
            self.assertEqual(report.int8_fallback_tensors, ["transformer_blocks.0.attn.to_v"])
            self.assertEqual(report.rotated_tensor_count, 2)

            exported = self.load_file(str(output))
            # INT4 layer: packed [N, K/2] + int4 marker
            self.assertEqual(list(exported[int4_name].shape), [8, 256])
            marker4 = bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist())
            self.assertEqual(marker4, b'{"format": "int4_tensorwise", "convrot": true, "convrot_groupsize": 256}')
            # INT8 fallback layer: full-shape int8 + int8_tensorwise marker
            self.assertEqual(list(exported[int8_name].shape), [8, 512])
            marker8 = bytes(exported["transformer_blocks.0.attn.to_v.comfy_quant"].tolist())
            self.assertEqual(marker8, b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}')
            # Both carry fp32 [out, 1] scales, no input_scale
            for layer in ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_v"):
                self.assertEqual(exported[f"{layer}.weight_scale"].dtype, torch.float32)
                self.assertEqual(list(exported[f"{layer}.weight_scale"].shape), [8, 1])
                self.assertNotIn(f"{layer}.input_scale", exported)

            # Independent payload oracles from the saved source tensors.
            src = self.load_file(str(source))
            q4_ref, s4_ref, rotated = _quantize_int4_tensorwise_per_row(src[int4_name], convrot=True, group_size=256)
            self.assertTrue(rotated)
            self.assertTrue(torch.equal(exported[int4_name], q4_ref))
            self.assertTrue(torch.equal(exported["transformer_blocks.0.attn.to_q.weight_scale"], s4_ref))
            from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row

            q8_ref, s8_ref, rotated8 = _quantize_int8_tensorwise_per_row(src[int8_name], convrot=True, group_size=256)
            self.assertTrue(rotated8)
            self.assertTrue(torch.equal(exported[int8_name], q8_ref))
            self.assertTrue(torch.equal(exported["transformer_blocks.0.attn.to_v.weight_scale"], s8_ref))

            with self.safe_open(str(output), framework="pt", device="cpu") as handle:
                header = handle.metadata() or {}
            self.assertEqual(header.get("target_dtype"), "int4_tensorwise")
            self.assertEqual(header.get("int4_tensor_count"), "1")
            self.assertEqual(header.get("int8_fallback_tensor_count"), "1")

    def test_writer_nondivisible_in_features_skips_rotation(self):
        # Mirrors the int8 sibling: even-but-not-divisible in_features must export
        # an unrotated payload whose marker omits the convrot keys.
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file({tensor_name: torch.randn(4, 192, dtype=torch.float32)}, str(source))
            report = write_int4_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source, output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(tensor_name, 4, 192)]),
                convrot=True, device="cpu",
            )
            self.assertEqual(report.rotated_tensor_count, 0)
            self.assertEqual(report.nonrotated_tensor_count, 1)
            exported = self.load_file(str(output))
            marker = bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist())
            self.assertEqual(marker, b'{"format": "int4_tensorwise"}')
            self.assertEqual(list(exported[tensor_name].shape), [4, 96])

    def test_writer_nondefault_groupsize_end_to_end(self):
        # 320 % 64 == 0 but 320 % 256 != 0: group 64 must rotate where group 256
        # would not, and the marker must carry the actual group size on BOTH the
        # int4 and the int8-fallback layer.
        torch = self.torch
        int4_name = "transformer_blocks.0.attn.to_q.weight"
        int8_name = "transformer_blocks.0.attn.to_v.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file(
                {
                    int4_name: torch.randn(8, 320, dtype=torch.float32),
                    int8_name: torch.randn(8, 320, dtype=torch.float32),
                },
                str(source),
            )
            report = write_int4_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source, output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(int4_name, 8, 320), _tensor_row(int8_name, 8, 320)]),
                convrot=True, convrot_groupsize=64, int8_fallback=["*.to_v"], device="cpu",
            )
            self.assertEqual(report.convrot_groupsize, 64)
            self.assertEqual(report.rotated_tensor_count, 2)
            exported = self.load_file(str(output))
            marker4 = bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist())
            self.assertEqual(marker4, b'{"format": "int4_tensorwise", "convrot": true, "convrot_groupsize": 64}')
            marker8 = bytes(exported["transformer_blocks.0.attn.to_v.comfy_quant"].tolist())
            self.assertEqual(marker8, b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 64}')
            # The rotation math must actually use group 64 (independent oracle).
            src = self.load_file(str(source))
            q_ref, s_ref, rotated = _quantize_int4_tensorwise_per_row(src[int4_name], convrot=True, group_size=64)
            self.assertTrue(rotated)
            self.assertTrue(torch.equal(exported[int4_name], q_ref))
            self.assertTrue(torch.equal(exported["transformer_blocks.0.attn.to_q.weight_scale"], s_ref))

    def test_writer_pure_int4_without_fallback(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file({tensor_name: torch.randn(8, 512, dtype=torch.float32)}, str(source))
            report = write_int4_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(tensor_name, 8, 512)]),
            )
            self.assertEqual(report.int4_tensor_count, 1)
            self.assertEqual(report.int8_fallback_tensor_count, 0)
            self.assertEqual(report.int8_fallback_globs, [])

    def test_writer_preserves_source_header_metadata(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file(
                {tensor_name: torch.randn(4, 256, dtype=torch.float32)},
                str(source),
                metadata={"config": '{"transformer": {"num_layers": 48}}', "model_version": "2.3.0"},
            )
            write_int4_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source, output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(tensor_name, 4, 256)]),
            )
            with self.safe_open(str(output), framework="pt", device="cpu") as handle:
                meta = handle.metadata()
            self.assertEqual(meta["config"], '{"transformer": {"num_layers": 48}}')
            self.assertEqual(meta["model_version"], "2.3.0")
            self.assertEqual(meta["target_dtype"], "int4_tensorwise")

    def test_writer_emits_index_timestep_zero_sentinel_for_qwen_edit_2511(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file({tensor_name: torch.randn(4, 256, dtype=torch.float32)}, str(source))
            write_int4_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source, output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(tensor_name, 4, 256)]),
                metadata={"model_id": "Qwen/Qwen-Image-Edit-2511", "model_family": "qwen_image_edit"},
            )
            exported = self.load_file(str(output))
            self.assertIn("__index_timestep_zero__", exported)

    def test_writer_rejects_overwriting_source(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            self.save_file({"transformer_blocks.0.attn.to_q.weight": torch.randn(4, 256, dtype=torch.float32)}, str(source))
            with self.assertRaisesRegex(Exception, "must not overwrite"):
                write_int4_tensorwise_inference_checkpoint_from_safetensors(
                    source_checkpoint=source, output_checkpoint=source,
                    tensor_index=_tensor_index([_tensor_row("transformer_blocks.0.attn.to_q.weight", 4, 256)]),
                )

    def test_cli_export_model_int4_tensorwise_mixed(self):
        torch = self.torch
        to_q = "transformer_blocks.0.attn.to_q.weight"
        to_v = "transformer_blocks.0.attn.to_v.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "qwen-two-tensor.safetensors"
            self.save_file(
                {
                    to_q: torch.randn(3072, 3072, dtype=torch.bfloat16),
                    to_v: torch.randn(3072, 3072, dtype=torch.bfloat16),
                },
                str(source),
            )
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: qwen-int4-tensorwise-mixed
model:
  family: qwen_image
  model_id: {source.name}
  source: local
  dtype: bf16
quant:
  algorithm: int4_tensorwise
  target_dtype: int4_tensorwise
  scale:
    granularity: per_channel
    axis: out_features
    method: amax
  rounding: nearest_even
  modules:
    include:
      - transformer_blocks.0.attn.to_q
      - transformer_blocks.0.attn.to_v
    exclude: []
    int8_fallback:
      - "*.to_v"
artifact:
  compatibility_target: L2
""",
                encoding="utf-8",
            )
            output = root / "export" / "model.safetensors"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    ["export-model-int4-tensorwise", "--config", str(config), "--source", str(source),
                     "--out", str(output), "--json", "--no-progress"]
                )
            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "model_written")
            self.assertEqual(result["int4_tensor_count"], 1)
            self.assertEqual(result["int8_fallback_tensor_count"], 1)
            self.assertTrue(result["convrot"])
            exported = self.load_file(str(output))
            self.assertEqual(list(exported[to_q].shape), [3072, 1536])  # packed
            self.assertEqual(list(exported[to_v].shape), [3072, 3072])  # int8 fallback
            marker4 = bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist())
            self.assertEqual(marker4, b'{"format": "int4_tensorwise", "convrot": true, "convrot_groupsize": 256}')
            marker8 = bytes(exported["transformer_blocks.0.attn.to_v.comfy_quant"].tolist())
            self.assertEqual(marker8, b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}')
            report = json.loads((root / "export" / "model.export_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["int8_fallback_tensors"], ["transformer_blocks.0.attn.to_v"])

    def test_cli_int8_fallback_flag_overrides_config(self):
        # The CLI flag REPLACES (not merges with) the config's int8_fallback list:
        # config says to_v -> int8, the flag says to_q -> int8; the export must have
        # to_q as the ONLY int8 layer.
        torch = self.torch
        to_q = "transformer_blocks.0.attn.to_q.weight"
        to_v = "transformer_blocks.0.attn.to_v.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.safetensors"
            self.save_file(
                {
                    to_q: torch.randn(3072, 3072, dtype=torch.bfloat16),
                    to_v: torch.randn(3072, 3072, dtype=torch.bfloat16),
                },
                str(source),
            )
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: qwen-int4-flag-override
model:
  family: qwen_image
  model_id: {source.name}
  source: local
  dtype: bf16
quant:
  algorithm: int4_tensorwise
  target_dtype: int4_tensorwise
  modules:
    include:
      - transformer_blocks.0.attn.to_q
      - transformer_blocks.0.attn.to_v
    exclude: []
    int8_fallback:
      - "*.to_v"
""",
                encoding="utf-8",
            )
            output = root / "export" / "model.safetensors"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(
                    ["export-model-int4-tensorwise", "--config", str(config), "--source", str(source),
                     "--out", str(output), "--int8-fallback", "*.to_q", "--json", "--no-progress"]
                )
            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["int4_tensor_count"], 1)
            self.assertEqual(result["int8_fallback_tensor_count"], 1)
            report = json.loads((root / "export" / "model.export_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["int8_fallback_tensors"], ["transformer_blocks.0.attn.to_q"])

    def test_cli_rejects_non_int4_tensorwise_target(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.safetensors"
            self.save_file({"transformer_blocks.0.attn.to_q.weight": torch.zeros((8, 64), dtype=torch.bfloat16)}, str(source))
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: wrong-target
model:
  family: qwen_image
  model_id: {source.name}
  source: local
quant:
  algorithm: int8_tensorwise
  target_dtype: int8_tensorwise
""",
                encoding="utf-8",
            )
            rc = main(["export-model-int4-tensorwise", "--config", str(config), "--source", str(source), "--out", str(root / "out"), "--json", "--no-progress"])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
