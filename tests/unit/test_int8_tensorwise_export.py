import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.backends.int8_tensorwise_model_export import (
    write_int8_tensorwise_inference_checkpoint_from_safetensors,
)
from comfy_quants.cli.main import main
from comfy_quants.core.artifact_layout import DEFAULT_ARTIFACT_PAYLOAD_LAYOUT
from comfy_quants.formats.registry import get_format


def _torch_safetensors_deps():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError:
        return None
    return torch, load_file, save_file


def _single_tensor_index(in_features: int = 256):
    tensor_name = "transformer_blocks.0.attn.to_q.weight"
    return {
        "schema_version": "quant_tensor_index.v1",
        "artifact_state": "model_export",
        "tensor_payload_state": "written_in_checkpoint",
        "payload_layout": DEFAULT_ARTIFACT_PAYLOAD_LAYOUT.to_dict(),
        "format": {
            "name": "int8_tensorwise",
            "storage_dtype": "int8",
            "scale_granularity": "per_channel",
            "scale_axis": "out_features",
            "scale_method": "amax",
            "rounding": "nearest_even",
        },
        "selection": {"algorithm": "int8_tensorwise", "algorithm_version": "0.1.0", "target_dtype": "int8_tensorwise", "quantized_tensor_count": 1},
        "tensors": [
            {
                "name": tensor_name,
                "source_name": tensor_name,
                "shape": [4, in_features],
                "source_dtype": "bf16",
                "quant_dtype": "int8_tensorwise",
                "storage_dtype": "int8",
                "algorithm": "int8_tensorwise",
                "scale": {"dtype": "fp32", "shape": [4], "granularity": "per_channel", "axis": "out_features", "tensor_name": f"{tensor_name}.scale"},
                "rounding": "nearest_even",
                "fallback": False,
                "compatibility_level": "L2",
                "metadata": {"module_name": "transformer_blocks.0.attn.to_q"},
            }
        ],
    }


class TestInt8TensorwiseExport(unittest.TestCase):
    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.load_file, self.save_file = deps

    def test_format_registered(self):
        fmt = get_format("int8_tensorwise")
        self.assertEqual(fmt.storage_dtype, "int8")
        self.assertEqual(fmt.bits, 8)
        self.assertEqual(fmt.scale_required, True)
        self.assertEqual(fmt.default_scale_granularity, "per_channel")

    def test_writer_emits_stock_marker_with_format_key(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        layer = "transformer_blocks.0.attn.to_q"
        bias_name = "transformer_blocks.0.attn.to_q.bias"
        other = "transformer_blocks.0.norm_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.int8_tensorwise.safetensors"
            self.save_file(
                {
                    tensor_name: torch.randn(4, 256, dtype=torch.float32),
                    bias_name: torch.ones((4,), dtype=torch.float32),
                    other: torch.ones((256,), dtype=torch.float32),
                },
                str(source),
            )
            report = write_int8_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_single_tensor_index(256),
                convrot=True,
            )
            self.assertEqual(report.status, "model_written")
            self.assertEqual(report.schema_version, "int8_tensorwise_checkpoint_export_report.v1")
            self.assertEqual(report.quantized_tensor_count, 1)
            self.assertEqual(report.rotated_tensor_count, 1)  # 256 % 256 == 0
            self.assertEqual(report.copied_tensor_count, 2)   # bias + norm

            exported = self.load_file(str(output))
            self.assertEqual(exported[tensor_name].dtype, torch.int8)
            ws = exported[f"{layer}.weight_scale"]
            self.assertEqual(ws.dtype, torch.float32)
            self.assertEqual(list(ws.shape), [4, 1])
            self.assertNotIn(f"{layer}.input_scale", exported)
            raw = bytes(exported[f"{layer}.comfy_quant"].tolist())
            # byte-exact stock ComfyUI save-path encoding: format first, default separators
            self.assertEqual(raw, b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}')
            marker = json.loads(raw.decode("utf-8"))
            self.assertEqual(marker["format"], "int8_tensorwise")
            self.assertNotIn("per_row", marker)

    def test_no_convrot_marker_omits_convrot_keys(self):
        torch = self.torch
        layer = "transformer_blocks.0.attn.to_q"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file({f"{layer}.weight": torch.randn(4, 256, dtype=torch.float32)}, str(source))
            report = write_int8_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source, output_checkpoint=output, tensor_index=_single_tensor_index(256), convrot=False,
            )
            self.assertEqual(report.rotated_tensor_count, 0)
            exported = self.load_file(str(output))
            raw = bytes(exported[f"{layer}.comfy_quant"].tolist())
            self.assertEqual(raw, b'{"format": "int8_tensorwise"}')

    def test_nondivisible_in_features_skips_rotation(self):
        torch = self.torch
        layer = "transformer_blocks.0.attn.to_q"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file({f"{layer}.weight": torch.randn(4, 192, dtype=torch.float32)}, str(source))
            report = write_int8_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source, output_checkpoint=output, tensor_index=_single_tensor_index(192), convrot=True,
            )
            # 192 % 256 != 0 -> comfy-kitchen would raise; the writer skips rotation
            # and must omit the marker's convrot keys for that layer.
            self.assertEqual(report.rotated_tensor_count, 0)
            self.assertEqual(report.nonrotated_tensor_count, 1)
            exported = self.load_file(str(output))
            raw = bytes(exported[f"{layer}.comfy_quant"].tolist())
            self.assertEqual(raw, b'{"format": "int8_tensorwise"}')

    def test_quant_math_matches_comfy_kitchen_recipe(self):
        # In-repo numeric guard for the CK-faithful math (division at weight dtype,
        # fp32 scale from bf16 amax); the external oracle check lives in
        # test_external_int8_tensorwise_parity.py.
        torch = self.torch
        from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row

        w = torch.randn(8, 512, dtype=torch.bfloat16)
        q, scale, rotated = _quantize_int8_tensorwise_per_row(w, convrot=False, group_size=256)
        self.assertFalse(rotated)
        self.assertEqual(scale.dtype, torch.float32)
        expected_scale = (w.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
        self.assertTrue(torch.equal(scale, expected_scale))
        expected_q = (w / expected_scale.to(w.dtype)).round_().clamp_(-128.0, 127.0).to(torch.int8)
        self.assertTrue(torch.equal(q, expected_q))

    def test_writer_preserves_source_header_metadata(self):
        # Stock ComfyUI builds e.g. the LTX-2 architecture from __metadata__["config"];
        # the writer must carry source header metadata through (artifact keys win).
        torch = self.torch
        from safetensors import safe_open

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
            write_int8_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source, output_checkpoint=output, tensor_index=_single_tensor_index(256),
            )
            with safe_open(str(output), framework="pt") as handle:
                meta = handle.metadata()
            self.assertEqual(meta["config"], '{"transformer": {"num_layers": 48}}')
            self.assertEqual(meta["model_version"], "2.3.0")
            self.assertEqual(meta["target_dtype"], "int8_tensorwise")  # artifact keys still present

    def test_writer_emits_index_timestep_zero_sentinel_for_qwen_edit_2511(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file({tensor_name: torch.randn(4, 256, dtype=torch.float32)}, str(source))
            write_int8_tensorwise_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_single_tensor_index(256),
                metadata={"model_family": "qwen_image_edit", "model_id": "Qwen/Qwen-Image-Edit-2511"},
            )
            exported = self.load_file(str(output))
            self.assertIn("__index_timestep_zero__", exported)
            self.assertEqual(exported["__index_timestep_zero__"].numel(), 0)

    def test_writer_rejects_overwriting_source(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.safetensors"
            self.save_file({"transformer_blocks.0.attn.to_q.weight": torch.randn(4, 256, dtype=torch.float32)}, str(source))
            with self.assertRaisesRegex(Exception, "must not overwrite"):
                write_int8_tensorwise_inference_checkpoint_from_safetensors(
                    source_checkpoint=source, output_checkpoint=source, tensor_index=_single_tensor_index(256),
                )

    def test_cli_export_model_int8_tensorwise(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "qwen-one-tensor.safetensors"
            self.save_file({tensor_name: torch.zeros((3072, 3072), dtype=torch.bfloat16)}, str(source))
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: qwen-one-tensor-int8-tensorwise
model:
  family: qwen_image
  model_id: {source.name}
  source: local
  dtype: bf16
quant:
  algorithm: int8_tensorwise
  target_dtype: int8_tensorwise
  scale:
    granularity: per_channel
    axis: out_features
    method: amax
  rounding: nearest_even
  modules:
    include:
      - transformer_blocks.0.attn.to_q
    exclude: []
artifact:
  compatibility_target: L2
""",
                encoding="utf-8",
            )
            output = root / "export" / "model.safetensors"
            captured = StringIO()
            with redirect_stdout(captured):
                rc = main(["export-model-int8-tensorwise", "--config", str(config), "--source", str(source), "--out", str(output), "--json", "--no-progress"])
            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "model_written")
            self.assertTrue(result["convrot"])
            exported = self.load_file(str(output))
            self.assertEqual(exported[tensor_name].dtype, torch.int8)
            self.assertEqual(list(exported["transformer_blocks.0.attn.to_q.weight_scale"].shape), [3072, 1])
            marker = json.loads(bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist()).decode("utf-8"))
            self.assertEqual(marker["format"], "int8_tensorwise")

    def test_cli_rejects_non_int8_tensorwise_target(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.safetensors"
            self.save_file({"transformer_blocks.0.attn.to_q.weight": torch.zeros((8, 256), dtype=torch.bfloat16)}, str(source))
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
  algorithm: int8_w8a8
  target_dtype: int8_w8a8
""",
                encoding="utf-8",
            )
            rc = main(["export-model-int8-tensorwise", "--config", str(config), "--source", str(source), "--out", str(root / "out"), "--json", "--no-progress"])
            self.assertEqual(rc, 2)  # ConfigurationError -> handle_cli_error -> exit 2


if __name__ == "__main__":
    unittest.main()
