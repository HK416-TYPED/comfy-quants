import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from comfy_quants.backends.nvfp4_model_export import (
    _quantize_nvfp4_maybe_convrot,
    write_nvfp4_inference_checkpoint_from_safetensors,
)
from comfy_quants.cli.main import main
from comfy_quants.core.artifact_layout import DEFAULT_ARTIFACT_PAYLOAD_LAYOUT
from comfy_quants.formats.nvfp4 import nvfp4_checkpoint_quant_config


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
        "quant_dtype": "nvfp4",
        "storage_dtype": "uint8",
        "algorithm": "nvfp4",
        "scale": {
            "dtype": "float8_e4m3fn",
            "shape": [out_features, in_features // 16],
            "granularity": "block",
            "axis": "in_features",
            "block_size": 16,
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
            "name": "nvfp4",
            "storage_dtype": "uint8",
            "scale_granularity": "block",
            "scale_axis": "in_features",
            "scale_method": "amax",
            "rounding": "nearest_even",
        },
        "selection": {"algorithm": "nvfp4", "algorithm_version": "0.1.0", "target_dtype": "nvfp4", "quantized_tensor_count": len(rows)},
        "tensors": rows,
    }


class TestNvFp4ConvRotExport(unittest.TestCase):
    """ConvRot (EXPERIMENTAL, runtime-pending) on the NVFP4 writer.

    The marker convention mirrors int8_tensorwise: ``format`` first, then
    ``convrot``/``convrot_groupsize`` only for layers that were actually rotated.
    With ``convrot=False`` (the default) the writer output must stay
    content-identical to the pre-ConvRot writer: no convrot keys in any marker or
    header entry (whole-file bytes are not comparable — safetensors serializes
    ``__metadata__`` in nondeterministic key order).
    """

    def setUp(self):
        deps = _torch_safetensors_deps()
        if deps is None:
            self.skipTest("torch and safetensors are required")
        self.torch, self.safe_open, self.load_file, self.save_file = deps

    def test_marker_builder(self):
        self.assertEqual(nvfp4_checkpoint_quant_config(), {"format": "nvfp4"})
        marker = nvfp4_checkpoint_quant_config(convrot=True)
        self.assertEqual(
            json.dumps(marker).encode("utf-8"),
            b'{"format": "nvfp4", "convrot": true, "convrot_groupsize": 256}',
        )
        self.assertEqual(
            json.dumps(nvfp4_checkpoint_quant_config(convrot=False)).encode("utf-8"),
            b'{"format": "nvfp4"}',
        )
        self.assertEqual(
            json.dumps(nvfp4_checkpoint_quant_config(convrot=True, convrot_groupsize=64)).encode("utf-8"),
            b'{"format": "nvfp4", "convrot": true, "convrot_groupsize": 64}',
        )

    def test_quantize_helper_composes_rotation_and_block_quant(self):
        torch = self.torch
        from comfy_quants.formats.convrot import build_hadamard, rotate_weight
        from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                w = torch.randn(8, 512, dtype=dtype)
                q, ws, ws2, rotated = _quantize_nvfp4_maybe_convrot(w, convrot=True, group_size=256)
                self.assertTrue(rotated)
                # Bit-exact composition of the two already-parity-locked primitives:
                # rotation at the SOURCE weight dtype, then the unchanged nvfp4 path.
                h = build_hadamard(256, device=w.device, dtype=w.dtype)
                q_ref, ws_ref, ws2_ref = quantize_nvfp4_block(rotate_weight(w, h, 256))
                self.assertTrue(torch.equal(q, q_ref))
                self.assertTrue(torch.equal(ws.view(torch.uint8), ws_ref.view(torch.uint8)))
                self.assertTrue(torch.equal(ws2, ws2_ref))

    def test_quantize_helper_skips_rotation_when_not_divisible(self):
        torch = self.torch
        from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

        torch.manual_seed(1)
        w = torch.randn(8, 64, dtype=torch.bfloat16)  # 64 % 256 != 0 but 64 % 16 == 0
        q, ws, ws2, rotated = _quantize_nvfp4_maybe_convrot(w, convrot=True, group_size=256)
        self.assertFalse(rotated)
        q_ref, ws_ref, ws2_ref = quantize_nvfp4_block(w)
        self.assertTrue(torch.equal(q, q_ref))
        self.assertTrue(torch.equal(ws.view(torch.uint8), ws_ref.view(torch.uint8)))
        self.assertTrue(torch.equal(ws2, ws2_ref))

    def test_rotated_linear_preserves_output(self):
        """Runtime contract semantics: rotating weight groups offline and activation
        groups online preserves the Linear output (H is symmetric orthogonal)."""
        torch = self.torch
        from comfy_quants.formats.convrot import build_hadamard, rotate_activation, rotate_weight

        torch.manual_seed(2)
        w = torch.randn(8, 512, dtype=torch.float32)
        x = torch.randn(3, 512, dtype=torch.float32)
        h = build_hadamard(256, device="cpu", dtype=torch.float32)
        y_ref = x @ w.T
        y_rot = rotate_activation(x, h, 256) @ rotate_weight(w, h, 256).T
        self.assertTrue(torch.allclose(y_rot, y_ref, atol=1e-4, rtol=1e-4))

    def test_writer_convrot_markers_and_report(self):
        torch = self.torch
        rotatable = "transformer_blocks.0.attn.to_q.weight"
        unrotatable = "transformer_blocks.0.attn.to_k.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.nvfp4.safetensors"
            self.save_file(
                {
                    rotatable: torch.randn(8, 512, dtype=torch.float32),
                    unrotatable: torch.randn(8, 64, dtype=torch.float32),
                },
                str(source),
            )
            report = write_nvfp4_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(rotatable, 8, 512), _tensor_row(unrotatable, 8, 64)]),
                convrot=True,
                device="cpu",  # pin: the payload oracle below is computed on cpu, and
                # matmul reduction order (rotation) is not bit-deterministic across devices
            )
            self.assertEqual(report.status, "model_written")
            self.assertTrue(report.convrot)
            self.assertEqual(report.convrot_groupsize, 256)
            self.assertEqual(report.quantized_tensor_count, 2)
            self.assertEqual(report.rotated_tensor_count, 1)
            self.assertEqual(report.nonrotated_tensor_count, 1)

            exported = self.load_file(str(output))
            marker_rot = bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist())
            self.assertEqual(marker_rot, b'{"format": "nvfp4", "convrot": true, "convrot_groupsize": 256}')
            marker_plain = bytes(exported["transformer_blocks.0.attn.to_k.comfy_quant"].tolist())
            self.assertEqual(marker_plain, b'{"format": "nvfp4"}')
            # Storage tensors keep the plain nvfp4 layout either way.
            self.assertEqual(list(exported[rotatable].shape), [8, 256])
            self.assertEqual(exported["transformer_blocks.0.attn.to_q.weight_scale"].dtype, torch.float8_e4m3fn)
            self.assertEqual(exported["transformer_blocks.0.attn.to_q.weight_scale_2"].dim(), 0)
            # Independent payload oracle: the WRITER's on-disk output must equal the
            # rotate-then-quantize reference computed from the saved source tensor —
            # a marker stamped over an unrotated payload must fail here.
            from comfy_quants.formats.convrot import build_hadamard, rotate_weight
            from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

            w_src = self.load_file(str(source))[rotatable]
            h = build_hadamard(256, device="cpu", dtype=w_src.dtype)
            q_ref, ws_ref, ws2_ref = quantize_nvfp4_block(rotate_weight(w_src, h, 256))
            q_plain, _, _ = quantize_nvfp4_block(w_src)
            self.assertFalse(torch.equal(q_ref, q_plain))  # rotation must actually change the payload
            self.assertTrue(torch.equal(exported[rotatable], q_ref))
            self.assertTrue(
                torch.equal(
                    exported["transformer_blocks.0.attn.to_q.weight_scale"].view(torch.uint8),
                    ws_ref.view(torch.uint8),
                )
            )
            self.assertTrue(torch.equal(exported["transformer_blocks.0.attn.to_q.weight_scale_2"], ws2_ref))

            with self.safe_open(str(output), framework="pt", device="cpu") as handle:
                header = handle.metadata() or {}
            # header values are json.dumps-encoded strings (writer _metadata_value)
            self.assertEqual(header.get("convrot"), "true")
            self.assertEqual(header.get("convrot_groupsize"), "256")

    def test_writer_nondefault_groupsize(self):
        torch = self.torch
        from comfy_quants.formats.convrot import build_hadamard, rotate_weight
        from comfy_quants.formats.nvfp4_blocked import quantize_nvfp4_block

        rotatable = "transformer_blocks.0.attn.to_q.weight"  # 512 % 64 == 0
        unrotatable = "transformer_blocks.0.attn.to_k.weight"  # 96 % 16 == 0, 96 % 64 != 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.nvfp4.safetensors"
            self.save_file(
                {
                    rotatable: torch.randn(8, 512, dtype=torch.float32),
                    unrotatable: torch.randn(8, 96, dtype=torch.float32),
                },
                str(source),
            )
            report = write_nvfp4_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(rotatable, 8, 512), _tensor_row(unrotatable, 8, 96)]),
                convrot=True,
                convrot_groupsize=64,
                device="cpu",  # pin for the cpu-computed payload oracle below
            )
            self.assertEqual(report.convrot_groupsize, 64)
            self.assertEqual(report.rotated_tensor_count, 1)
            self.assertEqual(report.nonrotated_tensor_count, 1)
            exported = self.load_file(str(output))
            marker = bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist())
            self.assertEqual(marker, b'{"format": "nvfp4", "convrot": true, "convrot_groupsize": 64}')
            marker_plain = bytes(exported["transformer_blocks.0.attn.to_k.comfy_quant"].tolist())
            self.assertEqual(marker_plain, b'{"format": "nvfp4"}')
            w_src = self.load_file(str(source))[rotatable]
            h = build_hadamard(64, device="cpu", dtype=w_src.dtype)
            q_ref, _ws_ref, _ws2_ref = quantize_nvfp4_block(rotate_weight(w_src, h, 64))
            self.assertTrue(torch.equal(exported[rotatable], q_ref))

    def test_writer_preserves_source_header_metadata(self):
        # Stock ComfyUI builds e.g. the LTX-2 architecture from __metadata__["config"];
        # the writer must carry source header metadata through (artifact keys win),
        # with convrot enabled as well.
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.safetensors"
            self.save_file(
                {tensor_name: torch.randn(8, 512, dtype=torch.float32)},
                str(source),
                metadata={"config": '{"transformer": {"num_layers": 48}}', "model_version": "2.3.0"},
            )
            write_nvfp4_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(tensor_name, 8, 512)]),
                convrot=True,
            )
            with self.safe_open(str(output), framework="pt", device="cpu") as handle:
                meta = handle.metadata()
            self.assertEqual(meta["config"], '{"transformer": {"num_layers": 48}}')
            self.assertEqual(meta["model_version"], "2.3.0")
            self.assertEqual(meta["target_dtype"], "nvfp4")  # artifact keys still present
            self.assertEqual(meta["convrot"], "true")

    def test_default_off_emits_no_convrot_keys(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.safetensors"
            output = root / "model.nvfp4.safetensors"
            self.save_file({tensor_name: torch.randn(8, 512, dtype=torch.float32)}, str(source))
            report = write_nvfp4_inference_checkpoint_from_safetensors(
                source_checkpoint=source,
                output_checkpoint=output,
                tensor_index=_tensor_index([_tensor_row(tensor_name, 8, 512)]),
            )
            self.assertFalse(report.convrot)
            self.assertEqual(report.rotated_tensor_count, 0)
            self.assertEqual(report.nonrotated_tensor_count, 1)
            exported = self.load_file(str(output))
            marker = bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist())
            self.assertEqual(marker, b'{"format": "nvfp4"}')
            with self.safe_open(str(output), framework="pt", device="cpu") as handle:
                header = handle.metadata() or {}
            self.assertNotIn("convrot", header)
            self.assertNotIn("convrot_groupsize", header)

    def test_cli_export_model_nvfp4_convrot(self):
        torch = self.torch
        tensor_name = "transformer_blocks.0.attn.to_q.weight"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "qwen-one-tensor.safetensors"
            self.save_file({tensor_name: torch.randn(3072, 3072, dtype=torch.bfloat16)}, str(source))
            config = root / "config.yaml"
            config.write_text(
                f"""
project:
  name: qwen-one-tensor-nvfp4-convrot
model:
  family: qwen_image
  model_id: {source.name}
  source: local
  dtype: bf16
quant:
  algorithm: nvfp4
  target_dtype: nvfp4
  scale:
    granularity: block
    axis: in_features
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
                rc = main(
                    [
                        "export-model-nvfp4",
                        "--config",
                        str(config),
                        "--source",
                        str(source),
                        "--out",
                        str(output),
                        "--convrot",
                        "--json",
                        "--no-progress",
                    ]
                )
            self.assertEqual(rc, 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["status"], "model_written")
            self.assertTrue(result["convrot"])
            self.assertEqual(result["rotated_tensor_count"], 1)
            exported = self.load_file(str(output))
            marker = bytes(exported["transformer_blocks.0.attn.to_q.comfy_quant"].tolist())
            self.assertEqual(marker, b'{"format": "nvfp4", "convrot": true, "convrot_groupsize": 256}')
            report = json.loads((root / "export" / "model.export_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["convrot"])
            self.assertEqual(report["rotated_tensor_count"], 1)
            self.assertEqual(report["nonrotated_tensor_count"], 0)


if __name__ == "__main__":
    unittest.main()
