import os
import unittest
from pathlib import Path


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _ck_root() -> Path:
    """comfy-kitchen checkout root (same cwd-anchored convention as the sibling
    external parity tests)."""
    return Path(os.environ.get("COMFY_QUANTS_COMFY_KITCHEN_SOURCE", str(Path.cwd().parent / "external" / "comfy-kitchen")))


def _load_ck_int4_functions():
    """Return comfy-kitchen's REAL eager int4_tensorwise functions when the checkout
    imports as a package, else SkipTest.

    Requires a checkout with the int4_tensorwise support (branch
    feat/int4-tensorwise-convrot or a release containing it); older checkouts
    lack the functions and the test skips.
    """
    import sys

    root = _ck_root()
    if not (root / "comfy_kitchen" / "backends" / "eager" / "quantization.py").is_file():
        raise unittest.SkipTest(f"comfy-kitchen source is not available at {root}")
    sys.path.insert(0, str(root))
    try:
        from comfy_kitchen.backends.eager import quantization as ck_quant  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - any import failure means no oracle
        raise unittest.SkipTest(f"comfy-kitchen package import failed: {exc}")
    finally:
        sys.path.remove(str(root))
    for fn in ("quantize_int4_rowwise", "quantize_int4_convrot_weight", "dequantize_int4_convrot_weight"):
        if not hasattr(ck_quant, fn):
            raise unittest.SkipTest(f"comfy-kitchen checkout predates int4_tensorwise ({fn} missing)")
    return ck_quant


class TestExternalInt4TensorwiseParity(unittest.TestCase):
    """Bit-faithfulness of the offline writer math to the comfy-kitchen runtime.

    The kitchen eager functions are the requantization path ComfyUI uses (e.g. on
    LoRA offload), so the writer must reproduce them exactly."""

    def setUp(self):
        torch = _torch()
        if torch is None:
            self.skipTest("torch is required")
        self.torch = torch
        self.ck = _load_ck_int4_functions()

    def test_rowwise_quant_bit_faithful(self):
        torch = self.torch
        from comfy_quants.backends.int4_tensorwise_model_export import _quantize_int4_tensorwise_per_row

        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                w = torch.randn(8, 192, dtype=dtype)  # 192 % 256 != 0 -> no rotation
                q_ours, scale_ours, rotated = _quantize_int4_tensorwise_per_row(w, convrot=True, group_size=256)
                self.assertFalse(rotated)
                q_theirs, scale_theirs = self.ck.quantize_int4_rowwise(w.clone())
                self.assertTrue(torch.equal(scale_ours, scale_theirs))
                self.assertTrue(torch.equal(q_ours, q_theirs))

    def test_convrot_weight_quant_bit_faithful(self):
        torch = self.torch
        from comfy_quants.backends.int4_tensorwise_model_export import _quantize_int4_tensorwise_per_row

        torch.manual_seed(1)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                w = torch.randn(16, 1024, dtype=dtype)
                q_ours, scale_ours, rotated = _quantize_int4_tensorwise_per_row(w, convrot=True, group_size=256)
                self.assertTrue(rotated)
                q_theirs, scale_theirs = self.ck.quantize_int4_convrot_weight(w.clone(), 256)
                self.assertTrue(torch.equal(scale_ours, scale_theirs))
                self.assertTrue(torch.equal(q_ours, q_theirs))

    def test_pack_convention_matches_kitchen_unpack(self):
        torch = self.torch
        from comfy_quants.formats.int4_common import pack_signed_int4_pairs, unpack_signed_int4_pairs

        torch.manual_seed(2)
        codes = torch.randint(-7, 8, (8, 64), dtype=torch.int8)
        packed = pack_signed_int4_pairs(codes)
        theirs = self.ck._int4_unpack_rowwise(packed)
        self.assertTrue(torch.equal(theirs, codes))
        self.assertTrue(torch.equal(unpack_signed_int4_pairs(self.ck._int4_pack_rowwise(codes)), codes))

    def test_dequant_convrot_returns_original_basis(self):
        torch = self.torch
        from comfy_quants.backends.int4_tensorwise_model_export import _quantize_int4_tensorwise_per_row

        torch.manual_seed(3)
        w = torch.randn(16, 512, dtype=torch.float32)
        q, scale, rotated = _quantize_int4_tensorwise_per_row(w, convrot=True, group_size=256)
        self.assertTrue(rotated)
        deq = self.ck.dequantize_int4_convrot_weight(q, scale, 256)
        rel = (w - deq).norm() / w.norm()
        self.assertLess(float(rel), 0.2)


if __name__ == "__main__":
    unittest.main()
