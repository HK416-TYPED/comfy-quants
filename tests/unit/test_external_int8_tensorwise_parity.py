import importlib.util
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


def _load_ck_int8_utils():
    """Load comfy-kitchen's tensor/int8_utils.py as the stock-int8 parity oracle.

    int8_utils.py imports only math + torch, so it loads standalone (the eager
    quantization module imports the comfy-kitchen registry and is NOT standalone).
    SkipTest if the comfy-kitchen source is unavailable or predates int8 support
    (< 0.2.11). Set COMFY_QUANTS_COMFY_KITCHEN_SOURCE to override.
    """
    path = _ck_root() / "comfy_kitchen" / "tensor" / "int8_utils.py"
    if not path.is_file():
        raise unittest.SkipTest(f"comfy-kitchen int8_utils.py oracle is not available at {path} (checkout predates int8 support?)")
    spec = importlib.util.spec_from_file_location("_comfy_quants_ck_int8_utils_oracle", path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot import int8_utils.py oracle at {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise unittest.SkipTest(f"int8_utils.py oracle import failed (missing dep?): {exc}")
    for fn in ("_build_hadamard", "_rotate_weight", "_rotate_activation"):
        if not hasattr(module, fn):
            raise unittest.SkipTest(f"comfy-kitchen int8_utils lacks {fn} (update comfy-kitchen)")
    return module


def _load_ck_rowwise_quant():
    """Return comfy-kitchen's REAL eager ``quantize_int8_rowwise`` when the checkout
    imports as a package (preferred — catches upstream recipe drift), else None.

    The eager module is not standalone-loadable (it imports the CK registry), so
    this imports the whole package off sys.path. Requires a checkout >= 0.2.15 for
    the dtype-preserving division recipe (0.2.12-0.2.14 divide in fp32).
    """
    import sys

    root = _ck_root()
    if not (root / "comfy_kitchen" / "backends" / "eager" / "quantization.py").is_file():
        return None
    sys.path.insert(0, str(root))
    try:
        from comfy_kitchen.backends.eager import quantization as ck_quant  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - any import failure falls back to the inline oracle
        return None
    finally:
        sys.path.remove(str(root))
    if not hasattr(ck_quant, "quantize_int8_rowwise") or not hasattr(ck_quant, "_int8_scale_for_math"):
        return None  # predates the >= 0.2.15 recipe
    return ck_quant.quantize_int8_rowwise


def _oracle_quantize_int8_rowwise(torch, x):
    """Oracle for quantize_int8_rowwise: prefers the REAL comfy-kitchen eager
    function; falls back to a pinned inline transcription of the >= 0.2.15 recipe
    (scale = amax.float()/127 clamp 1e-30 fp32; q = round(x / scale.to(x.dtype,
    zeros->tiny)).clamp(-128,127).to(int8))."""
    real = _load_ck_rowwise_quant()
    if real is not None:
        return real(x.clone())
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    scale_math = scale.to(device=x.device, dtype=x.dtype)
    scale_math = torch.where(scale_math == 0, torch.full_like(scale_math, torch.finfo(x.dtype).tiny), scale_math)
    q = (x / scale_math).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


class TestExternalInt8TensorwiseParity(unittest.TestCase):
    def setUp(self):
        torch = _torch()
        if torch is None:
            self.skipTest("torch is required")
        self.torch = torch
        self.oracle = _load_ck_int8_utils()

    def test_hadamard_matches_comfy_kitchen(self):
        torch = self.torch
        from comfy_quants.formats.convrot import build_hadamard

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                ours = build_hadamard(256, device="cpu", dtype=dtype)
                theirs = self.oracle._build_hadamard(256, device="cpu", dtype=dtype)
                self.assertTrue(torch.equal(ours, theirs))

    def test_rotate_weight_matches_comfy_kitchen(self):
        torch = self.torch
        from comfy_quants.formats.convrot import build_hadamard, rotate_weight

        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                w = torch.randn(8, 512, dtype=dtype)
                h_ours = build_hadamard(256, device="cpu", dtype=dtype)
                h_theirs = self.oracle._build_hadamard(256, device="cpu", dtype=dtype)
                ours = rotate_weight(w, h_ours, 256)
                theirs = self.oracle._rotate_weight(w, h_theirs, 256)
                self.assertTrue(torch.equal(ours, theirs))

    def test_convrot_weight_quant_bit_faithful_to_comfy_kitchen(self):
        """Full path: our writer quant vs CK quantize_int8_convrot_weight composed
        from the loaded oracle rotation + the eager rowwise recipe."""
        torch = self.torch
        from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row

        torch.manual_seed(1)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                w = torch.randn(16, 1024, dtype=dtype)
                q_ours, scale_ours, rotated = _quantize_int8_tensorwise_per_row(w, convrot=True, group_size=256)
                self.assertTrue(rotated)
                h = self.oracle._build_hadamard(256, device="cpu", dtype=w.dtype)
                w_rot = self.oracle._rotate_weight(w, h, 256)
                q_theirs, scale_theirs = _oracle_quantize_int8_rowwise(torch, w_rot)
                self.assertTrue(torch.equal(scale_ours, scale_theirs))
                self.assertTrue(torch.equal(q_ours, q_theirs))

    def test_norotate_rowwise_quant_bit_faithful(self):
        torch = self.torch
        from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row

        torch.manual_seed(2)
        w = torch.randn(8, 192, dtype=torch.bfloat16)  # 192 % 256 != 0 -> no rotation
        q_ours, scale_ours, rotated = _quantize_int8_tensorwise_per_row(w, convrot=True, group_size=256)
        self.assertFalse(rotated)
        q_theirs, scale_theirs = _oracle_quantize_int8_rowwise(torch, w)
        self.assertTrue(torch.equal(scale_ours, scale_theirs))
        self.assertTrue(torch.equal(q_ours, q_theirs))


if __name__ == "__main__":
    unittest.main()
