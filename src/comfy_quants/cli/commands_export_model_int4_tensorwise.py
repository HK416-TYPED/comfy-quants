"""export-model-int4-tensorwise command (W4A4 + ConvRot, mixed int8 fallback)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from comfy_quants.algorithms.registry import get_algorithm
from comfy_quants.algorithms.tensor_index import TensorIndexOptions, build_quant_tensor_index
from comfy_quants.backends.int4_tensorwise_model_export import (
    write_int4_tensorwise_inference_checkpoint_from_safetensors,
)
from comfy_quants.cli.commands_export_model import _resolve_source
from comfy_quants.cli.common import local_path, print_json
from comfy_quants.core.config import load_quant_config
from comfy_quants.core.errors import ConfigurationError
from comfy_quants.formats.convrot import CONVROT_GROUP_SIZE
from comfy_quants.formats.int4_tensorwise import INT4_TENSORWISE_FORMAT_NAME
from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.registry import get_adapter
from comfy_quants.utils.jsonio import write_json

_DEFAULT_CHECKPOINT_NAME = f"diffusion_pytorch_model.{INT4_TENSORWISE_FORMAT_NAME}.safetensors"


def register(subparsers):
    parser = subparsers.add_parser(
        "export-model-int4-tensorwise",
        help="Export a W4A4 int4 (+ConvRot) inference checkpoint with optional per-layer int8 fallback (comfy-kitchen TensorWiseINT4Layout runtime)",
    )
    parser.add_argument("--config", required=True, help="Quantization YAML/JSON config")
    parser.add_argument("--source", default=None, help="Local safetensors file, index JSON, or indexed directory")
    parser.add_argument("--out", required=True, help="Output .safetensors path or output directory")
    parser.add_argument("--device", default="auto", help="Torch device for tensor conversion; auto uses cuda:0 when available and falls back to cpu")
    parser.add_argument("--convrot", action=argparse.BooleanOptionalAction, default=True, help="Apply ConvRot (regular-Hadamard) weight rotation before INT4/INT8 quantization")
    parser.add_argument("--convrot-groupsize", type=int, default=CONVROT_GROUP_SIZE, help="ConvRot group size (power of four; default 256)")
    parser.add_argument(
        "--int8-fallback",
        action="append",
        default=None,
        metavar="GLOB",
        help="Module-name glob whose layers are written as int8_tensorwise instead of int4 (repeatable; overrides the config's quant.modules.int8_fallback list). The ConvRot paper's quality recipe keeps ~20%% of layers at W8A8.",
    )
    parser.add_argument("--hash-output", action="store_true", help="Compute a SHA256 hash after writing the checkpoint")
    parser.add_argument("--no-progress", action="store_true", help="Do not write export progress events to stderr")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=run)


def _resolve_output(out: str | Path) -> tuple[Path, Path]:
    path = local_path(out)
    if path.exists() and path.is_dir():
        checkpoint = path / _DEFAULT_CHECKPOINT_NAME
        report = path / "export_report.json"
    elif path.suffix == ".safetensors":
        checkpoint = path
        report = path.with_suffix(".export_report.json")
    else:
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = path / _DEFAULT_CHECKPOINT_NAME
        report = path / "export_report.json"
    return checkpoint, report


def _build_tensor_index(cfg, source: Path) -> dict:
    adapter = get_adapter(cfg.model.family)
    model_source = ModelSource(
        family=cfg.model.family,
        model_id=str(source),
        revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        source="local",
    )
    _inspection, graph = adapter.inspect(model_source)
    policy = adapter.default_policy(cfg.quant.target_dtype)
    policy.algorithm = cfg.quant.algorithm
    policy.include = cfg.quant.modules.get("include", policy.include)
    policy.exclude = cfg.quant.modules.get("exclude", policy.exclude)
    algorithm = get_algorithm(cfg.quant.algorithm)
    algorithm.plan(graph, policy)
    # int4_tensorwise weights are per-output-channel regardless of the config
    # scale section (same constraint as int8_tensorwise / ConvRot).
    return build_quant_tensor_index(
        graph,
        policy,
        TensorIndexOptions(
            algorithm=cfg.quant.algorithm,
            algorithm_version=getattr(algorithm, "version", "0.1.0"),
            target_dtype=cfg.quant.target_dtype,
            scale_granularity="per_channel",
            scale_axis="out_features",
            scale_method=cfg.quant.scale.method,
            rounding=cfg.quant.rounding,
            compatibility_level=cfg.artifact.compatibility_target,
            artifact_state="model_export",
            tensor_payload_state="written_in_checkpoint",
        ),
    )


def _resolve_int8_fallback(cfg, args) -> list[str]:
    if args.int8_fallback is not None:
        return [str(g) for g in args.int8_fallback]
    config_globs = cfg.quant.modules.get("int8_fallback") or []
    return [str(g) for g in config_globs]


def run(args) -> int:
    cfg = load_quant_config(args.config)
    if cfg.quant.target_dtype != INT4_TENSORWISE_FORMAT_NAME:
        raise ConfigurationError(
            f"export-model-int4-tensorwise requires quant.target_dtype '{INT4_TENSORWISE_FORMAT_NAME}', got {cfg.quant.target_dtype}"
        )

    source = _resolve_source(cfg, args.config, args.source)
    checkpoint, report_path = _resolve_output(args.out)
    tensor_index = _build_tensor_index(cfg, source)
    int8_fallback = _resolve_int8_fallback(cfg, args)

    def progress(event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)

    report = write_int4_tensorwise_inference_checkpoint_from_safetensors(
        source_checkpoint=source,
        output_checkpoint=checkpoint,
        tensor_index=tensor_index,
        target_dtype=cfg.quant.target_dtype,
        convrot=args.convrot,
        convrot_groupsize=args.convrot_groupsize,
        int8_fallback=int8_fallback,
        device=args.device,
        strict=True,
        config_source=source,
        hash_output=args.hash_output,
        metadata={
            "model_family": cfg.model.family,
            "model_id": cfg.model.model_id,
            "project": cfg.project.name,
        },
        progress=None if args.no_progress else progress,
    ).to_dict()
    report["tensor_index"] = {
        "schema_version": tensor_index.get("schema_version"),
        "selection": tensor_index.get("selection"),
        "format": tensor_index.get("format"),
    }
    write_json(report_path, report)

    result = {
        "status": report["status"],
        "output_checkpoint": report["output_checkpoint"],
        "report": str(report_path),
        "quantized_tensor_count": report["quantized_tensor_count"],
        "int4_tensor_count": report["int4_tensor_count"],
        "int8_fallback_tensor_count": report["int8_fallback_tensor_count"],
        "rotated_tensor_count": report["rotated_tensor_count"],
        "copied_tensor_count": report["copied_tensor_count"],
        "output_tensor_count": report["output_tensor_count"],
        "convrot": report["convrot"],
        "output_bytes": report["output_bytes"],
        "output_hash": report["output_hash"],
        "output_hash_state": report["output_hash_state"],
        "requested_device": report["requested_device"],
        "execution_device": report["execution_device"],
    }
    if args.json:
        print_json(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
