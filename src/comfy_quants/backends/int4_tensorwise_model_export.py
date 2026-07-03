"""Safetensors INT4 tensorwise (W4A4 + ConvRot, mixed-precision) exporter.

OFFLINE producer of checkpoints for the comfy-kitchen ``TensorWiseINT4Layout``
runtime (W4A4; kitchen branch ``feat/int4-tensorwise-convrot``, stock-ComfyUI
``QUANT_ALGOS["int4_tensorwise"]`` entry in flight). Mirrors the
``int8_tensorwise`` writer structure — the format is its 4-bit downward
extension:

- ``comfy_quant`` marker carries ``{"format": "int4_tensorwise"}`` plus
  ``convrot``/``convrot_groupsize`` only when the layer was rotated (same byte
  encoding as the int8_tensorwise marker: default ``json.dumps``).
- Quantization is bit-faithful to comfy-kitchen eager ``quantize_int4_rowwise``
  / ``quantize_int4_convrot_weight``: the Hadamard is built and applied at the
  SOURCE WEIGHT dtype, the fp32 per-row scale is ``amax/7`` clamped to
  >= 1e-30, the weight is DIVIDED by the scale cast back to the weight dtype
  (zeros replaced by tiny), round-to-nearest-even, clamped to ``[-7, 7]``
  (kitchen int4 emission contract — -8 never emitted), then packed 2/byte LOW
  nibble first into an int8 container ``[out, in//2]``.

Mixed precision (the ConvRot paper's quality recipe — ~20% of layers are
functionally 4-bit-sensitive): layers whose module name matches an
``int8_fallback`` glob are written as **int8_tensorwise** instead (identical
tensors, marker and quant kernel as ``int8_tensorwise_model_export`` — the
helper is imported, never reimplemented). ComfyUI resolves ``QUANT_ALGOS`` per
layer from each marker, so both formats coexist in one checkpoint.

Emits NO ``input_scale`` (activations are rotated online and quantized
dynamically inside the comfy-kitchen kernels).
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# Reuse the FP8 writer's framework-agnostic helpers (source iteration, device
# resolution, config copy, metadata, progress) — shared infra, same as the other
# full-checkpoint writers.
from comfy_quants.backends.inference_model_export import (
    _emit_progress,
    _iter_source_files,
    _layer_name_from_weight,
    _metadata_value,
    _require_safetensors,
    _require_torch,
    _collect_header_metadata,
    _maybe_add_index_timestep_zero,
    _merged_output_metadata,
    _resolve_model_config_path,
    _resolve_torch_device,
    _shape_list,
    _source_name,
)

# The mixed-precision INT8 side must be BIT-IDENTICAL to the int8_tensorwise
# writer, so its quant helper and marker builder are imported, not copied.
from comfy_quants.backends.int8_tensorwise_model_export import (
    _quantize_int8_tensorwise_per_row,
)
from comfy_quants.backends.safetensors_source import SafetensorsTensorSource
from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.convrot import CONVROT_GROUP_SIZE, build_hadamard, rotate_weight
from comfy_quants.formats.int4_common import pack_signed_int4_pairs
from comfy_quants.formats.int4_tensorwise import (
    INT4_TENSORWISE_FORMAT_NAME,
    int4_tensorwise_checkpoint_quant_config,
)
from comfy_quants.formats.int8_tensorwise import int8_tensorwise_checkpoint_quant_config
from comfy_quants.utils.hashing import hash_file


@dataclass
class Int4TensorwiseCheckpointExportReport:
    """Summary of a full int4_tensorwise (mixed-capable) checkpoint export."""

    source_checkpoint: str
    output_checkpoint: str
    quantized_tensor_count: int
    copied_tensor_count: int
    output_tensor_count: int
    schema_version: str = "int4_tensorwise_checkpoint_export_report.v1"
    status: str = "model_written"
    source_format: str = "safetensors"
    target_format: str = "safetensors"
    requested_device: str = "auto"
    execution_device: str = "cpu"
    output_tensor_device: str = "cpu"
    artifact_target: str = "comfyui_diffusion_model"
    target_dtype: str = INT4_TENSORWISE_FORMAT_NAME
    quant_storage_dtype: str = "int8"  # container: 2 signed int4 codes / byte
    weight_packing: str = "int4_pairs_low_nibble_first"
    scale_dtype: str = "fp32"
    scale_granularity: str = "per_channel"
    scale_axis: str | int | None = "out_features"
    convrot: bool = True
    convrot_groupsize: int = CONVROT_GROUP_SIZE
    rotated_tensor_count: int = 0
    nonrotated_tensor_count: int = 0
    int4_tensor_count: int = 0
    int8_fallback_tensor_count: int = 0
    int8_fallback_globs: list[str] = field(default_factory=list)
    int8_fallback_tensors: list[str] = field(default_factory=list)
    source_layout: str = "single_file"
    source_tensor_count: int = 0
    source_file_count: int = 0
    selected_source_files: dict[str, int] = field(default_factory=dict)
    missing_tensor_count: int = 0
    missing_tensors: list[str] = field(default_factory=list)
    quant_metadata_tensor_count: int = 0
    scale_tensor_count: int = 0
    output_bytes: int = 0
    output_hash: str = ""
    output_hash_state: str = "not_requested"
    config_path: str | None = None
    cuda_max_memory_allocated_bytes: int | None = None
    cuda_max_memory_reserved_bytes: int | None = None
    dtype_counts: dict[str, int] = field(default_factory=dict)
    written_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_int4_tensorwise_target_dtype(tensor_index: dict[str, Any], target_dtype: str | None = None) -> str:
    index_dtype = tensor_index.get("format", {}).get("name")
    resolved = target_dtype or index_dtype
    if resolved != INT4_TENSORWISE_FORMAT_NAME:
        raise PayloadWriteError(f"tensor index is not an {INT4_TENSORWISE_FORMAT_NAME} format: {resolved}")
    if index_dtype != INT4_TENSORWISE_FORMAT_NAME:
        raise PayloadWriteError(f"tensor index format {index_dtype} does not match requested target dtype {resolved}")
    return INT4_TENSORWISE_FORMAT_NAME


def _check_selected_rows(tensor_index: dict[str, Any], *, target_dtype: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in list(tensor_index.get("tensors") or []):
        if row.get("quant_dtype") != target_dtype:
            raise PayloadWriteError(f"unsupported quant dtype for int4_tensorwise export: {row.get('quant_dtype')}")
        if row.get("storage_dtype") != "int8":
            raise PayloadWriteError(f"unsupported storage dtype for int4_tensorwise export: {row.get('storage_dtype')}")
        name = _source_name(row)
        if name in selected:
            raise PayloadWriteError(f"duplicate selected tensor in export plan: {name}")
        _layer_name_from_weight(name)
        selected[name] = row
    return selected


def _marker_tensor(conf: dict[str, Any], *, device: str):
    """Encode the comfy_quant marker exactly like stock ComfyUI's save path
    (``ops.py``): default ``json.dumps`` separators + insertion order."""
    torch = _require_torch()
    data = json.dumps(conf).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8, device=device)


def _matches_int8_fallback(layer_name: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(layer_name, pattern) for pattern in globs)


def _quantize_int4_tensorwise_per_row(tensor, *, convrot: bool, group_size: int) -> tuple[Any, Any, bool]:
    """Symmetric per-output-channel packed INT4, optionally ConvRot-rotated first.

    Returns ``(packed_int4_weight [out, in//2], fp32_scale [out, 1], rotated)``.
    Bit-faithful to comfy-kitchen eager ``quantize_int4_convrot_weight`` /
    ``quantize_int4_rowwise``: Hadamard built and applied at the weight dtype,
    ``scale = amax.float()/7`` clamped to >= 1e-30, quantization by DIVISION with
    the scale cast back to the weight dtype (zeros -> dtype tiny),
    round-half-to-even, clamp [-7, 7], pack low-nibble-first. ConvRot gated on
    ``in % group_size == 0`` (ineligible layers are quantized unrotated and the
    marker omits the convrot keys).
    """
    torch = _require_torch()
    w = tensor.detach()
    if w.dim() != 2:
        raise PayloadWriteError("int4_tensorwise export requires a rank-2 weight tensor")
    if w.shape[1] % 2 != 0:
        raise PayloadWriteError(f"int4_tensorwise export requires even in_features, got {int(w.shape[1])}")
    rotated = False
    if convrot and w.shape[1] % group_size == 0:
        hadamard = build_hadamard(group_size, device=w.device, dtype=w.dtype)
        w = rotate_weight(w, hadamard, group_size)
        rotated = True
    abs_max = w.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 7.0).clamp(min=1e-30)
    scale_math = scale.to(device=w.device, dtype=w.dtype)
    scale_math = torch.where(
        scale_math == 0, torch.full_like(scale_math, torch.finfo(w.dtype).tiny), scale_math
    )
    codes = (w / scale_math).round_().clamp_(-7.0, 7.0).to(torch.int8)
    packed = pack_signed_int4_pairs(codes, validate=False)
    return packed.contiguous(), scale.to(torch.float32).contiguous(), rotated


def write_int4_tensorwise_inference_checkpoint_from_safetensors(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    tensor_index: dict[str, Any],
    target_dtype: str | None = None,
    convrot: bool = True,
    convrot_groupsize: int = CONVROT_GROUP_SIZE,
    int8_fallback: list[str] | None = None,
    device: str = "auto",
    strict: bool = True,
    config_source: str | Path | None = None,
    copy_config: bool = True,
    hash_output: bool = False,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Int4TensorwiseCheckpointExportReport:
    """Write a full checkpoint with selected weights stored as packed INT4 tensors.

    Layers whose module name matches an ``int8_fallback`` glob are written as
    int8_tensorwise instead (mixed-precision quality recipe) — identical bytes
    to the dedicated int8_tensorwise writer for those layers.
    """
    safe_open, save_file = _require_safetensors()
    torch = _require_torch()
    resolved_target_dtype = _resolve_int4_tensorwise_target_dtype(tensor_index, target_dtype)
    fallback_globs = [str(g) for g in (int8_fallback or [])]

    requested_device = str(device or "auto")
    execution_device_obj = _resolve_torch_device(requested_device)
    execution_device = str(execution_device_obj)
    cuda_peak_allocated: int | None = None
    cuda_peak_reserved: int | None = None
    if execution_device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device_obj)
    source = SafetensorsTensorSource.from_path(source_checkpoint)
    output_path = Path(output_checkpoint).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = _check_selected_rows(tensor_index, target_dtype=resolved_target_dtype)
    selected_names = list(selected)
    missing = source.missing_tensors(selected_names)
    if missing and strict:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise PayloadWriteError(f"source checkpoint is missing selected tensors: {preview}{suffix}")

    output_tensors: dict[str, Any] = {}
    dtype_counts: dict[str, int] = {}
    copied = 0
    quantized = 0
    rotated_count = 0
    int4_count = 0
    int8_count = 0
    int8_layers: list[str] = []
    source_files = _iter_source_files(source)
    output_resolved = output_path.resolve(strict=False)
    source_file_paths = {path.resolve(strict=False) for path, _names in source_files}
    if output_resolved in source_file_paths:
        raise PayloadWriteError(f"output checkpoint must not overwrite a source tensor file: {output_path}")

    _emit_progress(
        progress,
        stage="prepare",
        target_dtype=resolved_target_dtype,
        requested_device=requested_device,
        execution_device=execution_device,
        source_file_count=len(source_files),
        selected_tensor_count=len(selected_names),
        convrot=bool(convrot),
        int8_fallback_globs=fallback_globs,
    )
    source_header_metadata: dict[str, str] = {}
    for source_file_index, (source_file, tensor_names) in enumerate(source_files, start=1):
        if not source_file.is_file():
            raise PayloadWriteError(f"source tensor file is missing: {source_file}")
        _emit_progress(
            progress,
            stage="read_source_file",
            source_file=str(source_file),
            source_file_index=source_file_index,
            source_file_count=len(source_files),
            tensor_count=len(tensor_names),
        )
        with safe_open(str(source_file), framework="pt", device="cpu") as handle:
            _collect_header_metadata(handle, source_header_metadata)
            available = set(handle.keys())
            for tensor_name in tensor_names:
                if tensor_name not in available:
                    raise PayloadWriteError(f"safetensors index maps {tensor_name} to {source_file}, but the tensor is absent from that file")
                tensor = handle.get_tensor(tensor_name)
                row = selected.get(tensor_name)
                if row is None:
                    stored = tensor.detach().contiguous()
                    output_tensors[tensor_name] = stored
                    dtype = str(stored.dtype).replace("torch.", "")
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                    copied += 1
                    continue

                expected_shape = [int(dim) for dim in row.get("shape") or []]
                if _shape_list(tensor) != expected_shape:
                    raise PayloadWriteError(f"source tensor shape mismatch for {tensor_name}: expected {expected_shape}, got {_shape_list(tensor)}")

                if execution_device_obj.type == "cuda":
                    tensor_for_quant = tensor.to(device=execution_device_obj, non_blocking=True)
                else:
                    tensor_for_quant = tensor
                layer = _layer_name_from_weight(tensor_name)
                use_int8 = _matches_int8_fallback(layer, fallback_globs)
                if use_int8:
                    qweight, scale, rotated = _quantize_int8_tensorwise_per_row(
                        tensor_for_quant, convrot=convrot, group_size=convrot_groupsize
                    )
                    marker = int8_tensorwise_checkpoint_quant_config(
                        convrot=rotated, convrot_groupsize=convrot_groupsize
                    )
                    int8_count += 1
                    int8_layers.append(layer)
                else:
                    qweight, scale, rotated = _quantize_int4_tensorwise_per_row(
                        tensor_for_quant, convrot=convrot, group_size=convrot_groupsize
                    )
                    marker = int4_tensorwise_checkpoint_quant_config(
                        convrot=rotated, convrot_groupsize=convrot_groupsize
                    )
                    int4_count += 1
                output_tensors[tensor_name] = qweight.detach().to(device="cpu").contiguous()
                output_tensors[f"{layer}.weight_scale"] = scale.detach().to(device="cpu").contiguous()
                output_tensors[f"{layer}.comfy_quant"] = _marker_tensor(marker, device="cpu")
                dtype_counts["int8"] = dtype_counts.get("int8", 0) + 1
                dtype_counts["float32"] = dtype_counts.get("float32", 0) + 1
                dtype_counts["uint8"] = dtype_counts.get("uint8", 0) + 1
                quantized += 1
                if rotated:
                    rotated_count += 1
                _emit_progress(
                    progress,
                    stage="quantize_tensor",
                    target_dtype="int8_tensorwise" if use_int8 else resolved_target_dtype,
                    tensor_name=tensor_name,
                    quantized_tensor_count=quantized,
                    selected_tensor_count=len(selected_names),
                    convrot=rotated,
                    int8_fallback=use_int8,
                    execution_device=execution_device,
                )
                del qweight, scale
                if execution_device_obj.type == "cuda":
                    del tensor_for_quant

    _maybe_add_index_timestep_zero(output_tensors, dtype_counts, tensor_index, metadata)

    output_metadata = _merged_output_metadata(
        source_header_metadata,
        metadata,
        {
            "artifact_target": "comfyui_diffusion_model",
            # Family-agnostic (int8_tensorwise precedent), NOT the older writers'
            # hardcoded qwen_image_ prefix.
            "artifact_contract": "int4_tensorwise_inference_checkpoint.v1",
            "target_dtype": resolved_target_dtype,
            "quant_storage_dtype": "int8",
            "weight_packing": "int4_pairs_low_nibble_first",
            "scale_granularity": "per_channel",
            "scale_axis": "out_features",
            "convrot": bool(convrot),
            "convrot_groupsize": int(convrot_groupsize),
            "quantized_tensor_count": quantized,
            "int4_tensor_count": int4_count,
            "int8_fallback_tensor_count": int8_count,
        },
    )
    if execution_device_obj.type == "cuda":
        torch.cuda.synchronize(execution_device_obj)
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(execution_device_obj))
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved(execution_device_obj))
        torch.cuda.empty_cache()
    _emit_progress(
        progress,
        stage="save_checkpoint",
        output_checkpoint=str(output_path),
        output_tensor_count=len(output_tensors),
        output_tensor_device="cpu",
    )
    save_file(output_tensors, str(output_path), metadata={str(k): _metadata_value(v) for k, v in output_metadata.items()})

    copied_config_path: str | None = None
    if copy_config:
        config_path = _resolve_model_config_path(source, config_source)
        if config_path is not None:
            destination = output_path.parent / "config.json"
            if config_path.resolve() != destination.resolve():
                import shutil

                shutil.copy2(config_path, destination)
            copied_config_path = str(destination)

    output_hash = ""
    output_hash_state = "not_requested"
    if hash_output:
        _emit_progress(progress, stage="hash_checkpoint", output_checkpoint=str(output_path))
        output_hash = hash_file(output_path)
        output_hash_state = "written"
    output_bytes = output_path.stat().st_size
    written_files = [
        {
            "path": str(output_path),
            "kind": "int4_tensorwise_inference_checkpoint",
            "state": "written",
            "tensor_count": len(output_tensors),
            "bytes": output_bytes,
            "hash": output_hash,
            "hash_state": output_hash_state,
        }
    ]
    if copied_config_path is not None:
        config_file = Path(copied_config_path)
        written_files.append(
            {
                "path": str(config_file),
                "kind": "model_config",
                "state": "copied",
                "bytes": config_file.stat().st_size,
                "hash": hash_file(config_file),
            }
        )

    return Int4TensorwiseCheckpointExportReport(
        source_checkpoint=str(source.source_path),
        output_checkpoint=str(output_path),
        quantized_tensor_count=quantized,
        copied_tensor_count=copied,
        output_tensor_count=len(output_tensors),
        requested_device=requested_device,
        execution_device=execution_device,
        output_tensor_device="cpu",
        convrot=bool(convrot),
        convrot_groupsize=int(convrot_groupsize),
        rotated_tensor_count=rotated_count,
        nonrotated_tensor_count=quantized - rotated_count,
        int4_tensor_count=int4_count,
        int8_fallback_tensor_count=int8_count,
        int8_fallback_globs=fallback_globs,
        int8_fallback_tensors=int8_layers,
        source_layout=source.layout,
        source_tensor_count=len(source.file_map),
        source_file_count=len(set(source.file_map.values())),
        selected_source_files=source.selected_file_counts(selected_names),
        missing_tensor_count=len(missing),
        missing_tensors=missing,
        quant_metadata_tensor_count=quantized,
        scale_tensor_count=quantized,
        output_bytes=output_bytes,
        output_hash=output_hash,
        output_hash_state=output_hash_state,
        config_path=copied_config_path,
        cuda_max_memory_allocated_bytes=cuda_peak_allocated,
        cuda_max_memory_reserved_bytes=cuda_peak_reserved,
        dtype_counts=dict(sorted(dtype_counts.items())),
        written_files=written_files,
    )
