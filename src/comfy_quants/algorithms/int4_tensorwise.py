"""INT4 tensorwise (int4_tensorwise, W4A4 + ConvRot) quantization planner.

Like ``int8_tensorwise``, the planner only assigns per-module quantize/keep
actions and stamps the target dtype; the int4/ConvRot math (and the per-layer
INT8 mixed-precision fallback) lives in the backend writer
(``backends/int4_tensorwise_model_export.py``), bit-faithful to comfy-kitchen's
eager ``quantize_int4_rowwise`` / ``quantize_int4_convrot_weight``.
"""

from __future__ import annotations

from comfy_quants.algorithms.base import AlgorithmPlanStep
from comfy_quants.algorithms.tensor_index import module_selected_by_policy
from comfy_quants.core.graph import ModelGraph
from comfy_quants.core.policy import QuantPolicy


class Int4TensorwiseAlgorithm:
    name = "int4_tensorwise"
    version = "0.1.0"

    def plan(self, graph: ModelGraph, policy: QuantPolicy) -> list[AlgorithmPlanStep]:
        steps: list[AlgorithmPlanStep] = []
        for index, module in enumerate(graph.modules):
            action = "quantize" if module_selected_by_policy(module, policy) else "keep_bf16"
            if not module.quantizable:
                action = module.default_action
            steps.append(AlgorithmPlanStep(
                step_id=f"{index:06d}",
                module_name=module.name,
                action=action,
                algorithm=self.name if action == "quantize" else "none",
                target_dtype=policy.target_dtype if action == "quantize" else "bf16",
            ))
        return steps


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_algorithm(Int4TensorwiseAlgorithm())
