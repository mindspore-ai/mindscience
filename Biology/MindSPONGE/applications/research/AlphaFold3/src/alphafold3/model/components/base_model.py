"""Defines interface of a BaseModel."""

from collections.abc import Mapping
import dataclasses
from typing import Any, TypeAlias
from alphafold3 import structure
import numpy as np
import mindspore as ms

ModelResult: TypeAlias = Mapping[str, Any]
ScalarNumberOrArray: TypeAlias = Mapping[str, float | int | np.ndarray]

# Eval result will contain scalars (e.g. metrics or losses), selected from the
# forward pass outputs or computed in the online evaluation; np.ndarrays or
# jax.Arrays generated from the forward pass outputs (e.g. distogram expected
# distances) or batch inputs; protein structures (predicted and ground-truth).
EvalResultValue: TypeAlias = (
    float | int | np.ndarray | ms.Tensor | structure.Structure
)
# Eval result may be None for some metrics if they are not computable.
EvalResults: TypeAlias = Mapping[str, EvalResultValue | None]
# Interface metrics are all floats or None.
InterfaceMetrics: TypeAlias = Mapping[str, float | None]
# Interface results are a mapping from interface name to mappings from score
# type to metric value.
InterfaceResults: TypeAlias = Mapping[str, Mapping[str, InterfaceMetrics]]
# Eval output consists of full eval results and a dict of interface metrics.
EvalOutput: TypeAlias = tuple[EvalResults, InterfaceResults]

# Signature for `apply` method of hk.transform_with_state called on a BaseModel.
# ForwardFn: TypeAlias = Callable[
#     [hk.Params, hk.State, jax.Array, features.BatchDict],
#     tuple[ModelResult, hk.State],
# ]


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    """Postprocessed model result."""

    # Predicted protein structure.
    predicted_structure: structure.Structure = dataclasses.field()
    # Useful numerical data (scalars or arrays) to be saved at inference time.
    numerical_data: ScalarNumberOrArray = dataclasses.field(
        default_factory=dict)
    # Smaller numerical data (usually scalar) to be saved as inference metadata.
    metadata: ScalarNumberOrArray = dataclasses.field(default_factory=dict)
    # Additional dict for debugging, e.g. raw outputs of a model forward pass.
    debug_outputs: ModelResult | None = dataclasses.field(default_factory=dict)
    # Model identifier.
    model_id: bytes = b''
