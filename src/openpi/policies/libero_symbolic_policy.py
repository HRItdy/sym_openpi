"""
LIBERO policy with symbolic operator conditioning.

This policy extends the standard LiberoInputs/LiberoOutputs to support:
  - Symbolic prompts: "Task: {desc}. Now: {op}. State: {prec}."
  - An augmented 8-dim action space: 7 robot DoF + 1 segment_done flag
    - segment_done = 0.0 for all frames within a segment
    - segment_done = 1.0 at the last frame of each operator segment
  - LiberoSymbolicOutputs exposes a `segment_done` field (shape: [action_horizon])
    that the inference script polls to decide when to advance to the next operator.

Text input format
-----------------
Build the prompt using `build_symbolic_prompt()`:

    "Task: put the black bowl on top of the cabinet. "
    "Now: pick up the black bowl. "
    "State: Empty[gripper]."

Effects are NOT included in the current segment's prompt; they are naturally
encoded as the preconditions of the next segment and appear there instead.

Operator termination
--------------------
During inference, `LiberoSymbolicOutputs` returns `segment_done` alongside
`actions`. Poll `segment_done.mean() > threshold` (or check whether the mean of
the last few predicted steps exceeds the threshold) to decide when to switch the
prompt to the next operator. See `examples/libero/main_symbolic.py` for a
complete inference loop.
"""

import dataclasses
import math

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def build_symbolic_prompt(
    task_description: str,
    operator_description: str,
    preconditions: list[str],
) -> str:
    """Build a symbolic prompt for one operator segment.

    Args:
        task_description: High-level task language ("put the black bowl on top
            of the cabinet").
        operator_description: Natural-language description of the current
            operator ("pick up the black bowl").
        preconditions: List of predicate strings that hold before this operator
            executes (["Empty[gripper]"]).

    Returns:
        A single prompt string that fits within the model's token budget.
    """
    prec_str = "; ".join(preconditions) if preconditions else "none"
    return (
        f"Task: {task_description}. "
        f"Now: {operator_description}. "
        f"State: {prec_str}."
    )


# ---------------------------------------------------------------------------
# Image helper (shared with the original libero_policy)
# ---------------------------------------------------------------------------


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# ---------------------------------------------------------------------------
# Quat → axis-angle (used by data conversion script)
# ---------------------------------------------------------------------------


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to axis-angle representation."""
    if quat[3] > 1.0:
        quat = quat.copy()
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat = quat.copy()
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ---------------------------------------------------------------------------
# Input / output transforms
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LiberoSymbolicInputs(transforms.DataTransformFn):
    """Convert observations to model input format.

    Identical to LiberoInputs but documents that `actions` are expected to be
    8-dimensional (7 robot DoF + 1 segment_done flag) during training.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Only mask the padding image for pi0; pi0-FAST sees all images.
                "right_wrist_0_rgb": (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                ),
            },
        }

        # Actions (training only): shape (action_horizon, 8) after data loader
        # stacking. The 8th dimension is the segment_done flag.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoSymbolicOutputs(transforms.DataTransformFn):
    """Convert model outputs back to environment format.

    Returns:
        actions       – shape (action_horizon, 7), the raw robot DoF actions.
        segment_done  – shape (action_horizon,), predicted operator-completion
                        signal. Use ``segment_done.mean() > threshold`` (or a
                        similar aggregation) to decide when to advance to the
                        next operator during inference.
    """

    def __call__(self, data: dict) -> dict:
        actions_full = np.asarray(data["actions"])   # (action_horizon, model_action_dim)
        robot_actions = actions_full[:, :7]           # (action_horizon, 7)
        segment_done = actions_full[:, 7]             # (action_horizon,)
        return {
            "actions": robot_actions,
            "segment_done": segment_done,
        }


# ---------------------------------------------------------------------------
# Example input (for policy server testing)
# ---------------------------------------------------------------------------


def make_libero_symbolic_example() -> dict:
    """Random input example compatible with LiberoSymbolicInputs."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": build_symbolic_prompt(
            task_description="do something",
            operator_description="pick up the object",
            preconditions=["Empty[gripper]"],
        ),
    }
