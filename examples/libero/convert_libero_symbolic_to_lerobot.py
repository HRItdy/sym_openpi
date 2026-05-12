"""
Convert segmented LIBERO data with symbolic annotations to LeRobot format.

Input
-----
  --hdf5_dir   Directory containing per-task LIBERO HDF5 files
               (e.g. KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet.hdf5)
  --json_dir   Directory containing the merged segmentation JSON files produced
               by your annotation pipeline
               (e.g. KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet_merged.json)

Output
------
  A LeRobot dataset (saved to $HF_LEROBOT_HOME/<repo_name>) with:
    image        : (256, 256, 3) uint8, agent-view
    wrist_image  : (256, 256, 3) uint8, wrist camera
    state        : (8,)  float32  [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    actions      : (8,)  float32  [7 robot DoF delta-actions + segment_done flag]
    task         : str   symbolic prompt for the active operator

Symbolic prompt format
-----------------------
  "Task: {task_description}. Now: {operator_description}. State: {precondition1}; ..."

  Effects are NOT included in the current segment's prompt; they are naturally
  encoded as the preconditions of the next segment.

Action dimensions
-----------------
  dims 0-6  : original 7-DoF robot actions (already delta-encoded in LIBERO)
  dim  7    : segment_done  (0.0 within a segment, 1.0 at the last frame)

Usage
-----
  uv run examples/libero/convert_libero_symbolic_to_lerobot.py \\
      --hdf5_dir /path/to/libero_hdf5 \\
      --json_dir  /path/to/segmentation_json \\
      --repo_name your_hf_username/libero_symbolic

  Optional flags:
      --push_to_hub     push dataset to Hugging Face Hub
      --fps 10          dataset frame rate (default: 10)

Notes on HDF5 structure
------------------------
  This script expects the standard LIBERO HDF5 layout:

    data/
      demo_0/
        obs/
          agentview_image          : (T, H, W, 3) uint8  *or* (T, 3, H, W)
          robot0_eye_in_hand_image : (T, H, W, 3) uint8  *or* (T, 3, H, W)
          robot0_eef_pos           : (T, 3)  float64
          robot0_eef_quat          : (T, 4)  float64   [x, y, z, w]
          robot0_gripper_qpos      : (T, 2)  float64
        actions                    : (T, 7)  float64

  If your HDF5 uses different key names, adjust the _LIBERO_HDF5_KEYS dict below.

  The LIBERO-90/100 datasets use this layout instead:

    data/
      demo_0/
        obs/
          agentview_rgb            : (T, H, W, 3) uint8
          eye_in_hand_rgb          : (T, H, W, 3) uint8
          ee_pos                   : (T, 3)  float64
          ee_ori                   : (T, 3)  float64  axis-angle (NOT quaternion)
          gripper_states           : (T, 2)  float64
        actions                    : (T, 7)  float64
"""

import json
import pathlib
import shutil
from dataclasses import dataclass

import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ---------------------------------------------------------------------------
# Key mapping – edit here if your HDF5 uses different observation names
# ---------------------------------------------------------------------------
_LIBERO_HDF5_KEYS = {
    "agentview_image": "obs/agentview_rgb",
    "wrist_image": "obs/eye_in_hand_rgb",
    "eef_pos": "obs/ee_pos",
    "eef_ori": "obs/ee_ori",       # axis-angle (T, 3), not quaternion
    "gripper_qpos": "obs/gripper_states",
    "actions": "actions",
}

DEFAULT_REPO_NAME = "your_hf_username/libero_symbolic"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _parse_image(img: np.ndarray) -> np.ndarray:
    """Ensure image is (H, W, 3) uint8.

    LIBERO HDF5 files may store images as (T, 3, H, W) or (T, H, W, 3).
    This function operates on a *single* frame, not the full (T, ...) array.
    """
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if not np.issubdtype(img.dtype, np.uint8):
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_symbolic_prompt(
    task_description: str,
    operator_description: str,
    preconditions: list[str],
) -> str:
    """Return the symbolic prompt for one operator segment."""
    prec_str = "; ".join(preconditions) if preconditions else "none"
    return (
        f"Task: {task_description}. "
        f"Now: {operator_description}. "
        f"State: {prec_str}."
    )


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------


@dataclass
class ConvertArgs:
    hdf5_dir: str
    json_dir: str
    repo_name: str = DEFAULT_REPO_NAME
    push_to_hub: bool = False
    fps: int = 10


def main(args: ConvertArgs) -> None:
    hdf5_dir = pathlib.Path(args.hdf5_dir)
    json_dir = pathlib.Path(args.json_dir)

    output_path = HF_LEROBOT_HOME / args.repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            # 7 robot DoF + 1 segment_done flag
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    json_files = sorted(json_dir.glob("*_merged.json"))
    if not json_files:
        # Also accept non-merged filenames
        json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON annotation files found in {json_dir}")

    for json_path in json_files:
        print(f"\nProcessing {json_path.name} …")
        with open(json_path) as fh:
            annotation = json.load(fh)

        task_name: str = annotation["task_name"]
        task_description: str = annotation["canonical_operators"]["language"]

        # Locate the matching HDF5 file.
        # Try both naming conventions: task_name_demo.hdf5 and task_name.hdf5
        hdf5_path = hdf5_dir / f"{task_name}_demo.hdf5"
        if not hdf5_path.exists():
            hdf5_path = hdf5_dir / f"{task_name}.hdf5"
        if not hdf5_path.exists():
            print(f"  [SKIP] HDF5 not found for {task_name}")
            continue

        with h5py.File(hdf5_path, "r") as hf:
            demo_root = hf["data"]

            for demo_name, demo_segments in annotation["segments"].items():
                if demo_name not in demo_root:
                    print(f"  [SKIP] {demo_name} not in HDF5")
                    continue

                d = demo_root[demo_name]

                # ── Observations ──────────────────────────────────────────
                raw_images = d[_LIBERO_HDF5_KEYS["agentview_image"]][:]       # (T, …)
                raw_wrist = d[_LIBERO_HDF5_KEYS["wrist_image"]][:]            # (T, …)
                eef_pos = d[_LIBERO_HDF5_KEYS["eef_pos"]][:].astype(np.float32)      # (T, 3)
                eef_ori = d[_LIBERO_HDF5_KEYS["eef_ori"]][:].astype(np.float32)      # (T, 3)
                gripper = d[_LIBERO_HDF5_KEYS["gripper_qpos"]][:].astype(np.float32) # (T, 2)
                actions_raw = d[_LIBERO_HDF5_KEYS["actions"]][:].astype(np.float32)  # (T, 7)

                states = np.concatenate([eef_pos, eef_ori, gripper], axis=-1)         # (T, 8)

                num_frames = len(actions_raw)

                # ── Per-frame annotation arrays ───────────────────────────
                segment_done = np.zeros(num_frames, dtype=np.float32)
                frame_prompts: list[str] = [""] * num_frames

                for seg in demo_segments:
                    start: int = seg["start_frame"]
                    end: int = seg["end_frame"]
                    ann = seg["annotation"]

                    prompt = build_symbolic_prompt(
                        task_description=task_description,
                        operator_description=ann["description"],
                        preconditions=ann["preconditions"],
                    )
                    for t in range(start, min(end + 1, num_frames)):
                        frame_prompts[t] = prompt

                    # Mark the last frame of this segment as done.
                    last_frame = min(end, num_frames - 1)
                    segment_done[last_frame] = 1.0

                # Fall back to task description for any un-annotated frames.
                fallback_prompt = f"Task: {task_description}."
                for t in range(num_frames):
                    if not frame_prompts[t]:
                        frame_prompts[t] = fallback_prompt

                # ── Write frames ──────────────────────────────────────────
                for t in range(num_frames):
                    # IMPORTANT: rotate 180° to match LIBERO training convention.
                    img = _parse_image(raw_images[t])
                    img = np.ascontiguousarray(img[::-1, ::-1])

                    wrist = _parse_image(raw_wrist[t])
                    wrist = np.ascontiguousarray(wrist[::-1, ::-1])

                    aug_action = np.concatenate(
                        [actions_raw[t], [segment_done[t]]], axis=0
                    )  # (8,)

                    dataset.add_frame(
                        {
                            "image": img,
                            "wrist_image": wrist,
                            "state": states[t],
                            "actions": aug_action,
                            "task": frame_prompts[t],
                        }
                    )

                dataset.save_episode()
                print(f"  Saved {demo_name}: {num_frames} frames, {len(demo_segments)} segments")

    print(f"\nDataset saved to {output_path}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "symbolic", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(ConvertArgs))
