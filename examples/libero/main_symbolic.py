"""
Symbolic LIBERO inference — two-session design.

Session 1  (mode=session1)
    Run the trained symbolic π₀ policy on a task that is already in the training
    dataset.  The canonical operator sequence is loaded directly from the
    segmentation annotation JSONs and fed to the model as symbolic prompts, one
    operator at a time.  The model's ``segment_done`` output (8th action
    dimension) triggers automatic subtask transitions.  Use this session to
    verify that the model has learned operator-level task structure on seen tasks.

Session 2  (mode=session2)
    Generalise to unseen LIBERO-10 tasks via LLM-assisted task planning:
      1. Retrieve the K most similar tasks from the training annotation database
         using TF-IDF cosine similarity on task descriptions.
      2. Build a few-shot prompt from their annotated operator sequences
         (descriptions + preconditions + effects).
      3. Call an LLM (OpenAI / Anthropic) to decompose the new task into an
         operator sequence in the same structured format.
      4. Drive the trained symbolic policy through the generated operator
         sequence, using ``segment_done`` to advance between subtasks.

──────────────────────────────────────────────────────────────────────────────
Prerequisites
──────────────────────────────────────────────────────────────────────────────
1.  Start the policy server (separate terminal, GPU machine):

        python scripts/serve_policy.py \\
            policy:checkpoint \\
            --policy.config pi0_libero_symbolic \\
            --policy.dir checkpoints/pi0_libero_symbolic/exp/50000

2.  For Session 2, set one of:
        export OPENAI_API_KEY=sk-...
        export ANTHROPIC_API_KEY=sk-ant-...

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
    # Session 1 — evaluate on tasks already in the training dataset
    python examples/libero/main_symbolic.py session1 \\
        --annotation_dir /path/to/segmentation_jsons \\
        --task_suite_name libero_spatial \\
        --num_trials_per_task 10

    # Session 2 — LIBERO-10 with LLM decomposition
    python examples/libero/main_symbolic.py session2 \\
        --annotation_dir /path/to/segmentation_jsons \\
        --task_suite_name libero_10 \\
        --llm_provider openai \\
        --top_k 5 \\
        --num_trials_per_task 10
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
import re
from typing import Literal

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Operator:
    """One atomic operator in a task plan."""

    description: str
    preconditions: list[str]
    effects: list[str]

    def __str__(self) -> str:
        prec = "; ".join(self.preconditions) if self.preconditions else "none"
        eff = "; ".join(self.effects) if self.effects else "none"
        return f'"{self.description}"  pre=[{prec}]  eff=[{eff}]'


@dataclasses.dataclass
class TaskRecord:
    """A training task with its canonical operator sequence."""

    task_name: str          # e.g. "KITCHEN_SCENE5_put_the_black_bowl_..."
    task_description: str   # e.g. "put the black bowl on top of the cabinet"
    operators: list[Operator]


# ──────────────────────────────────────────────────────────────────────────────
# Task database — loads segmentation annotation JSONs
# ──────────────────────────────────────────────────────────────────────────────


class TaskDatabase:
    """Loads segmentation annotation JSONs and supports similarity retrieval.

    Expected JSON format (produced by convert_libero_symbolic_to_lerobot.py):
    {
        "task_name": "KITCHEN_SCENE5_...",
        "canonical_operators": {"language": "<task description>"},
        "segments": {
            "demo_0": [
                {
                    "start_frame": 0, "end_frame": 45,
                    "annotation": {
                        "description": "pick up the black bowl",
                        "preconditions": ["Empty[gripper]"],
                        "effects": ["Holding[black_bowl]"]
                    }
                },
                ...
            ],
            ...
        }
    }
    """

    def __init__(self, annotation_dir: str) -> None:
        self.records: list[TaskRecord] = []
        self._load(pathlib.Path(annotation_dir))
        self._build_index()

    # ------------------------------------------------------------------
    def _load(self, annotation_dir: pathlib.Path) -> None:
        json_files = sorted(annotation_dir.glob("*_merged.json"))
        if not json_files:
            json_files = sorted(annotation_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No annotation JSON files found in {annotation_dir}. "
                "Run convert_libero_symbolic_to_lerobot.py first."
            )

        for json_path in json_files:
            with open(json_path) as fh:
                ann = json.load(fh)

            task_name: str = ann["task_name"]
            task_description: str = ann["canonical_operators"]["language"]

            # Derive canonical operator sequence from the first demo.
            demo_keys = sorted(ann["segments"].keys())
            segments = ann["segments"][demo_keys[0]]

            operators = [
                Operator(
                    description=seg["annotation"].get("description", ""),
                    preconditions=seg["annotation"].get("preconditions", []),
                    effects=seg["annotation"].get("effects", []),
                )
                for seg in segments
            ]

            self.records.append(
                TaskRecord(
                    task_name=task_name,
                    task_description=task_description,
                    operators=operators,
                )
            )

        logging.info("Loaded %d task records from %s", len(self.records), annotation_dir)

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        if not _HAS_SKLEARN:
            logging.warning(
                "scikit-learn not found; retrieval falls back to Jaccard word overlap. "
                "Install it with: pip install scikit-learn"
            )
            self._vectorizer = None
            self._tfidf_matrix = None
            return
        descriptions = [r.task_description for r in self.records]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._tfidf_matrix = self._vectorizer.fit_transform(descriptions)

    # ------------------------------------------------------------------
    def get_by_description(self, task_description: str) -> TaskRecord | None:
        """Exact or case-insensitive match on task description."""
        query = task_description.strip().lower()
        for r in self.records:
            if r.task_description.strip().lower() == query:
                return r
        return None

    # ------------------------------------------------------------------
    def retrieve_similar(self, query: str, top_k: int = 5) -> list[TaskRecord]:
        """Return the top_k most similar records to the query description."""
        if self._vectorizer is not None:
            q_vec = self._vectorizer.transform([query])
            sims = _cosine_similarity(q_vec, self._tfidf_matrix)[0]
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [self.records[i] for i in top_idx]

        # Fallback: Jaccard word overlap
        query_words = set(query.lower().split())
        scored = []
        for r in self.records:
            ref_words = set(r.task_description.lower().split())
            union = query_words | ref_words
            score = len(query_words & ref_words) / len(union) if union else 0.0
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]


# ──────────────────────────────────────────────────────────────────────────────
# LLM task decomposer
# ──────────────────────────────────────────────────────────────────────────────


def _format_example(record: TaskRecord) -> str:
    """Format a TaskRecord as a few-shot example block."""
    lines = [f'Task: "{record.task_description}"', "Operators:"]
    for i, op in enumerate(record.operators, 1):
        prec = "; ".join(op.preconditions) if op.preconditions else "none"
        eff = "; ".join(op.effects) if op.effects else "none"
        lines.append(
            f'  {i}. description: "{op.description}"'
            f' | preconditions: [{prec}]'
            f' | effects: [{eff}]'
        )
    return "\n".join(lines)


def build_llm_prompt(task_description: str, similar_tasks: list[TaskRecord]) -> str:
    """Build a few-shot decomposition prompt for the LLM."""
    examples = "\n\n".join(_format_example(r) for r in similar_tasks)
    return f"""You are a robot task planner for a Panda robot arm in a kitchen tabletop environment.
Your job is to decompose a manipulation task into a sequence of atomic operator steps,
each with symbolic STRIPS-style preconditions and effects.

Predicates:
  Empty[gripper]      – gripper holds nothing
  Holding[X]          – gripper is holding object X
  On[X, Y]            – object X rests on top of object Y
  In[X, Y]            – object X is inside container / receptacle Y
  At[location]        – arm is positioned near a location
  Open[X]             – drawer or door X is open
  Closed[X]           – drawer or door X is closed

Rules:
  - Use only the predicates listed above (invent new ones only when necessary).
  - Preconditions must be true *before* the operator begins.
  - Effects replace the relevant preconditions after the operator completes.
  - The last operator's effects should reflect the goal state of the task.
  - Keep descriptions concise and action-oriented (verb phrase).

──────────────────────────────────────────────
Few-shot examples from the training dataset:
──────────────────────────────────────────────

{examples}

──────────────────────────────────────────────
New task to decompose:
──────────────────────────────────────────────

Task: "{task_description}"

Return ONLY a valid JSON object with an "operators" array — no markdown, no extra text.

{{
  "operators": [
    {{
      "description": "<verb phrase describing this atomic step>",
      "preconditions": ["<predicate>", ...],
      "effects": ["<predicate>", ...]
    }},
    ...
  ]
}}"""


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    try:
        import openai
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-opus-4-7-20251101") -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def decompose_with_llm(
    task_description: str,
    similar_tasks: list[TaskRecord],
    provider: Literal["openai", "anthropic"] = "openai",
    llm_model: str | None = None,
) -> list[Operator]:
    """Call an LLM to produce an operator sequence for a new task."""
    prompt = build_llm_prompt(task_description, similar_tasks)

    logging.info("Calling %s LLM for task: %s", provider, task_description)
    if provider == "openai":
        raw = call_openai(prompt, model=llm_model or "gpt-4o")
    elif provider == "anthropic":
        raw = call_anthropic(prompt, model=llm_model or "claude-opus-4-7-20251101")
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        data = json.loads(cleaned)
        operators = [
            Operator(
                description=op["description"],
                preconditions=op.get("preconditions", []),
                effects=op.get("effects", []),
            )
            for op in data["operators"]
        ]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(
            f"Failed to parse LLM response as operator list: {exc}\n"
            f"Raw response:\n{raw}"
        ) from exc

    logging.info("LLM generated %d operators:", len(operators))
    for i, op in enumerate(operators):
        logging.info("  [%d] %s", i, op)
    return operators


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

_LIBERO_ENV_RESOLUTION = 256
_LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

_MAX_STEPS_BY_SUITE: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _get_libero_env(task, resolution: int, seed: int):
    task_description: str = task.language
    bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = quat.copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - q[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def _preprocess_obs(obs: dict, resize_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a raw LIBERO observation dict into (img, wrist_img, state)."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate(
        [
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    )
    return img, wrist, state


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic prompt helper
# ──────────────────────────────────────────────────────────────────────────────


def build_symbolic_prompt(
    task_description: str,
    operator: Operator,
) -> str:
    """Format the symbolic prompt for one operator (mirrors libero_symbolic_policy)."""
    prec_str = "; ".join(operator.preconditions) if operator.preconditions else "none"
    return (
        f"Task: {task_description}. "
        #f"Now: {operator.description}. "
        f"State: {prec_str}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic inference loop
# ──────────────────────────────────────────────────────────────────────────────


def run_symbolic_episode(
    env,
    initial_state,
    client: _websocket_client_policy.WebsocketClientPolicy,
    task_description: str,
    operators: list[Operator],
    *,
    resize_size: int = 224,
    replan_steps: int = 5,
    segment_done_threshold: float = 0.5,
    max_steps: int = 520,
    num_steps_wait: int = 10,
    max_steps_per_op: int = 150,
) -> tuple[bool, list[np.ndarray]]:
    """
    Run one symbolic episode.

    Subtask transition mechanism
    ----------------------------
    After each inference call the policy returns:
      - ``actions``      shape (action_horizon, 7)  — robot DoF actions
      - ``segment_done`` shape (action_horizon,)     — operator-completion signal

    We take the mean of ``segment_done`` over the next ``replan_steps`` steps.
    If it exceeds ``segment_done_threshold`` we advance op_idx → op_idx + 1 and
    query the model again with the updated symbolic prompt.

    A per-operator step timeout (``max_steps_per_op``) serves as a hard fallback
    in case the model fails to signal completion.

    Returns
    -------
    success : bool — whether the environment reported task completion
    replay_images : list[np.ndarray] — preprocessed frames for video saving
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    prompts = [build_symbolic_prompt(task_description, op) for op in operators]

    op_idx = 0
    steps_on_current_op = 0
    t = 0
    done = False
    replay_images: list[np.ndarray] = []
    action_plan: collections.deque = collections.deque()

    logging.info("Task: %s", task_description)
    logging.info("Plan (%d operators):", len(operators))
    for i, op in enumerate(operators):
        logging.info("  [%d] %s", i, op)

    while t < max_steps + num_steps_wait:
        # Wait for objects to settle after reset
        if t < num_steps_wait:
            obs, _, _, _ = env.step(_LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img, wrist, state = _preprocess_obs(obs, resize_size)
        replay_images.append(img)

        current_prompt = prompts[min(op_idx, len(prompts) - 1)]

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": current_prompt,
            }
            result = client.infer(element)
            robot_actions: np.ndarray = np.asarray(result["actions"])  # (H, 7)
            segment_done: np.ndarray | None = (
                np.asarray(result["segment_done"]) if "segment_done" in result else None
            )

            # ── Subtask transition check ─────────────────────────────────
            if segment_done is not None and op_idx < len(operators) - 1:
                window = segment_done[: min(replan_steps, len(segment_done))]
                if float(window.mean()) > segment_done_threshold:
                    op_idx += 1
                    steps_on_current_op = 0
                    action_plan.clear()
                    logging.info(
                        "  [segment_done] → operator [%d]: %s",
                        op_idx,
                        operators[op_idx].description,
                    )
                    # Re-query with updated prompt so actions match new operator
                    element["prompt"] = prompts[op_idx]
                    result = client.infer(element)
                    robot_actions = np.asarray(result["actions"])
                    segment_done = (
                        np.asarray(result["segment_done"])
                        if "segment_done" in result
                        else None
                    )

            action_plan.extend(robot_actions[:replan_steps])

        action = action_plan.popleft()
        obs, _reward, done, _info = env.step(action.tolist())
        steps_on_current_op += 1

        # ── Timeout fallback ─────────────────────────────────────────────
        if steps_on_current_op >= max_steps_per_op and op_idx < len(operators) - 1:
            op_idx += 1
            steps_on_current_op = 0
            action_plan.clear()
            logging.info(
                "  [timeout] → operator [%d]: %s",
                op_idx,
                operators[op_idx].description,
            )

        if done:
            break
        t += 1

    return bool(done), replay_images


# ──────────────────────────────────────────────────────────────────────────────
# Session runners
# ──────────────────────────────────────────────────────────────────────────────


def run_session1(args: "Session1Args") -> None:
    """
    Session 1: evaluate the symbolic policy on tasks from the training dataset.

    For each task in the chosen LIBERO suite, the script looks up the canonical
    operator sequence from the annotation database and drives the model through
    it, monitoring ``segment_done`` for automatic subtask transitions.

    Tasks that do not appear in the annotation database are skipped.
    """
    np.random.seed(args.seed)

    db = TaskDatabase(args.annotation_dir)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = _MAX_STEPS_BY_SUITE.get(args.task_suite_name, 400)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    total_episodes = total_successes = 0

    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, _LIBERO_ENV_RESOLUTION, args.seed)

        # Look up this task's canonical operator plan
        record = db.get_by_description(task_description)
        if record is None:
            logging.warning(
                "Task %r not found in annotation database — skipping.", task_description
            )
            env.close()
            continue

        logging.info(
            "\n=== Task [%d] %r — %d operators ===",
            task_id,
            task_description,
            len(record.operators),
        )

        task_episodes = task_successes = 0
        for ep_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes", leave=False):
            success, frames = run_symbolic_episode(
                env,
                initial_states[ep_idx % len(initial_states)],
                client,
                task_description,
                record.operators,
                resize_size=args.resize_size,
                replan_steps=args.replan_steps,
                segment_done_threshold=args.segment_done_threshold,
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                max_steps_per_op=args.max_steps_per_op,
            )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            suffix = "success" if success else "failure"
            tag = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"s1_{tag}_ep{ep_idx}_{suffix}.mp4",
                [np.asarray(f) for f in frames],
                fps=10,
            )
            logging.info(
                "Episode %d: %s  (task SR %.1f%%, total SR %.1f%%)",
                ep_idx,
                suffix,
                100.0 * task_successes / task_episodes,
                100.0 * total_successes / total_episodes,
            )

        logging.info(
            "Task %r success rate: %.1f%%",
            task_description,
            100.0 * task_successes / max(task_episodes, 1),
        )
        env.close()

    logging.info(
        "\n=== Session 1 final: %d/%d (%.1f%%) ===",
        total_successes,
        total_episodes,
        100.0 * total_successes / max(total_episodes, 1),
    )


def run_session2(args: "Session2Args") -> None:
    """
    Session 2: LIBERO-10 evaluation with LLM-generated task decomposition.

    For each unseen LIBERO-10 task:
      1. Retrieve the ``top_k`` most similar training tasks from the annotation DB.
      2. Use their operator sequences as few-shot context to prompt an LLM
         (OpenAI gpt-4o or Anthropic claude-opus-4-7) to decompose the new task.
      3. Cache the generated plan (one LLM call per unique task description).
      4. Drive the symbolic policy through the generated operator sequence.
    """
    np.random.seed(args.seed)

    db = TaskDatabase(args.annotation_dir)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = _MAX_STEPS_BY_SUITE.get(args.task_suite_name, 520)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Cache LLM plans: task_description → list[Operator]
    plan_cache: dict[str, list[Operator]] = {}

    total_episodes = total_successes = 0

    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, _LIBERO_ENV_RESOLUTION, args.seed)

        # ── Generate plan once per task ───────────────────────────────────
        if task_description not in plan_cache:
            similar = db.retrieve_similar(task_description, top_k=args.top_k)
            logging.info(
                "Retrieved %d similar tasks for %r:", len(similar), task_description
            )
            for r in similar:
                logging.info("  • %s", r.task_description)

            try:
                operators = decompose_with_llm(
                    task_description,
                    similar,
                    provider=args.llm_provider,
                    llm_model=args.llm_model or None,
                )
            except Exception as exc:
                logging.error("LLM decomposition failed: %s — skipping task.", exc)
                env.close()
                continue

            plan_cache[task_description] = operators

            # Optionally persist plan to disk for inspection / reproducibility
            if args.plan_output_dir:
                plan_dir = pathlib.Path(args.plan_output_dir)
                plan_dir.mkdir(parents=True, exist_ok=True)
                plan_path = plan_dir / f"{task.problem_folder}_{task_id}_plan.json"
                with open(plan_path, "w") as fh:
                    json.dump(
                        {
                            "task_description": task_description,
                            "similar_tasks": [r.task_description for r in similar],
                            "operators": [
                                {
                                    "description": op.description,
                                    "preconditions": op.preconditions,
                                    "effects": op.effects,
                                }
                                for op in operators
                            ],
                        },
                        fh,
                        indent=2,
                    )
                logging.info("Plan saved to %s", plan_path)

        operators = plan_cache[task_description]

        logging.info(
            "\n=== Task [%d] %r — %d LLM operators ===",
            task_id,
            task_description,
            len(operators),
        )

        task_episodes = task_successes = 0
        for ep_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes", leave=False):
            success, frames = run_symbolic_episode(
                env,
                initial_states[ep_idx % len(initial_states)],
                client,
                task_description,
                operators,
                resize_size=args.resize_size,
                replan_steps=args.replan_steps,
                segment_done_threshold=args.segment_done_threshold,
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                max_steps_per_op=args.max_steps_per_op,
            )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            suffix = "success" if success else "failure"
            tag = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"s2_{tag}_ep{ep_idx}_{suffix}.mp4",
                [np.asarray(f) for f in frames],
                fps=10,
            )
            logging.info(
                "Episode %d: %s  (task SR %.1f%%, total SR %.1f%%)",
                ep_idx,
                suffix,
                100.0 * task_successes / task_episodes,
                100.0 * total_successes / total_episodes,
            )

        logging.info(
            "Task %r success rate: %.1f%%",
            task_description,
            100.0 * task_successes / max(task_episodes, 1),
        )
        env.close()

    logging.info(
        "\n=== Session 2 final: %d/%d (%.1f%%) ===",
        total_successes,
        total_episodes,
        100.0 * total_successes / max(total_episodes, 1),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument dataclasses
# ──────────────────────────────────────────────────────────────────────────────

_SHARED_DOCSTRING = """
Shared parameters
-----------------
annotation_dir : Path to the directory containing segmentation JSON files
    (produced by examples/libero/convert_libero_symbolic_to_lerobot.py).
host / port : Address of the running policy server
    (started with scripts/serve_policy.py).
task_suite_name : Which LIBERO suite to evaluate on.
num_trials_per_task : Number of rollouts per task.
replan_steps : How many actions to execute before re-querying the policy.
segment_done_threshold : Mean segment_done value above which we advance to
    the next operator (0 = advance immediately, 1 = never advance).
max_steps_per_op : Hard timeout per operator; forces transition if exceeded.
"""


@dataclasses.dataclass
class Session1Args:
    """Session 1 — evaluate the symbolic policy on trained tasks."""

    annotation_dir: str
    """Directory with segmentation JSON files from the training dataset."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 50
    num_steps_wait: int = 10

    # Inference
    resize_size: int = 224
    replan_steps: int = 5
    segment_done_threshold: float = 0.5
    max_steps_per_op: int = 150

    # Output
    video_out_path: str = "data/libero/videos/session1"
    seed: int = 7


@dataclasses.dataclass
class Session2Args:
    """Session 2 — LIBERO-10 evaluation with LLM-generated task decomposition."""

    annotation_dir: str
    """Directory with segmentation JSON files from the training dataset."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # LIBERO
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 50
    num_steps_wait: int = 10

    # Inference
    resize_size: int = 224
    replan_steps: int = 5
    segment_done_threshold: float = 0.5
    max_steps_per_op: int = 150

    # Retrieval
    top_k: int = 5
    """Number of similar training tasks to use as few-shot LLM context."""

    # LLM
    llm_provider: Literal["openai", "anthropic"] = "openai"
    """LLM provider for task decomposition."""
    llm_model: str = ""
    """Override the default model (e.g. 'gpt-4o-mini', 'claude-haiku-4-5-20251001').
    Leave empty to use the provider default."""

    # Output
    video_out_path: str = "data/libero/videos/session2"
    plan_output_dir: str = "data/libero/plans"
    """Directory to save the LLM-generated plans as JSON for inspection."""
    seed: int = 7


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main(args: Session1Args | Session2Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if isinstance(args, Session1Args):
        run_session1(args)
    else:
        run_session2(args)


if __name__ == "__main__":
    tyro.cli(
        tyro.conf.Subcommands[Session1Args | Session2Args],
        description=__doc__,
    )
