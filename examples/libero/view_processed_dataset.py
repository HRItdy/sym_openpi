"""
Inspect a single sample from the fully-transformed dataset that the JAX model receives.

The pipeline applied here mirrors training exactly:
  1. LeRobot raw data  (image, wrist_image, state, actions, task)
  2. RepackTransform   (observation/image, observation/wrist_image, …)
  3. LiberoSymbolicInputs  (image/{base_0_rgb,…}, image_mask/…, state, actions, prompt)
  4. DeltaActions      (if extra_delta_transform=True in config)
  5. ResizeImages      (224×224)
  6. TokenizePrompt    (tokenized_prompt, tokenized_prompt_mask; consumes 'prompt')
  7. PadStatesAndActions (state → 32 dims, actions → 32 dims)

Normalization is intentionally skipped so values remain in physical units.

Usage
-----
  uv run examples/libero/view_processed_dataset.py
  uv run examples/libero/view_processed_dataset.py --config-name pi0_libero_symbolic --sample-index 42
  uv run examples/libero/view_processed_dataset.py --save-images ./inspect_out
  uv run examples/libero/view_processed_dataset.py --gui
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# ── openpi imports ────────────────────────────────────────────────────────────
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


# ── helpers ───────────────────────────────────────────────────────────────────

STATE_LABELS  = ["ee_x", "ee_y", "ee_z", "aa_r", "aa_p", "aa_y", "grip_l", "grip_r"]
ACTION_LABELS = ["dx",   "dy",   "dz",   "daa_r","daa_p","daa_y","grip",   "seg_done"]


def _fmt_arr(arr: np.ndarray, max_vals: int = 8) -> str:
    flat = arr.flat
    vals = [f"{x:.4f}" for _, x in zip(range(max_vals), flat)]
    suffix = " …" if arr.size > max_vals else ""
    return f"[{', '.join(vals)}{suffix}]"


def _section(title: str) -> None:
    width = 72
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _print_sample(sample: dict, tokenizer_spm=None) -> None:
    """Pretty-print every key in a processed sample."""

    _section("RAW KEY INVENTORY")
    for k, v in _walk(sample):
        arr = np.asarray(v)
        print(f"  {k:<45s}  shape={arr.shape}  dtype={arr.dtype}")

    # ── images ────────────────────────────────────────────────────────────────
    _section("IMAGES  (uint8 or float32, range shown)")
    img_dict = sample.get("image", {})
    mask_dict = sample.get("image_mask", {})
    for cam, img in img_dict.items():
        arr = np.asarray(img)
        mask = bool(np.asarray(mask_dict.get(cam, True)))
        r = f"{arr.min():.1f}–{arr.max():.1f}"
        print(f"  {cam:<25s}  {str(arr.shape):<20s}  {arr.dtype}  range={r}  mask={mask}")

    # ── state ─────────────────────────────────────────────────────────────────
    _section("STATE  (first 8 physical dims; rest are zero-padding)")
    state = np.asarray(sample["state"])
    print(f"  shape={state.shape}  dtype={state.dtype}")
    for i, label in enumerate(STATE_LABELS):
        if i < state.shape[-1]:
            print(f"    [{i:2d}] {label:<12s} = {state[i]:.6f}")
    if state.shape[-1] > len(STATE_LABELS):
        pad_start = len(STATE_LABELS)
        print(f"    [{pad_start:2d}…{state.shape[-1]-1:2d}] (zero-padding)")

    # ── actions ───────────────────────────────────────────────────────────────
    _section("ACTIONS  (horizon × action_dim; first frame shown)")
    if "actions" in sample:
        actions = np.asarray(sample["actions"])
        print(f"  shape={actions.shape}  dtype={actions.dtype}")
        first = actions[0]
        for i, label in enumerate(ACTION_LABELS):
            if i < first.shape[-1]:
                print(f"    [{i:2d}] {label:<12s} = {first[i]:.6f}")
        if first.shape[-1] > len(ACTION_LABELS):
            pad_start = len(ACTION_LABELS)
            print(f"    [{pad_start:2d}…{first.shape[-1]-1:2d}] (zero-padding)")
        seg_done_col = actions[:, 7] if actions.shape[-1] > 7 else None
        if seg_done_col is not None:
            pct = seg_done_col.mean() * 100
            print(f"\n  seg_done across horizon: {_fmt_arr(seg_done_col)}  (mean={pct:.1f}%)")
    else:
        print("  (not present in sample)")

    # ── tokenized prompt ──────────────────────────────────────────────────────
    _section("TOKENIZED PROMPT  (what the language model sees)")
    tokens = np.asarray(sample["tokenized_prompt"])
    mask   = np.asarray(sample["tokenized_prompt_mask"])
    active = int(mask.sum())
    print(f"  tokenized_prompt      shape={tokens.shape}  dtype={tokens.dtype}")
    print(f"  tokenized_prompt_mask shape={mask.shape}  active_tokens={active}/{len(tokens)}")
    print(f"  token IDs (active):   {tokens[:active].tolist()}")
    if tokenizer_spm is not None:
        decoded = tokenizer_spm.decode(tokens[:active].tolist())
        print(f"\n  Decoded text:\n    {decoded!r}")
    else:
        print("\n  (pass --decode to see decoded text; requires sentencepiece download)")


def _walk(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten nested dicts for display."""
    items = []
    for k, v in d.items():
        full_key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_walk(v, full_key))
        else:
            items.append((full_key, v))
    return items


def _save_images(sample: dict, out_dir: pathlib.Path) -> None:
    from PIL import Image  # local import so PIL is optional

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dict = sample.get("image", {})
    for cam, img in img_dict.items():
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        path = out_dir / f"{cam}.png"
        pil.save(path)
        print(f"  Saved {cam} → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def _build_dataset(config_name: str):
    """Build config + fully-transformed dataset (no norm stats)."""
    config      = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset     = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            *data_config.model_transforms.inputs,
        ],
    )
    return config, dataset


def _load_spm():
    import sentencepiece
    import openpi.shared.download as download
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════

PAL = {
    "bg":     "#0f1117", "panel":  "#161b22", "border": "#21262d",
    "accent": "#58a6ff", "green":  "#3fb950", "red":    "#f85149",
    "text":   "#e6edf3", "muted":  "#8b949e",
    "r0":     "#0d1117", "r1":     "#161b22",
}

IMG_W, IMG_H = 224, 224


def _to_photo(arr: np.ndarray, w: int, h: int):
    from PIL import Image, ImageTk
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        # float32 in [-1, 1] → uint8 [0, 255]
        arr = np.clip((arr + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((w, h), Image.BILINEAR)
    return ImageTk.PhotoImage(pil)


class Viewer:
    """Tkinter GUI for browsing fully-transformed samples."""

    def __init__(self, dataset, config, spm=None):
        import tkinter as tk
        from tkinter import ttk

        self._tk  = tk
        self._ttk = ttk
        self.dataset = dataset
        self.config  = config
        self.spm     = spm
        self._n      = len(dataset)
        self._idx    = 0
        self._cache: dict[int, dict] = {}  # sample cache
        self._photo_base  = None
        self._photo_wrist = None

        self.root = tk.Tk()
        self._setup_style()
        self._build_ui()
        self._show(0)

    def _setup_style(self):
        tk  = self._tk
        ttk = self._ttk
        self.root.title("Processed Dataset Inspector  —  JAX Model Input")
        self.root.configure(bg=PAL["bg"])
        self.root.minsize(1200, 760)
        try:
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1280x800")

        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("TFrame",   background=PAL["bg"])
        s.configure("P.TFrame", background=PAL["panel"])
        s.configure("TLabel",   background=PAL["bg"],    foreground=PAL["text"],
                    font=("Courier New", 10))
        s.configure("H.TLabel", background=PAL["bg"],    foreground=PAL["accent"],
                    font=("Courier New", 12, "bold"))
        s.configure("M.TLabel", background=PAL["panel"], foreground=PAL["muted"],
                    font=("Courier New", 9))
        s.configure("TButton",  background=PAL["border"], foreground=PAL["text"],
                    font=("Courier New", 10), borderwidth=0, relief="flat")
        s.map("TButton", background=[("active", "#30363d")])
        s.configure("Treeview", background=PAL["panel"],  foreground=PAL["text"],
                    fieldbackground=PAL["panel"], rowheight=22,
                    font=("Courier New", 9))
        s.configure("Treeview.Heading", background=PAL["border"],
                    foreground=PAL["muted"], font=("Courier New", 9, "bold"))
        s.map("Treeview",
              background=[("selected", PAL["accent"])],
              foreground=[("selected", "#0d1117")])

    def _build_ui(self):
        tk  = self._tk
        ttk = self._ttk
        root = self.root

        # ── top nav bar ───────────────────────────────────────────────────────
        nav = ttk.Frame(root, style="P.TFrame", height=48)
        nav.pack(fill=tk.X, pady=(0, 1))
        nav.pack_propagate(False)

        ttk.Button(nav, text="|<", width=3,
                   command=lambda: self._seek(0)).pack(side=tk.LEFT, padx=(8, 2), pady=8)
        ttk.Button(nav, text="<",  width=3,
                   command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=2, pady=8)
        ttk.Button(nav, text=">",  width=3,
                   command=lambda: self._step(1)).pack(side=tk.LEFT, padx=2, pady=8)
        ttk.Button(nav, text=">|", width=3,
                   command=lambda: self._seek(self._n - 1)).pack(side=tk.LEFT, padx=2, pady=8)

        ttk.Label(nav, text="Sample:", background=PAL["panel"],
                  foreground=PAL["muted"],
                  font=("Courier New", 10)).pack(side=tk.LEFT, padx=(16, 4), pady=8)

        self._var_idx = tk.StringVar(value="0")
        entry = tk.Entry(nav, textvariable=self._var_idx, width=7,
                         bg=PAL["border"], fg=PAL["text"],
                         insertbackground=PAL["text"],
                         relief="flat", font=("Courier New", 10))
        entry.pack(side=tk.LEFT, pady=8)
        entry.bind("<Return>", self._on_entry)

        self._var_total = tk.StringVar(value=f"/ {self._n - 1}")
        ttk.Label(nav, textvariable=self._var_total, background=PAL["panel"],
                  foreground=PAL["muted"],
                  font=("Courier New", 10)).pack(side=tk.LEFT, padx=4, pady=8)

        self._var_info = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self._var_info, background=PAL["panel"],
                  foreground=PAL["accent"],
                  font=("Courier New", 10)).pack(side=tk.LEFT, padx=20, pady=8)

        # slider
        self._slider_var = tk.IntVar(value=0)
        tk.Scale(nav, from_=0, to=max(self._n - 1, 1),
                 orient=tk.HORIZONTAL, variable=self._slider_var,
                 bg=PAL["panel"], fg=PAL["text"],
                 troughcolor=PAL["border"], highlightthickness=0,
                 showvalue=False, length=400,
                 command=self._on_slider).pack(side=tk.RIGHT, padx=12, pady=8)

        root.bind("<Left>",  lambda e: self._step(-1))
        root.bind("<Right>", lambda e: self._step(1))
        root.bind("<Home>",  lambda e: self._seek(0))
        root.bind("<End>",   lambda e: self._seek(self._n - 1))

        # ── main content ──────────────────────────────────────────────────────
        content = ttk.Frame(root)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # ── left: cameras ─────────────────────────────────────────────────────
        cam_col = ttk.Frame(content)
        cam_col.pack(side=tk.LEFT, anchor="n")

        ttk.Label(cam_col, text="base_0_rgb  (224×224)",
                  foreground=PAL["muted"],
                  font=("Courier New", 8)).pack(anchor="w")
        self._base_canvas = tk.Canvas(cam_col, width=IMG_W, height=IMG_H,
                                      bg=PAL["border"], highlightthickness=0)
        self._base_canvas.pack(pady=(2, 10))

        ttk.Label(cam_col, text="left_wrist_0_rgb  (224×224)",
                  foreground=PAL["muted"],
                  font=("Courier New", 8)).pack(anchor="w")
        self._wrist_canvas = tk.Canvas(cam_col, width=IMG_W, height=IMG_H,
                                       bg=PAL["border"], highlightthickness=0)
        self._wrist_canvas.pack(pady=(2, 0))

        # ── right: data panels ────────────────────────────────────────────────
        right = ttk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(16, 0))

        # prompt box
        ttk.Label(right, text="TOKENIZED PROMPT  (decoded)",
                  foreground=PAL["muted"],
                  font=("Courier New", 8, "bold")).pack(anchor="w")
        pf = tk.Frame(right, bg=PAL["panel"],
                      highlightbackground=PAL["border"], highlightthickness=1)
        pf.pack(fill=tk.X, pady=(2, 10))
        self._prompt_box = tk.Text(pf, height=4, wrap=tk.WORD,
                                   bg=PAL["panel"], fg=PAL["accent"],
                                   font=("Courier New", 9),
                                   relief="flat", bd=6, state=tk.DISABLED)
        self._prompt_box.pack(fill=tk.X)

        # seg_done timeline
        ttk.Label(right,
                  text="SEG_DONE  across action horizon  (dim 7 of actions; red=1.0)",
                  foreground=PAL["muted"],
                  font=("Courier New", 8, "bold")).pack(anchor="w")
        self._tl = tk.Canvas(right, height=22, bg=PAL["border"],
                             highlightthickness=0)
        self._tl.pack(fill=tk.X, pady=(2, 10))

        # state + action tables
        tables = ttk.Frame(right)
        tables.pack(fill=tk.BOTH, expand=True)
        self._state_tree = self._make_tree(tables, "STATE  (physical dims)", STATE_LABELS)
        self._action_tree = self._make_tree(tables, "ACTIONS  t=0  (physical dims)", ACTION_LABELS)

        # bottom: token id row
        bot = ttk.Frame(root, style="P.TFrame", height=32)
        bot.pack(fill=tk.X, side=tk.BOTTOM)
        bot.pack_propagate(False)
        self._var_tokens = tk.StringVar(value="")
        ttk.Label(bot, textvariable=self._var_tokens, background=PAL["panel"],
                  foreground=PAL["muted"],
                  font=("Courier New", 8)).pack(side=tk.LEFT, padx=10, pady=6)

    def _make_tree(self, parent, title: str, labels: list[str]) -> object:
        ttk = self._ttk
        tk  = self._tk
        f = ttk.Frame(parent)
        f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        ttk.Label(f, text=title, foreground=PAL["muted"],
                  font=("Courier New", 8, "bold")).pack(anchor="w")
        t = ttk.Treeview(f, columns=("d", "v"), show="headings",
                         height=len(labels))
        t.heading("d", text="dim");   t.column("d", width=80,  anchor="e")
        t.heading("v", text="value"); t.column("v", width=120, anchor="w")
        t.pack(fill=tk.BOTH, expand=True)
        for i, lbl in enumerate(labels):
            t.insert("", self._tk.END, iid=lbl, values=(lbl, "—"),
                     tags=("e" if i % 2 == 0 else "o",))
        t.tag_configure("e", background=PAL["r0"])
        t.tag_configure("o", background=PAL["r1"])
        return t

    # ── loading ───────────────────────────────────────────────────────────────

    def _get_sample(self, idx: int) -> dict:
        if idx not in self._cache:
            self._cache[idx] = self.dataset[idx]
            if len(self._cache) > 32:  # evict oldest
                oldest = next(iter(self._cache))
                del self._cache[oldest]
        return self._cache[idx]

    # ── rendering ─────────────────────────────────────────────────────────────

    def _show(self, idx: int):
        self._idx = idx
        self._var_idx.set(str(idx))
        self._slider_var.set(idx)

        sample = self._get_sample(idx)

        # images
        img_dict = sample.get("image", {})
        if "base_0_rgb" in img_dict:
            self._photo_base = _to_photo(np.asarray(img_dict["base_0_rgb"]), IMG_W, IMG_H)
            self._base_canvas.create_image(0, 0, anchor=self._tk.NW, image=self._photo_base)
        if "left_wrist_0_rgb" in img_dict:
            self._photo_wrist = _to_photo(np.asarray(img_dict["left_wrist_0_rgb"]), IMG_W, IMG_H)
            self._wrist_canvas.create_image(0, 0, anchor=self._tk.NW, image=self._photo_wrist)

        # prompt
        tokens = np.asarray(sample["tokenized_prompt"])
        mask   = np.asarray(sample["tokenized_prompt_mask"])
        active = int(mask.sum())
        if self.spm is not None:
            text = self.spm.decode(tokens[:active].tolist())
        else:
            text = f"(run with --decode to see text)  token_ids={tokens[:active].tolist()}"
        self._prompt_box.config(state=self._tk.NORMAL)
        self._prompt_box.delete("1.0", self._tk.END)
        self._prompt_box.insert("1.0", text)
        self._prompt_box.config(state=self._tk.DISABLED)
        self._var_tokens.config = None  # silence unused warning
        self._var_tokens.set(
            f"token_ids (active {active}/{len(tokens)}): "
            + str(tokens[:active].tolist())[:120]
        )

        # state table
        state = np.asarray(sample["state"])
        for i, lbl in enumerate(STATE_LABELS):
            if i < state.shape[-1]:
                self._state_tree.set(lbl, "v", f"{state[i]:.6f}")

        # action table (first frame)
        if "actions" in sample:
            actions = np.asarray(sample["actions"])
            first = actions[0]
            for i, lbl in enumerate(ACTION_LABELS):
                if i < first.shape[-1]:
                    self._action_tree.set(lbl, "v", f"{first[i]:.6f}")
            # seg_done timeline
            if actions.shape[-1] > 7:
                self._draw_timeline(actions[:, 7])
        else:
            self._draw_timeline(np.zeros(1))

        # nav info
        ah = self.config.model.action_horizon
        ad = self.config.model.action_dim
        tl = self.config.model.max_token_len
        self._var_info.set(
            f"action_horizon={ah}  action_dim={ad}  max_token_len={tl}"
        )

    def _draw_timeline(self, seg_done: np.ndarray):
        c = self._tl
        c.delete("all")
        w = c.winfo_width() or 700
        h = 22
        n = len(seg_done)
        bw = max(1, w / n)
        for i, v in enumerate(seg_done):
            x0, x1 = int(i * bw), int((i + 1) * bw)
            c.create_rectangle(x0, 2, x1, h - 2,
                               fill=PAL["red"] if v > 0.5 else "#1c2128",
                               outline="")

    # ── navigation ────────────────────────────────────────────────────────────

    def _seek(self, idx: int):
        self._show(max(0, min(idx, self._n - 1)))

    def _step(self, d: int):
        self._seek(self._idx + d)

    def _on_slider(self, val):
        idx = int(float(val))
        if idx != self._idx:
            self._show(idx)

    def _on_entry(self, _event=None):
        try:
            idx = int(self._var_idx.get())
            self._seek(idx)
        except ValueError:
            pass

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect one fully-transformed sample as seen by the JAX model"
    )
    ap.add_argument("--config-name",   default="pi0_libero_symbolic_low_mem",
                    help="TrainConfig name (default: pi0_libero_symbolic_low_mem)")
    ap.add_argument("--sample-index",  type=int, default=0,
                    help="Index into the dataset (default: 0)")
    ap.add_argument("--save-images",   default=None, metavar="DIR",
                    help="If given, save camera images as PNG files to this directory")
    ap.add_argument("--decode",        action="store_true",
                    help="Decode tokenized_prompt back to text (downloads tokenizer once)")
    ap.add_argument("--gui",           action="store_true",
                    help="Open interactive Tkinter GUI for browsing the full dataset")
    args = ap.parse_args()

    print(f"Config : {args.config_name}")

    config, dataset = _build_dataset(args.config_name)
    n = len(dataset)
    print(f"Dataset: {n} samples  (action_horizon={config.model.action_horizon}, "
          f"action_dim={config.model.action_dim}, max_token_len={config.model.max_token_len})")

    # ── sentencepiece tokenizer (needed for --decode and --gui) ───────────────
    spm = None
    if args.decode or args.gui:
        print("Loading sentencepiece tokenizer …")
        spm = _load_spm()

    # ── GUI mode ──────────────────────────────────────────────────────────────
    if args.gui:
        viewer = Viewer(dataset, config, spm=spm)
        viewer.run()
        return

    # ── CLI mode ──────────────────────────────────────────────────────────────
    print(f"Sample : {args.sample_index}")
    if args.sample_index >= n:
        print(f"Error: sample index {args.sample_index} out of range [0, {n-1}]", file=sys.stderr)
        sys.exit(1)

    sample = dataset[args.sample_index]
    _print_sample(sample, tokenizer_spm=spm)

    if args.save_images:
        _section("SAVING IMAGES")
        _save_images(sample, pathlib.Path(args.save_images))

    print()


if __name__ == "__main__":
    main()
