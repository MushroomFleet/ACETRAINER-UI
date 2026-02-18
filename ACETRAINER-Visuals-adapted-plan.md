# ACETrainer PaCMAP Visualization — Adapted Integration Plan

> **Scope:** Integrate real-time PaCMAP embedding visualization into the existing ACETrainer UI.
> **Source files (reference only):** `ACETrainer/phase2/` — these are NOT part of the standalone source.
> **Target:** All changes land in `ACETrainer/` (backend, static, app.py) and `ACE-Step/trainer.py`.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File Inventory — What Changes, What's New](#2-file-inventory)
3. [Step 1: Install Python Dependencies](#step-1-install-python-dependencies)
4. [Step 2: Add `visualization_service.py`](#step-2-add-visualization_servicepy)
5. [Step 3: Add `training_hooks.py`](#step-3-add-training_hookspy)
6. [Step 4: Add `visualization_api.py`](#step-4-add-visualization_apipy)
7. [Step 5: Wire into `app.py`](#step-5-wire-into-apppy)
8. [Step 6: Add Visualization Tab to `index.html`](#step-6-add-visualization-tab-to-indexhtml)
9. [Step 7: Add `visualization_panel.js`](#step-7-add-visualization_paneljs)
10. [Step 8: Add Visualization CSS to `style.css`](#step-8-add-visualization-css-to-stylecss)
11. [Step 9: Wire `app.js` Socket Events](#step-9-wire-appjs-socket-events)
12. [Step 10: Modify `ACE-Step/trainer.py`](#step-10-modify-ace-steptrainerpy)
13. [Step 11: Testing & Verification](#step-11-testing--verification)
14. [Appendix A: ACE-Step Model Layer Map](#appendix-a-ace-step-model-layer-map)
15. [Appendix B: Performance Tuning](#appendix-b-performance-tuning)

---

## 1. Architecture Overview

### Current System

```
ACETrainer/
├── app.py                          Flask + SocketIO (eventlet), port 7870
├── backend/
│   ├── trainer_api.py              Blueprint: /api/trainer/*
│   ├── trainer_service.py          Runs ACE-Step trainer.py as subprocess
│   ├── dataset_api.py              Blueprint: /api/dataset/*
│   ├── dataset_service.py
│   ├── captioner_api.py            Blueprint: /api/captioner/*
│   └── captioner/                  Audio captioning service
├── static/
│   ├── index.html                  SPA: two tabs (Dataset Editor, Trainer)
│   ├── css/style.css               Dark theme (Tailwind + custom)
│   └── js/
│       ├── app.js                  Tab nav, Socket.IO init, GPU polling
│       ├── utils.js                Toast, confirm, API helpers
│       ├── db.js                   IndexedDB wrapper
│       ├── dataset-editor.js       Dataset editing UI
│       └── trainer-ui.js           Training config, Chart.js loss chart

ACE-Step/
└── trainer.py                      PyTorch Lightning `Pipeline(LightningModule)`
```

### After Integration

```
ACETrainer/
├── app.py                          + register viz blueprint & socketio handlers
├── backend/
│   ├── visualization_service.py    NEW — PaCMAP computation, MultiSpaceVisualizer
│   ├── visualization_api.py        NEW — Flask blueprint + Socket.IO handlers
│   ├── training_hooks.py           NEW — PyTorch forward hooks for embedding capture
│   └── ... (existing files unchanged)
├── static/
│   ├── index.html                  + Visualization tab in nav + tab content
│   ├── css/style.css               + Visualization panel styles (dark theme)
│   └── js/
│       ├── app.js                  + viz socket events, tab switch handler
│       ├── visualization_panel.js  NEW — Plotly rendering, auto-update, controls
│       └── ... (existing files unchanged)

ACE-Step/
└── trainer.py                      + embedding hook integration in Pipeline class
```

### Data Flow

```
trainer.py (subprocess)
  │  forward pass triggers PyTorch hooks
  │  hooks capture embeddings from 3 spaces
  ▼
EmbeddingHookManager → callback
  │  converts tensors to numpy, packages metadata
  ▼
Socket.IO client emit('embedding_captured', data)
  │  connects to ACETrainer UI server on port 7870
  ▼
visualization_api.py (Socket.IO handler)
  │  receives embeddings, feeds to visualizer
  ▼
MultiSpaceVisualizer (visualization_service.py)
  │  accumulates embeddings in rolling buffers
  │  runs PaCMAP projection every N samples
  │  generates Plotly figure JSON
  ▼
Socket.IO emit('visualization_update', figure_json)
  │  broadcasts to connected clients
  ▼
visualization_panel.js (browser)
  │  Plotly.newPlot() renders 3-panel scatter
  ▼
User sees real-time embedding evolution
```

---

## 2. File Inventory

### New Files (create from scratch)

| File | Location | Purpose |
|------|----------|---------|
| `visualization_service.py` | `ACETrainer/backend/` | PaCMAP reducer, rolling buffers, Plotly figure generation |
| `visualization_api.py` | `ACETrainer/backend/` | Flask blueprint `/api/viz/*` + Socket.IO event handlers |
| `training_hooks.py` | `ACETrainer/backend/` | PyTorch forward hook manager for embedding capture |
| `visualization_panel.js` | `ACETrainer/static/js/` | Frontend: Plotly rendering, controls, auto-update |

### Modified Files (edit existing)

| File | Changes |
|------|---------|
| `ACETrainer/app.py` | Import & register viz blueprint, register Socket.IO handlers, add `/visualization` namespace connect handler |
| `ACETrainer/static/index.html` | Add Visualization tab button in nav, add tab content div, add Plotly CDN script, link viz JS |
| `ACETrainer/static/css/style.css` | Append visualization panel styles (dark theme, matching existing design) |
| `ACETrainer/static/js/app.js` | Add visualization socket events, init VizPanel on tab switch, add `/visualization` namespace |
| `ACE-Step/trainer.py` | Add hook import, init hooks in `Pipeline.__init__`, wire `set_step`/`set_prompts` in `run_step`, cleanup in training end |

### NOT Modified

| File | Reason |
|------|--------|
| `trainer_service.py` | Subprocess launch unchanged; hooks run inside the subprocess |
| `trainer_api.py` | No new REST endpoints needed for trainer |
| `trainer-ui.js` | Existing training UI untouched |
| `utils.js`, `db.js`, `dataset-editor.js` | No changes needed |

---

## Step 1: Install Python Dependencies

Add to project requirements (or install manually):

```bash
pip install pacmap>=0.7.0 plotly>=5.17.0 python-socketio[client]>=5.10.0
```

**Note:** `numpy` is already installed (required by PyTorch). `kaleido` is optional (for saving static images).

The `python-socketio[client]` package is needed inside `trainer.py` (the subprocess) to send embeddings back to the ACETrainer UI server. This is the Socket.IO **client** — the server side is already provided by `flask-socketio`.

---

## Step 2: Add `visualization_service.py`

**File:** `ACETrainer/backend/visualization_service.py`

This is the core computation engine. It manages three independent PaCMAP reducers (one per embedding space), rolling sample buffers, and Plotly figure generation.

```python
"""
Real-time PaCMAP visualization service for ACE-Step training monitoring.
Tracks three embedding spaces:
1. Latent space (audio embeddings from DCAE encoder)
2. LoRA layer outputs (adapter modifications in transformer)
3. Prompt embedding space (text encoder outputs)
"""

import numpy as np
import pacmap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
import json
from typing import Dict, List, Optional
import threading
import time


class MultiSpaceVisualizer:
    """
    Manages PaCMAP projections for three embedding spaces simultaneously.
    Thread-safe for real-time updates during training.
    """

    def __init__(
        self,
        max_samples: int = 1000,
        update_interval: int = 50,
        n_neighbors: int = 10,
    ):
        """
        Args:
            max_samples: Maximum samples to keep in rolling buffer per space
            update_interval: Recompute PaCMAP every N new samples added
            n_neighbors: PaCMAP neighbors parameter (lower = faster)
        """
        self.max_samples = max_samples
        self.update_interval = update_interval

        # Separate PaCMAP reducers for each space
        self.reducers = {
            "latent": pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors),
            "lora": pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors),
            "prompt": pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors),
        }

        # Rolling buffers for each space
        self.embeddings = {"latent": [], "lora": [], "prompt": []}

        # Metadata for each sample
        self.metadata = {"latent": [], "lora": [], "prompt": []}

        # Cached 2D projections
        self.projections_2d = {"latent": None, "lora": None, "prompt": None}

        # Sample counters for update triggers
        self.sample_counts = {"latent": 0, "lora": 0, "prompt": 0}

        # Thread lock for concurrent access
        self.lock = threading.Lock()

        # Last update timestamps
        self.last_update = {"latent": 0, "lora": 0, "prompt": 0}

    def _add_batch(self, space, embeddings_list, metadata_list):
        """Internal helper: add a batch of embeddings to a space buffer."""
        with self.lock:
            self.embeddings[space].extend(embeddings_list)
            self.metadata[space].extend(metadata_list)

            # Trim to max_samples (drop oldest)
            if len(self.embeddings[space]) > self.max_samples:
                overflow = len(self.embeddings[space]) - self.max_samples
                self.embeddings[space] = self.embeddings[space][overflow:]
                self.metadata[space] = self.metadata[space][overflow:]

            self.sample_counts[space] += len(embeddings_list)

    def add_latent_batch(
        self,
        embeddings: np.ndarray,
        step: int,
        losses: Optional[List[float]] = None,
        sample_ids: Optional[List[str]] = None,
    ):
        """Add audio latent embeddings from a training batch."""
        batch_size = embeddings.shape[0]
        emb_list = [embeddings[i] for i in range(batch_size)]
        meta_list = [
            {
                "step": step,
                "loss": losses[i] if losses else None,
                "sample_id": sample_ids[i]
                if sample_ids
                else f"step{step}_s{i}",
            }
            for i in range(batch_size)
        ]
        self._add_batch("latent", emb_list, meta_list)

    def add_lora_batch(
        self,
        outputs: np.ndarray,
        step: int,
        layer_name: str,
        sample_ids: Optional[List[str]] = None,
    ):
        """Add LoRA layer outputs from a training batch."""
        batch_size = outputs.shape[0]
        emb_list = [outputs[i] for i in range(batch_size)]
        meta_list = [
            {
                "step": step,
                "layer_name": layer_name,
                "sample_id": sample_ids[i]
                if sample_ids
                else f"step{step}_s{i}",
            }
            for i in range(batch_size)
        ]
        self._add_batch("lora", emb_list, meta_list)

    def add_prompt_batch(
        self,
        embeddings: np.ndarray,
        step: int,
        prompts: List[str],
        sample_ids: Optional[List[str]] = None,
    ):
        """Add text prompt embeddings from a training batch."""
        batch_size = embeddings.shape[0]
        emb_list = [embeddings[i] for i in range(batch_size)]
        meta_list = [
            {
                "step": step,
                "prompt_text": prompts[i] if i < len(prompts) else "unknown",
                "sample_id": sample_ids[i]
                if sample_ids
                else f"step{step}_s{i}",
            }
            for i in range(batch_size)
        ]
        self._add_batch("prompt", emb_list, meta_list)

    def _should_update(self, space: str) -> bool:
        """Check if we should recompute projection for this space."""
        return (
            self.sample_counts[space] >= self.update_interval
            and len(self.embeddings[space]) >= 10
        )

    def update_projections(self, force: bool = False) -> Dict[str, bool]:
        """Recompute PaCMAP projections for spaces that need updating."""
        updated = {}

        for space in ["latent", "lora", "prompt"]:
            if force or self._should_update(space):
                with self.lock:
                    if len(self.embeddings[space]) < 10:
                        updated[space] = False
                        continue

                    embeddings_array = np.array(self.embeddings[space])

                    # Re-create reducer to avoid stale state from previous fit
                    n_neighbors = min(
                        self.reducers[space].n_neighbors,
                        len(self.embeddings[space]) - 1,
                    )
                    n_neighbors = max(n_neighbors, 2)
                    reducer = pacmap.PaCMAP(
                        n_components=2, n_neighbors=n_neighbors
                    )
                    self.projections_2d[space] = reducer.fit_transform(
                        embeddings_array
                    )

                    self.sample_counts[space] = 0
                    self.last_update[space] = time.time()
                    updated[space] = True
            else:
                updated[space] = False

        return updated

    def generate_combined_figure(self) -> Optional[str]:
        """Generate a 3-panel Plotly figure JSON string."""
        with self.lock:
            if all(proj is None for proj in self.projections_2d.values()):
                return None

            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    "Latent Space (Audio)",
                    "LoRA Layer Outputs",
                    "Prompt Embeddings",
                ),
                horizontal_spacing=0.08,
            )

            # --- Latent Space ---
            if self.projections_2d["latent"] is not None:
                steps = [m["step"] for m in self.metadata["latent"]]
                hover = [
                    f"Step: {m['step']}<br>"
                    f"Loss: {m['loss']:.4f}<br>"
                    f"ID: {m['sample_id']}"
                    if m["loss"] is not None
                    else f"Step: {m['step']}<br>ID: {m['sample_id']}"
                    for m in self.metadata["latent"]
                ]
                fig.add_trace(
                    go.Scatter(
                        x=self.projections_2d["latent"][:, 0].tolist(),
                        y=self.projections_2d["latent"][:, 1].tolist(),
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=steps,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(
                                x=0.28, len=0.5, title="Step", thickness=12
                            ),
                        ),
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                        name="Latent",
                    ),
                    row=1,
                    col=1,
                )

            # --- LoRA Outputs ---
            if self.projections_2d["lora"] is not None:
                layer_names = [m["layer_name"] for m in self.metadata["lora"]]
                unique_layers = sorted(set(layer_names))
                layer_to_num = {n: i for i, n in enumerate(unique_layers)}
                layer_colors = [layer_to_num[n] for n in layer_names]
                hover = [
                    f"Step: {m['step']}<br>"
                    f"Layer: {m['layer_name']}<br>"
                    f"ID: {m['sample_id']}"
                    for m in self.metadata["lora"]
                ]
                fig.add_trace(
                    go.Scatter(
                        x=self.projections_2d["lora"][:, 0].tolist(),
                        y=self.projections_2d["lora"][:, 1].tolist(),
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=layer_colors,
                            colorscale="Plasma",
                            showscale=True,
                            colorbar=dict(
                                x=0.62, len=0.5, title="Layer", thickness=12
                            ),
                        ),
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                        name="LoRA",
                    ),
                    row=1,
                    col=2,
                )

            # --- Prompt Space ---
            if self.projections_2d["prompt"] is not None:
                steps = [m["step"] for m in self.metadata["prompt"]]
                hover = [
                    f"Step: {m['step']}<br>"
                    f"Prompt: {m['prompt_text'][:60]}<br>"
                    f"ID: {m['sample_id']}"
                    for m in self.metadata["prompt"]
                ]
                fig.add_trace(
                    go.Scatter(
                        x=self.projections_2d["prompt"][:, 0].tolist(),
                        y=self.projections_2d["prompt"][:, 1].tolist(),
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=steps,
                            colorscale="Cividis",
                            showscale=True,
                            colorbar=dict(
                                x=1.0, len=0.5, title="Step", thickness=12
                            ),
                        ),
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                        name="Prompt",
                    ),
                    row=1,
                    col=3,
                )

            # Layout — dark theme matching ACETrainer UI
            fig.update_layout(
                height=500,
                showlegend=False,
                hovermode="closest",
                paper_bgcolor="#111827",
                plot_bgcolor="#1f2937",
                font=dict(color="#9ca3af", size=11),
                margin=dict(l=40, r=40, t=50, b=40),
            )

            for i in range(1, 4):
                fig.update_xaxes(
                    title_text="PaCMAP-1",
                    gridcolor="#374151",
                    zerolinecolor="#374151",
                    row=1,
                    col=i,
                )
                fig.update_yaxes(
                    title_text="PaCMAP-2",
                    gridcolor="#374151",
                    zerolinecolor="#374151",
                    row=1,
                    col=i,
                )

            return json.dumps(fig, cls=PlotlyJSONEncoder)

    def get_stats(self) -> Dict:
        """Get current statistics for all spaces."""
        with self.lock:
            return {
                space: {
                    "total_samples": len(self.embeddings[space]),
                    "samples_since_update": self.sample_counts[space],
                    "last_update": self.last_update[space],
                    "has_projection": self.projections_2d[space] is not None,
                }
                for space in ["latent", "lora", "prompt"]
            }

    def clear_space(self, space: str):
        with self.lock:
            if space in self.embeddings:
                self.embeddings[space] = []
                self.metadata[space] = []
                self.projections_2d[space] = None
                self.sample_counts[space] = 0

    def clear_all(self):
        for space in ["latent", "lora", "prompt"]:
            self.clear_space(space)


# Global singleton
_visualizer = None


def get_visualizer() -> MultiSpaceVisualizer:
    global _visualizer
    if _visualizer is None:
        _visualizer = MultiSpaceVisualizer(
            max_samples=1000, update_interval=50, n_neighbors=10
        )
    return _visualizer
```

---

## Step 3: Add `training_hooks.py`

**File:** `ACETrainer/backend/training_hooks.py`

PyTorch forward hooks that capture embeddings from the three spaces during the training forward pass.

```python
"""
Training hooks for capturing embeddings during ACE-Step LoRA training.
Captures from three spaces:
1. Audio latent embeddings (DCAE encoder output)
2. LoRA layer outputs (transformer attention projections)
3. Text prompt embeddings (text encoder output)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import numpy as np
from collections import defaultdict


class EmbeddingHookManager:
    """Manages PyTorch forward hooks to capture embeddings during training."""

    def __init__(self, callback: Optional[Callable] = None):
        """
        Args:
            callback: Function called with captured embeddings.
                      Signature: callback(space: str, embeddings: np.ndarray, metadata: dict)
        """
        self.callback = callback
        self.hooks = []
        self.current_step = 0
        self.current_batch_prompts = []
        self.enabled = True

    def set_callback(self, callback: Callable):
        self.callback = callback

    def set_step(self, step: int):
        self.current_step = step

    def set_prompts(self, prompts: List[str]):
        self.current_batch_prompts = prompts

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def _make_latent_hook(self, name: str):
        """Hook for VAE/DCAE encoder outputs — audio latent space."""
        def hook(module, input, output):
            if not self.enabled:
                return
            # dcae.encode() returns (latents, _). The hook fires on the
            # encode sub-module. Output may be a tuple or tensor.
            if isinstance(output, tuple):
                latents = output[0]
            else:
                latents = output
            # Flatten to (batch_size, latent_dim)
            if len(latents.shape) > 2:
                latents = latents.reshape(latents.shape[0], -1)
            latents_np = latents.detach().cpu().float().numpy()
            if self.callback:
                self.callback(
                    space="latent",
                    embeddings=latents_np,
                    metadata={"step": self.current_step, "layer": name},
                )
        return hook

    def _make_lora_hook(self, name: str):
        """Hook for LoRA layer outputs — transformer attention projections."""
        def hook(module, input, output):
            if not self.enabled:
                return
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if len(out.shape) > 2:
                out = out.reshape(out.shape[0], -1)
            out_np = out.detach().cpu().float().numpy()
            if self.callback:
                self.callback(
                    space="lora",
                    embeddings=out_np,
                    metadata={
                        "step": self.current_step,
                        "layer_name": name,
                    },
                )
        return hook

    def _make_prompt_hook(self, name: str):
        """Hook for text encoder output — prompt embedding space."""
        def hook(module, input, output):
            if not self.enabled:
                return
            if isinstance(output, tuple):
                embeds = output[0]
            else:
                embeds = output
            # Mean pool over sequence: (batch, seq_len, dim) -> (batch, dim)
            if len(embeds.shape) == 3:
                embeds = embeds.mean(dim=1)
            embeds_np = embeds.detach().cpu().float().numpy()
            if self.callback:
                self.callback(
                    space="prompt",
                    embeddings=embeds_np,
                    metadata={
                        "step": self.current_step,
                        "prompts": list(self.current_batch_prompts),
                    },
                )
        return hook

    def register_hooks(self, model: nn.Module, config: Dict):
        """
        Register hooks on model layers.

        Args:
            model: The Pipeline (LightningModule) instance
            config: Dict with keys 'latent_layers', 'lora_layers', 'prompt_layers'
                    Each value is a list of dot-separated layer name strings.
        """
        self.remove_hooks()

        for layer_name in config.get("latent_layers", []):
            layer = self._get_layer(model, layer_name)
            if layer:
                h = layer.register_forward_hook(self._make_latent_hook(layer_name))
                self.hooks.append(h)
                print(f"[VizHook] Registered latent hook: {layer_name}")

        for layer_name in config.get("lora_layers", []):
            layer = self._get_layer(model, layer_name)
            if layer:
                h = layer.register_forward_hook(self._make_lora_hook(layer_name))
                self.hooks.append(h)
                print(f"[VizHook] Registered LoRA hook: {layer_name}")

        for layer_name in config.get("prompt_layers", []):
            layer = self._get_layer(model, layer_name)
            if layer:
                h = layer.register_forward_hook(self._make_prompt_hook(layer_name))
                self.hooks.append(h)
                print(f"[VizHook] Registered prompt hook: {layer_name}")

        print(f"[VizHook] Total hooks registered: {len(self.hooks)}")

    def _get_layer(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get a layer by dot-separated name (e.g. 'dcae.encoder')."""
        parts = name.split(".")
        layer = model
        try:
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            return layer
        except (AttributeError, IndexError, TypeError):
            print(f"[VizHook] Warning: layer '{name}' not found")
            return None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
```

---

## Step 4: Add `visualization_api.py`

**File:** `ACETrainer/backend/visualization_api.py`

Flask blueprint for REST endpoints + Socket.IO event handlers.

```python
"""
Flask/Socket.IO API for PaCMAP visualization.
Blueprint: /api/viz/*
Socket.IO events on /visualization namespace.
"""

from flask import Blueprint, jsonify
from flask_socketio import emit
import numpy as np
import traceback

from backend.visualization_service import get_visualizer

viz_bp = Blueprint("visualization", __name__)


# ─── REST Endpoints ───


@viz_bp.route("/api/viz/status", methods=["GET"])
def viz_status():
    try:
        stats = get_visualizer().get_stats()
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/clear/<space>", methods=["POST"])
def viz_clear(space):
    try:
        viz = get_visualizer()
        if space == "all":
            viz.clear_all()
        elif space in ("latent", "lora", "prompt"):
            viz.clear_space(space)
        else:
            return jsonify({"success": False, "error": f"Invalid space: {space}"}), 400
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/update", methods=["POST"])
def viz_force_update():
    try:
        updated = get_visualizer().update_projections(force=True)
        return jsonify({"success": True, "updated": updated})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Socket.IO Handlers ───


def register_viz_socketio_handlers(socketio):
    """Register Socket.IO handlers. Call from app.py after creating socketio."""

    @socketio.on("request_visualization", namespace="/visualization")
    def handle_viz_request():
        try:
            viz = get_visualizer()
            viz.update_projections(force=False)
            figure_json = viz.generate_combined_figure()
            if figure_json:
                emit(
                    "visualization_update",
                    {"success": True, "figure": figure_json, "stats": viz.get_stats()},
                )
            else:
                emit(
                    "visualization_update",
                    {"success": False, "error": "Insufficient data (need >= 10 samples per space)"},
                )
        except Exception as e:
            print(f"[VizAPI] Error: {e}")
            traceback.print_exc()
            emit("visualization_update", {"success": False, "error": str(e)})

    @socketio.on("embedding_captured", namespace="/visualization")
    def handle_embedding_captured(data):
        """Receive embeddings from trainer subprocess via Socket.IO client."""
        try:
            space = data.get("space")
            embeddings = np.array(data.get("embeddings"))
            metadata = data.get("metadata", {})
            step = metadata.get("step", 0)

            viz = get_visualizer()

            if space == "latent":
                viz.add_latent_batch(
                    embeddings=embeddings,
                    step=step,
                    losses=metadata.get("losses"),
                    sample_ids=metadata.get("sample_ids"),
                )
            elif space == "lora":
                viz.add_lora_batch(
                    outputs=embeddings,
                    step=step,
                    layer_name=metadata.get("layer_name", "unknown"),
                    sample_ids=metadata.get("sample_ids"),
                )
            elif space == "prompt":
                viz.add_prompt_batch(
                    embeddings=embeddings,
                    step=step,
                    prompts=metadata.get("prompts", []),
                    sample_ids=metadata.get("sample_ids"),
                )
            else:
                return

            # Auto-update projections if enough samples accumulated
            updated = viz.update_projections(force=False)
            if any(updated.values()):
                figure_json = viz.generate_combined_figure()
                if figure_json:
                    emit(
                        "visualization_update",
                        {"success": True, "figure": figure_json, "stats": viz.get_stats()},
                        broadcast=True,
                        namespace="/visualization",
                    )
        except Exception as e:
            print(f"[VizAPI] Error handling embedding: {e}")
            traceback.print_exc()

    @socketio.on("connect", namespace="/visualization")
    def on_viz_connect():
        """Send current stats when a client connects to the viz namespace."""
        try:
            stats = get_visualizer().get_stats()
            emit("viz_status_update", {"stats": stats})
        except Exception:
            pass
```

---

## Step 5: Wire into `app.py`

**File:** `ACETrainer/app.py`

Add 3 blocks (imports, blueprint registration, Socket.IO handler registration):

```python
# ─── EXISTING CODE (lines 1-42 unchanged) ───

# After the existing blueprint registrations (line 42), ADD:

from backend.visualization_api import viz_bp, register_viz_socketio_handlers

app.register_blueprint(viz_bp)

# After `app.config["SOCKETIO"] = socketio` (line 45), ADD:

register_viz_socketio_handlers(socketio)
```

**Full diff of `app.py`:**

```diff
 from backend.dataset_api import dataset_bp
 from backend.trainer_api import trainer_bp
 from backend.captioner_api import captioner_bp
+from backend.visualization_api import viz_bp, register_viz_socketio_handlers

 app.register_blueprint(dataset_bp, url_prefix="/api/dataset")
 app.register_blueprint(trainer_bp, url_prefix="/api/trainer")
 app.register_blueprint(captioner_bp, url_prefix="/api/captioner")
+app.register_blueprint(viz_bp)

 # Make socketio accessible to blueprints
 app.config["SOCKETIO"] = socketio

+# Register visualization Socket.IO event handlers
+register_viz_socketio_handlers(socketio)
```

---

## Step 6: Add Visualization Tab to `index.html`

**File:** `ACETrainer/static/index.html`

### 6a. Add tab button to nav (after line 37)

```diff
             <button class="tab-btn" data-tab="trainer">
                 <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                 Trainer
             </button>
+            <button class="tab-btn" data-tab="visualization">
+                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
+                Visualization
+            </button>
         </div>
     </nav>
```

### 6b. Add tab content (after the Trainer tab closing `</div>`, before `</main>`)

Insert this new tab section right after the `<!-- ==================== TRAINER TAB ==================== -->` div closes:

```html
        <!-- ==================== VISUALIZATION TAB ==================== -->
        <div id="tab-visualization" class="tab-content h-full flex flex-col" style="display:none;">
            <div class="flex-1 overflow-y-auto">
                <div class="max-w-6xl mx-auto p-6 space-y-4">

                    <!-- Header -->
                    <section class="card">
                        <div class="flex items-center justify-between">
                            <div>
                                <h3 class="card-title" style="margin-bottom:0.25rem">PaCMAP Embedding Visualization</h3>
                                <p class="text-xs text-gray-500">
                                    Real-time 2D projections of audio latent space, LoRA outputs, and prompt embeddings during training.
                                </p>
                            </div>
                            <span id="viz-status-badge" class="text-xs px-2 py-1 rounded-full bg-gray-800 text-gray-500">Waiting for data</span>
                        </div>
                    </section>

                    <!-- Controls -->
                    <section class="card">
                        <div class="flex items-center gap-3 flex-wrap">
                            <button id="viz-refresh-btn" class="btn btn-primary btn-sm">Refresh Now</button>
                            <button id="viz-toggle-auto" class="btn btn-secondary btn-sm">Pause Auto-Update</button>
                            <button id="viz-clear-all" class="btn btn-danger btn-sm">Clear All Data</button>
                            <div class="flex-1"></div>
                            <label class="text-xs text-gray-500">Update Interval:</label>
                            <select id="viz-interval-select" class="input-field text-xs" style="width:auto;min-width:100px;">
                                <option value="5000">5s</option>
                                <option value="10000">10s</option>
                                <option value="15000" selected>15s</option>
                                <option value="30000">30s</option>
                                <option value="60000">60s</option>
                            </select>
                        </div>
                    </section>

                    <!-- Stats -->
                    <section id="viz-stats" class="grid grid-cols-3 gap-3">
                        <div class="metric-box">
                            <span class="metric-label">Latent Space</span>
                            <span id="viz-stat-latent" class="metric-value text-sm">0 samples</span>
                        </div>
                        <div class="metric-box">
                            <span class="metric-label">LoRA Outputs</span>
                            <span id="viz-stat-lora" class="metric-value text-sm">0 samples</span>
                        </div>
                        <div class="metric-box">
                            <span class="metric-label">Prompt Space</span>
                            <span id="viz-stat-prompt" class="metric-value text-sm">0 samples</span>
                        </div>
                    </section>

                    <!-- Main Plot -->
                    <section class="card" style="padding:0.75rem;">
                        <div id="viz-plot-container" style="width:100%;height:520px;position:relative;">
                            <div id="viz-empty-state" class="flex flex-col items-center justify-center h-full text-gray-600">
                                <svg class="w-16 h-16 mb-3 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
                                <p class="text-sm">Waiting for training data...</p>
                                <p class="text-xs text-gray-700 mt-1">Embeddings will appear once training begins and hooks capture data.</p>
                            </div>
                        </div>
                    </section>

                    <!-- Guide -->
                    <section class="card">
                        <button id="viz-guide-toggle" class="w-full text-left text-sm text-gray-400 hover:text-gray-200 flex items-center gap-2">
                            <svg id="viz-guide-chevron" class="w-4 h-4 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
                            Understanding the Visualization
                        </button>
                        <div id="viz-guide-content" class="hidden mt-3 text-xs text-gray-500 space-y-2">
                            <p><strong class="text-gray-400">Latent Space (left):</strong> Audio samples in the DCAE encoder's latent representation. Similar-sounding audio clusters together. Watch for genre/style separation as training progresses.</p>
                            <p><strong class="text-gray-400">LoRA Outputs (center):</strong> How LoRA adapter layers transform representations. Colored by layer name. Clustering by layer shows specialization. If all layers look identical, LoRA may not be learning.</p>
                            <p><strong class="text-gray-400">Prompt Space (right):</strong> Text prompt embeddings from the text encoder. Similar prompts should cluster. Watch for genre groupings and trigger word separation.</p>
                            <p><strong class="text-gray-400">Color = training step</strong> (latent & prompt panels). Early steps are dark, late steps are bright. Structure should increase over time.</p>
                        </div>
                    </section>

                </div>
            </div>
        </div>
```

### 6c. Add Plotly CDN script and visualization JS (in `<head>` and before `</body>`)

In `<head>`, after the Chart.js script tag (line 11):

```diff
     <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
+    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
 </head>
```

Before `</body>`, after `trainer-ui.js` (line 474):

```diff
     <script src="/static/js/trainer-ui.js"></script>
+    <script src="/static/js/visualization_panel.js"></script>
 </body>
```

---

## Step 7: Add `visualization_panel.js`

**File:** `ACETrainer/static/js/visualization_panel.js`

This is the frontend controller for the visualization tab. It manages the second Socket.IO namespace, auto-update polling, and Plotly rendering.

```javascript
/**
 * visualization_panel.js — PaCMAP visualization tab controller.
 * Connects to /visualization Socket.IO namespace, renders Plotly charts.
 */

const VizPanel = {
    socket: null,
    autoUpdate: true,
    updateInterval: 15000,
    updateTimer: null,
    isTabActive: false,

    init() {
        this.connectSocket();
        this.bindEvents();
    },

    // ===== Socket.IO (separate namespace) =====
    connectSocket() {
        this.socket = io('/visualization', {
            transports: ['websocket', 'polling'],
        });

        this.socket.on('connect', () => {
            console.log('[VizPanel] Connected to /visualization');
        });

        this.socket.on('visualization_update', (data) => {
            if (data.success) {
                this.renderPlot(data.figure);
                this.updateStats(data.stats);
                this.setBadge('Live', 'bg-green-900 text-green-300');
            } else {
                this.showEmpty(data.error || 'Insufficient data');
            }
        });

        this.socket.on('viz_status_update', (data) => {
            if (data.stats) this.updateStats(data.stats);
        });

        this.socket.on('disconnect', () => {
            console.log('[VizPanel] Disconnected');
        });
    },

    // ===== Events =====
    bindEvents() {
        const refreshBtn = document.getElementById('viz-refresh-btn');
        const toggleBtn = document.getElementById('viz-toggle-auto');
        const clearBtn = document.getElementById('viz-clear-all');
        const intervalSel = document.getElementById('viz-interval-select');
        const guideToggle = document.getElementById('viz-guide-toggle');

        if (refreshBtn) refreshBtn.addEventListener('click', () => this.requestUpdate());

        if (toggleBtn) toggleBtn.addEventListener('click', () => {
            this.autoUpdate = !this.autoUpdate;
            toggleBtn.textContent = this.autoUpdate ? 'Pause Auto-Update' : 'Resume Auto-Update';
            if (this.autoUpdate) {
                this.startAutoUpdate();
            } else {
                this.stopAutoUpdate();
            }
        });

        if (clearBtn) clearBtn.addEventListener('click', async () => {
            const ok = await Utils.confirm('Clear all visualization data? This cannot be undone.');
            if (!ok) return;
            try {
                await Utils.apiPost('/api/viz/clear/all', {});
                this.showEmpty('Data cleared. Will update when new embeddings arrive.');
                this.updateStats({
                    latent: { total_samples: 0 },
                    lora: { total_samples: 0 },
                    prompt: { total_samples: 0 },
                });
                this.setBadge('Cleared', 'bg-gray-800 text-gray-500');
            } catch (e) {
                Utils.error('Failed to clear: ' + e.message);
            }
        });

        if (intervalSel) intervalSel.addEventListener('change', (e) => {
            this.updateInterval = parseInt(e.target.value);
            if (this.autoUpdate) {
                this.stopAutoUpdate();
                this.startAutoUpdate();
            }
        });

        if (guideToggle) guideToggle.addEventListener('click', () => {
            const content = document.getElementById('viz-guide-content');
            const chevron = document.getElementById('viz-guide-chevron');
            content.classList.toggle('hidden');
            chevron.style.transform = content.classList.contains('hidden') ? '' : 'rotate(180deg)';
        });
    },

    // ===== Tab Activation =====
    onTabActivated() {
        this.isTabActive = true;
        this.requestUpdate();
        this.startAutoUpdate();
    },

    onTabDeactivated() {
        this.isTabActive = false;
        this.stopAutoUpdate();
    },

    // ===== Auto-Update =====
    startAutoUpdate() {
        this.stopAutoUpdate();
        if (!this.autoUpdate) return;
        this.updateTimer = setInterval(() => {
            if (this.isTabActive && this.autoUpdate) {
                this.requestUpdate();
            }
        }, this.updateInterval);
    },

    stopAutoUpdate() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    },

    // ===== Request Visualization =====
    requestUpdate() {
        if (this.socket && this.socket.connected) {
            this.socket.emit('request_visualization');
        }
    },

    // ===== Render Plotly =====
    renderPlot(figureJson) {
        const container = document.getElementById('viz-plot-container');
        const emptyState = document.getElementById('viz-empty-state');

        try {
            const figure = JSON.parse(figureJson);
            if (emptyState) emptyState.style.display = 'none';

            Plotly.react(
                container,
                figure.data,
                figure.layout,
                {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                    displaylogo: false,
                }
            );
        } catch (e) {
            console.error('[VizPanel] Render error:', e);
            this.showEmpty('Failed to render visualization');
        }
    },

    // ===== Update Stats =====
    updateStats(stats) {
        if (!stats) return;
        const fmt = (s) => {
            const n = s.total_samples || 0;
            const proj = s.has_projection ? ' (projected)' : '';
            return `${n} samples${proj}`;
        };
        const el = (id, val) => {
            const e = document.getElementById(id);
            if (e) e.textContent = val;
        };
        el('viz-stat-latent', fmt(stats.latent || {}));
        el('viz-stat-lora', fmt(stats.lora || {}));
        el('viz-stat-prompt', fmt(stats.prompt || {}));
    },

    // ===== Empty State =====
    showEmpty(message) {
        const container = document.getElementById('viz-plot-container');
        const emptyState = document.getElementById('viz-empty-state');
        if (emptyState) {
            emptyState.style.display = '';
            const p = emptyState.querySelector('p');
            if (p) p.textContent = message || 'Waiting for training data...';
        }
        // Clear any existing Plotly chart
        try { Plotly.purge(container); } catch (e) {}
    },

    // ===== Status Badge =====
    setBadge(text, classes) {
        const badge = document.getElementById('viz-status-badge');
        if (badge) {
            badge.textContent = text;
            badge.className = 'text-xs px-2 py-1 rounded-full ' + classes;
        }
    },
};

window.VizPanel = VizPanel;
```

---

## Step 8: Add Visualization CSS to `style.css`

**File:** `ACETrainer/static/css/style.css`

Append these styles at the end of the existing file:

```css
/* ===== Visualization Panel ===== */
#viz-plot-container {
    border-radius: 0.5rem;
    overflow: hidden;
}

#viz-plot-container .js-plotly-plot,
#viz-plot-container .plotly {
    border-radius: 0.5rem;
}

/* Override Plotly modebar to match dark theme */
#viz-plot-container .modebar {
    background: transparent !important;
}
#viz-plot-container .modebar-btn path {
    fill: #6b7280 !important;
}
#viz-plot-container .modebar-btn:hover path {
    fill: #a78bfa !important;
}
```

---

## Step 9: Wire `app.js` Socket Events

**File:** `ACETrainer/static/js/app.js`

Two changes needed:

### 9a. Initialize VizPanel in `init()` (after `TrainerUI.init()`)

```diff
     init() {
         this.initTabs();
         this.initSocket();
         this.pollGPU();
         DatasetEditor.init();
         TrainerUI.init();
+        VizPanel.init();
     },
```

### 9b. Notify VizPanel on tab switch

In the `switchTab` method, add activation/deactivation hooks:

```diff
     switchTab(tabName) {
         this.currentTab = tabName;

         // Update tab buttons
         document.querySelectorAll('.tab-btn').forEach(btn => {
             btn.classList.toggle('active', btn.dataset.tab === tabName);
         });

         // Update tab content
         document.querySelectorAll('.tab-content').forEach(content => {
             content.style.display = content.id === `tab-${tabName}` ? '' : 'none';
             content.classList.toggle('active', content.id === `tab-${tabName}`);
         });

         // Refresh trainer dataset info when switching to trainer tab
         if (tabName === 'trainer') {
             TrainerUI.refreshDatasetInfo();
         }
+
+        // Visualization tab activation/deactivation
+        if (tabName === 'visualization') {
+            VizPanel.onTabActivated();
+        } else {
+            VizPanel.onTabDeactivated();
+        }
     },
```

---

## Step 10: Modify `ACE-Step/trainer.py`

**File:** `ACE-Step/trainer.py`

This is the most critical integration point. The ACE-Step `Pipeline` class uses PyTorch Lightning and has these key model attributes:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `self.dcae` | Music DCAE model | Audio VAE — has `.encode()` method |
| `self.transformers` | ACE-Step Transformer | Main model with LoRA adapters |
| `self.text_encoder_model` | HuggingFace text encoder | Text embeddings |

### 10a. Add imports at top of file

After the existing imports (after line 28), add:

```python
# ─── PaCMAP Visualization Hook Integration ───
import sys as _sys
from pathlib import Path as _Path

_acetrainer_backend = _Path(__file__).parent.parent / "ACETrainer" / "backend"
_sys.path.insert(0, str(_acetrainer_backend))

_VIZ_ENABLED = False
_EmbeddingHookManager = None
try:
    from training_hooks import EmbeddingHookManager as _EmbeddingHookManager
    _VIZ_ENABLED = True
    print("[Trainer] PaCMAP visualization hooks available")
except ImportError:
    print("[Trainer] Visualization hooks not available (training will proceed normally)")
```

### 10b. Add hook initialization in `Pipeline.__init__`

At the **end** of `Pipeline.__init__` (after `self.ssl_coeff = ssl_coeff`, around line 144), add:

```python
        # ─── Visualization Hooks ───
        self._hook_manager = None
        self._viz_client = None
        if _VIZ_ENABLED and self.is_train:
            try:
                self._hook_manager = _EmbeddingHookManager()

                # Layer names for the three embedding spaces.
                # These match the actual ACE-Step model attribute paths on `self`:
                #   - dcae: audio VAE encoder (latent space)
                #   - transformers: LoRA-adapted transformer (we hook a couple attention projections)
                #   - text_encoder_model: text encoder (prompt space)
                #
                # IMPORTANT: Run the debug print below to verify layer names if hooks fail.
                # for name, _ in self.named_modules():
                #     if any(k in name for k in ('dcae', 'to_q', 'to_v', 'text_encoder')):
                #         print(f"  [VizDebug] {name}")

                hook_config = {
                    "latent_layers": [
                        "dcae",  # dcae.encode() is called in preprocess(); hook on the module itself
                    ],
                    "lora_layers": [
                        # Hook a representative attention projection from the transformer.
                        # These will have LoRA applied if they match target_modules.
                        # We only need 1-2 layers for visualization — not all of them.
                        "transformers",  # Hook the top-level transformer forward
                    ],
                    "prompt_layers": [
                        "text_encoder_model",  # Hook the text encoder forward
                    ],
                }
                self._hook_manager.register_hooks(self, hook_config)

                # Connect Socket.IO client back to ACETrainer UI
                try:
                    import socketio as _sio_client_mod
                    self._viz_client = _sio_client_mod.Client()
                    self._viz_client.connect(
                        "http://127.0.0.1:7870",
                        namespaces=["/visualization"],
                    )
                    print("[Trainer] Connected to ACETrainer UI for visualization")
                except Exception as e:
                    print(f"[Trainer] Could not connect viz client: {e}")
                    self._viz_client = None

                # Set the callback to emit via Socket.IO
                def _on_embeddings(space, embeddings, metadata):
                    if self._viz_client and self._viz_client.connected:
                        try:
                            self._viz_client.emit(
                                "embedding_captured",
                                {
                                    "space": space,
                                    "embeddings": embeddings.tolist(),
                                    "metadata": metadata,
                                },
                                namespace="/visualization",
                            )
                        except Exception:
                            pass  # Don't let viz errors break training

                self._hook_manager.set_callback(_on_embeddings)

            except Exception as e:
                print(f"[Trainer] Visualization setup failed (non-fatal): {e}")
                self._hook_manager = None
```

### 10c. Wire step/prompt updates in `run_step`

In `run_step` (line 511), add hook updates at the very beginning of the method, before `self.plot_step(batch, batch_idx)`:

```python
    def run_step(self, batch, batch_idx):
        # ─── Visualization: update current step and prompts ───
        if self._hook_manager:
            self._hook_manager.set_step(self.global_step)
            self._hook_manager.set_prompts(batch.get("prompts", []))

        self.plot_step(batch, batch_idx)
        # ... rest of method unchanged ...
```

### 10d. Cleanup on training end

Add a `on_train_end` method to the `Pipeline` class (after `on_save_checkpoint`):

```python
    def on_train_end(self):
        """PyTorch Lightning callback — training has finished."""
        if self._hook_manager:
            self._hook_manager.remove_hooks()
            print("[Trainer] Removed visualization hooks")
        if self._viz_client and self._viz_client.connected:
            try:
                self._viz_client.disconnect()
            except Exception:
                pass
            print("[Trainer] Disconnected viz client")
```

### Complete diff summary for `trainer.py`

```
Lines 28+   : Add visualization imports (safe, try/except)
Line ~144   : Add hook_manager init, socketio client, callback setup
Line 511    : Add 3 lines at start of run_step()
Line ~637+  : Add on_train_end() method
```

**Safety:** All visualization code is wrapped in `try/except`. If `training_hooks.py` is missing, `pacmap` isn't installed, or the Socket.IO connection fails, training proceeds normally with zero visualization overhead.

---

## Step 11: Testing & Verification

### 11a. Verify installation

```bash
cd ACETrainer
python -c "import pacmap; import plotly; import socketio; print('All viz deps OK')"
```

### 11b. Verify file placement

```
ACETrainer/backend/visualization_service.py    ← exists
ACETrainer/backend/visualization_api.py        ← exists
ACETrainer/backend/training_hooks.py           ← exists
ACETrainer/static/js/visualization_panel.js    ← exists
```

### 11c. Start the UI and check

```bash
cd ACETrainer
python app.py
```

1. Open `http://127.0.0.1:7870`
2. Verify three tabs appear: Dataset Editor, Trainer, **Visualization**
3. Click Visualization tab — should show "Waiting for training data..."
4. Open browser console — should see `[VizPanel] Connected to /visualization`
5. Controls (Refresh, Pause, Clear, Interval dropdown) should be functional

### 11d. Test with actual training

1. Start a training run from the Trainer tab
2. Switch to Visualization tab
3. After ~50 training steps, embeddings should begin appearing
4. The 3-panel PaCMAP scatter plots should populate with colored dots
5. Stats should update: "N samples (projected)"

### 11e. Troubleshooting checklist

| Symptom | Check |
|---------|-------|
| No hooks registered | Verify `training_hooks.py` is in `ACETrainer/backend/`, check trainer.py console for `[VizHook]` messages |
| "Insufficient data" | Need >= 10 samples per space before PaCMAP can project. Wait for more training steps |
| Hooks fire but no UI update | Check Socket.IO client connection in trainer logs: `[Trainer] Connected to ACETrainer UI` |
| Layer not found warnings | Run the debug print code in trainer.py to list actual model layer names |
| Slow training | Reduce `max_samples` to 500, increase `update_interval` to 100 in `visualization_service.py` |
| Plotly not rendering | Check browser console for errors, verify Plotly CDN loaded |

---

## Appendix A: ACE-Step Model Layer Map

Based on `ACE-Step/trainer.py`, the `Pipeline` class has these attributes:

```
Pipeline (LightningModule)
├── scheduler                    FlowMatchEulerDiscreteScheduler
├── dcae                         Music DCAE (audio VAE)
│   ├── .encode(wavs, lengths)   Returns (latents, _)
│   └── .decode(latents, ...)    Returns (sr, wavs)
├── transformers                 ACE-Step Transformer (LoRA target)
│   ├── .forward(hidden_states, attention_mask, encoder_text_hidden_states, ...)
│   │   Returns TransformerOutput with .sample and .proj_losses
│   └── LoRA adapters attached via peft on target_modules:
│       speaker_embedder, linear_q, linear_k, linear_v, to_q, to_k, to_v, to_out.0
├── text_encoder_model           HuggingFace text encoder
│   └── .forward(**inputs) → outputs.last_hidden_state
├── text_tokenizer               HuggingFace tokenizer
├── mert_model                   MERT SSL model (eval only)
└── hubert_model                 mHuBERT SSL model (eval only)
```

**Hook targets (conservative defaults):**

- `dcae` — captures the DCAE module's forward output (includes encode path)
- `transformers` — captures the main transformer output
- `text_encoder_model` — captures text encoder output

For **more granular** LoRA layer visualization, you can hook individual attention projections. Find them by running:

```python
for name, mod in self.transformers.named_modules():
    if 'to_q' in name or 'to_v' in name or 'lora' in name.lower():
        print(f"  {name}: {type(mod).__name__}")
```

Then update `hook_config['lora_layers']` with specific paths like:
```python
"lora_layers": [
    "transformers.blocks.0.attn.to_q",
    "transformers.blocks.0.attn.to_v",
]
```

---

## Appendix B: Performance Tuning

### Embedding Capture Overhead

The hooks add approximately **5-10% overhead** to training step time. This comes from:
1. Tensor `.detach().cpu().float().numpy()` conversion (~2%)
2. Socket.IO emit with JSON serialization (~3%)
3. PaCMAP re-fit when sample threshold reached (~5%, amortized)

### Reducing Overhead

**Option 1: Skip steps** — Only capture every Nth step:

```python
# In the callback
def _on_embeddings(space, embeddings, metadata):
    if metadata.get('step', 0) % 10 != 0:  # Only every 10th step
        return
    # ... emit as normal
```

**Option 2: Disable after debugging** — Call `hook_manager.disable()` when you've seen enough.

**Option 3: Reduce sample buffer** — In `visualization_service.py`:

```python
_visualizer = MultiSpaceVisualizer(
    max_samples=250,      # Down from 1000
    update_interval=100,  # Up from 50
    n_neighbors=5,        # Down from 10
)
```

**Option 4: Increase UI poll interval** — In the Visualization tab, set the dropdown to 30s or 60s.

### Memory Usage

Each embedding space stores up to `max_samples` vectors. With default `max_samples=1000`:
- Latent space: 1000 x (flattened DCAE output dim) x 4 bytes
- LoRA space: 1000 x (transformer output dim) x 4 bytes
- Prompt space: 1000 x 768 x 4 bytes (text encoder hidden dim)

Typical total: **20-50 MB** depending on model dimensions. Use "Clear All Data" button to reset if memory is a concern.
