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

        # Snapshot saving
        self.save_dir = None       # Set by trainer or API when training starts
        self.auto_save = True      # Whether to auto-save on projection update

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

        # Auto-save if any space was updated and save_dir is set
        if self.auto_save and self.save_dir and any(updated.values()):
            try:
                latest_step = 0
                for space in ["latent", "lora", "prompt"]:
                    if self.metadata[space]:
                        latest_step = max(latest_step, self.metadata[space][-1].get("step", 0))
                self.save_snapshot(self.save_dir, latest_step)
            except Exception as e:
                print(f"[VizService] Auto-save failed (non-fatal): {e}")

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

    def set_save_dir(self, path: str):
        """Set the directory where snapshots will be saved."""
        self.save_dir = path
        print(f"[VizService] Snapshot save directory: {path}")

    def save_snapshot(self, save_dir: str, step: int) -> Dict[str, str]:
        """
        Persist the current visualization state to disk.

        Saves:
          - Interactive HTML (always)
          - Static PNG (if kaleido installed) -- these become animation frames
          - Raw embeddings + projections as .npz (always)

        Returns:
            Dict of {format: filepath} for files successfully written.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        saved = {}

        # --- 1. Interactive HTML ---
        figure_json = self.generate_combined_figure()
        if figure_json:
            import plotly.io as pio
            fig = pio.from_json(figure_json)
            html_path = os.path.join(save_dir, f"viz_step_{step}_pacmap.html")
            fig.write_html(html_path, include_plotlyjs="cdn")
            saved["html"] = html_path

            # Also save a "latest" copy for quick access
            latest_path = os.path.join(save_dir, "viz_latest_pacmap.html")
            fig.write_html(latest_path, include_plotlyjs="cdn")

            # --- 2. Static PNG (optional, needs kaleido) -- animation frame ---
            try:
                png_path = os.path.join(save_dir, f"viz_step_{step}_pacmap.png")
                fig.write_image(png_path, width=1800, height=500, scale=2)
                saved["png"] = png_path

                latest_png = os.path.join(save_dir, "viz_latest_pacmap.png")
                fig.write_image(latest_png, width=1800, height=500, scale=2)
            except (ValueError, ImportError, OSError) as e:
                print(f"[VizService] PNG export skipped (kaleido not available): {e}")

        # --- 3. Raw embeddings + projections as NPZ ---
        with self.lock:
            npz_data = {}
            for space in ["latent", "lora", "prompt"]:
                if len(self.embeddings[space]) > 0:
                    npz_data[f"{space}_embeddings"] = np.array(self.embeddings[space])
                    npz_data[f"{space}_metadata"] = np.array(
                        [json.dumps(m) for m in self.metadata[space]], dtype=object
                    )
                if self.projections_2d[space] is not None:
                    npz_data[f"{space}_projections_2d"] = self.projections_2d[space]

            if npz_data:
                npz_path = os.path.join(save_dir, f"viz_step_{step}_embeddings.npz")
                np.savez_compressed(npz_path, **npz_data)
                saved["npz"] = npz_path

        return saved

    def create_animation(self, save_dir: str) -> Dict[str, str]:
        """
        Stitch all saved per-step PNGs into animated GIF and WebM files.

        Called at training end. Scans save_dir for viz_step_*_pacmap.png,
        sorts by step number, assembles into animations.

        Returns:
            Dict of {format: filepath} for animations successfully created.
        """
        import os
        import glob
        import re

        created = {}

        # Find all per-step PNGs, sorted by step number
        pattern = os.path.join(save_dir, "viz_step_*_pacmap.png")
        png_files = glob.glob(pattern)

        # Exclude the "latest" file
        png_files = [f for f in png_files if "latest" not in os.path.basename(f)]

        if len(png_files) < 2:
            print(f"[VizService] Not enough frames for animation ({len(png_files)} PNGs found)")
            return created

        # Sort by step number extracted from filename
        def extract_step(path):
            match = re.search(r"viz_step_(\d+)_pacmap\.png", os.path.basename(path))
            return int(match.group(1)) if match else 0

        png_files.sort(key=extract_step)
        print(f"[VizService] Assembling animation from {len(png_files)} frames...")

        # --- 1. Animated GIF via Pillow ---
        try:
            from PIL import Image

            frames = []
            for png_path in png_files:
                img = Image.open(png_path)
                # Convert to RGB (Plotly PNGs may have alpha)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                frames.append(img)

            if frames:
                gif_path = os.path.join(save_dir, "pacmap_evolution.gif")

                # Adaptive frame duration: target ~15-30 seconds total playback
                total_target_ms = max(15000, min(30000, len(frames) * 500))
                frame_duration_ms = total_target_ms // len(frames)
                frame_duration_ms = max(100, min(2000, frame_duration_ms))

                # Hold last frame longer so it's visible
                durations = [frame_duration_ms] * len(frames)
                durations[-1] = frame_duration_ms * 3

                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=durations,
                    loop=0,  # infinite loop
                    optimize=True,
                )
                created["gif"] = gif_path
                gif_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
                print(f"[VizService] GIF saved: {gif_path} ({gif_size_mb:.1f} MB, {len(frames)} frames)")

        except Exception as e:
            print(f"[VizService] GIF creation failed (non-fatal): {e}")

        # --- 2. WebM via ffmpeg (better quality, smaller file) ---
        try:
            import subprocess
            import shutil

            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                webm_path = os.path.join(save_dir, "pacmap_evolution.webm")

                # Create a temporary file list for ffmpeg (handles arbitrary step numbers)
                concat_list_path = os.path.join(save_dir, "_ffmpeg_framelist.txt")
                # Adaptive framerate: target ~15-30 second playback
                fps = max(1, min(10, len(png_files) // 15))
                if fps < 1:
                    fps = 1

                with open(concat_list_path, "w") as f:
                    for png_path in png_files:
                        # Use forward slashes for ffmpeg compatibility
                        safe_path = png_path.replace("\\", "/")
                        f.write(f"file '{safe_path}'\n")
                        f.write(f"duration {1.0 / fps:.4f}\n")
                    # Repeat last frame (ffmpeg concat demuxer quirk)
                    safe_path = png_files[-1].replace("\\", "/")
                    f.write(f"file '{safe_path}'\n")
                    f.write(f"duration {3.0 / fps:.4f}\n")

                cmd = [
                    ffmpeg_path,
                    "-y",                          # overwrite
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c:v", "libvpx-vp9",          # VP9 codec for WebM
                    "-b:v", "2M",                   # bitrate
                    "-pix_fmt", "yuv420p",
                    "-an",                          # no audio
                    webm_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                # Clean up temp file
                try:
                    os.remove(concat_list_path)
                except OSError:
                    pass

                if result.returncode == 0 and os.path.exists(webm_path):
                    created["webm"] = webm_path
                    webm_size_mb = os.path.getsize(webm_path) / (1024 * 1024)
                    print(f"[VizService] WebM saved: {webm_path} ({webm_size_mb:.1f} MB, {len(png_files)} frames)")
                else:
                    print(f"[VizService] ffmpeg WebM encoding failed: {result.stderr[:500]}")
            else:
                print("[VizService] ffmpeg not found on PATH, skipping WebM")

        except Exception as e:
            print(f"[VizService] WebM creation failed (non-fatal): {e}")

        return created

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
