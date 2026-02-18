# Visualization Snapshot Saving & Animated Timeline — Implementation Plan

> **Goal:** Automatically persist PaCMAP visualization snapshots during training AND stitch them into an animated GIF/WebM timeline when training completes — so you get a visual movie of how embeddings evolved across the entire run.

---

## Current State (The Gap)

| Artifact | Storage | Persists to Disk? |
|----------|---------|:-:|
| Eval audio (target + predicted WAVs) | `eval_results/step_N/` | Yes |
| LoRA adapter checkpoints | `checkpoints/epoch=N-step=N_lora/` | Yes |
| TensorBoard scalar logs | `events.out.tfevents.*` | Yes |
| PaCMAP Plotly figures | In-memory only, emitted over Socket.IO | **No** |
| Raw embedding buffers (3 spaces) | In-memory `MultiSpaceVisualizer` | **No** |
| 2D PaCMAP projections | In-memory `MultiSpaceVisualizer` | **No** |
| Animated embedding evolution | Does not exist | **No** |

**Nothing from the visualization pipeline is ever written to disk.** When the Flask server restarts or the browser tab closes, all embedding history and rendered charts are lost permanently.

---

## What We Will Save

### Per-Update Snapshots (during training)

Each time projections are updated (every `update_interval` samples, currently 50), we persist **three artifacts**:

| File | Format | Purpose | Dependencies |
|------|--------|---------|:--:|
| `viz_step_{N}_pacmap.html` | Interactive Plotly HTML | Fully self-contained, openable in any browser, zoomable/hoverable | None (`fig.write_html()`) |
| `viz_step_{N}_pacmap.png` | Static PNG image | Quick visual reference + **animation frames** | `kaleido` (install) |
| `viz_step_{N}_embeddings.npz` | NumPy compressed archive | Raw embeddings + projections + metadata for offline re-analysis | None (`np.savez_compressed()`) |

### Animated Timeline (at training end)

When training completes, the saved PNGs are stitched into an animated timeline:

| File | Format | Purpose | Dependencies |
|------|--------|---------|:--:|
| `pacmap_evolution.gif` | Animated GIF | Universally viewable, embeddable in reports/Discord/GitHub, no player needed | `Pillow` (already installed) |
| `pacmap_evolution.webm` | WebM video | Smaller file size, higher quality, playable in all modern browsers | `ffmpeg` (already on system PATH) |

Both are saved into the same `viz_snapshots/` directory alongside the per-step PNGs.

### Final Directory Structure

```
workspace/exps/logs/lightning_logs/<run_name>/
├── checkpoints/
├── eval_results/
│   ├── step_0/
│   │   ├── target_wav_*.wav          (existing)
│   │   ├── pred_wav_*.wav            (existing)
│   │   └── key_prompt_lyric_*.txt    (existing)
│   └── step_1000/
├── viz_snapshots/                     NEW
│   ├── viz_step_50_pacmap.png         frame 1
│   ├── viz_step_50_pacmap.html
│   ├── viz_step_50_embeddings.npz
│   ├── viz_step_100_pacmap.png        frame 2
│   ├── viz_step_100_pacmap.html
│   ├── viz_step_100_embeddings.npz
│   ├── ...                            ...more frames...
│   ├── viz_step_5000_pacmap.png       final frame
│   ├── viz_step_5000_pacmap.html
│   ├── viz_step_5000_embeddings.npz
│   ├── viz_latest_pacmap.html         (always-overwritten for quick access)
│   ├── viz_latest_pacmap.png
│   ├── pacmap_evolution.gif           ANIMATED — full training timeline
│   └── pacmap_evolution.webm          ANIMATED — smaller, higher quality
├── events.out.tfevents.*
└── hparams.yaml
```

---

## Available Tools (already installed, zero new dependencies for animation)

| Tool | Version | Use For |
|------|---------|---------|
| Pillow | 12.0.0 | GIF assembly — `Image.save(save_all=True, append_images=[...])` |
| ffmpeg | 5.1 (system) | WebM encoding — `ffmpeg -framerate 4 -pattern_type glob -i '*.png' -c:v libvpx-vp9 out.webm` |
| kaleido | **not installed** | PNG export from Plotly — `fig.write_image()` — **must install** |

---

## Implementation Steps

### Step 1: Install `kaleido` for PNG Export

```bash
pip install kaleido
```

Without kaleido, `fig.write_image()` raises an error and **no PNGs = no animation frames**. This is the one required new dependency. If kaleido is unavailable on a given system, the code gracefully skips PNG export (and animation) and only saves HTML + NPZ.

---

### Step 2: Add Snapshot Saving to `visualization_service.py`

Add a `save_snapshot()` method to `MultiSpaceVisualizer` and state for save directory tracking.

**New attributes in `__init__`:**

```python
self.save_dir = None       # Set by trainer or API when training starts
self.auto_save = True      # Whether to auto-save on projection update
```

**New method `set_save_dir`:**

```python
def set_save_dir(self, path: str):
    """Set the directory where snapshots will be saved."""
    self.save_dir = path
    print(f"[VizService] Snapshot save directory: {path}")
```

**New method `save_snapshot`:**

```python
def save_snapshot(self, save_dir: str, step: int) -> Dict[str, str]:
    """
    Persist the current visualization state to disk.

    Saves:
      - Interactive HTML (always)
      - Static PNG (if kaleido installed) — these become animation frames
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

        # --- 2. Static PNG (optional, needs kaleido) — animation frame ---
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
```

---

### Step 3: Add Animated Timeline Assembly to `visualization_service.py`

New standalone method on `MultiSpaceVisualizer`:

```python
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

    if len(png_files) < 2:
        print(f"[VizService] Not enough frames for animation ({len(png_files)} PNGs found)")
        return created

    # Sort by step number extracted from filename
    def extract_step(path):
        match = re.search(r'viz_step_(\d+)_pacmap\.png', os.path.basename(path))
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
            if img.mode != 'RGB':
                img = img.convert('RGB')
            frames.append(img)

        if frames:
            gif_path = os.path.join(save_dir, "pacmap_evolution.gif")

            # Adaptive frame duration: faster for many frames, slower for few
            # Target ~15-30 seconds total playback
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

            with open(concat_list_path, "w") as f:
                for png_path in png_files:
                    # Duration per frame in seconds
                    f.write(f"file '{png_path}'\n")
                    f.write(f"duration {1.0 / fps:.4f}\n")
                # Repeat last frame (ffmpeg concat demuxer quirk)
                f.write(f"file '{png_files[-1]}'\n")
                f.write(f"duration {3.0 / fps:.4f}\n")

            cmd = [
                ffmpeg_path,
                "-y",                          # overwrite
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c:v", "libvpx-vp9",          # VP9 codec for WebM
                "-b:v", "2M",                   # bitrate
                "-pix_fmt", "yuva420p",
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
```

---

### Step 4: Auto-Save on Projection Update

Modify `update_projections()` in `MultiSpaceVisualizer` to call `save_snapshot()` after successful PaCMAP recomputation:

```python
def update_projections(self, force: bool = False) -> Dict[str, bool]:
    """Recompute PaCMAP projections for spaces that need updating."""
    updated = {}
    # ... existing per-space PaCMAP fit logic (unchanged) ...

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
```

---

### Step 5: Add REST Endpoints for Save, Save Dir, and Animation

Add to `visualization_api.py`:

```python
@viz_bp.route("/api/viz/save", methods=["POST"])
def viz_save():
    """Manually trigger a snapshot save."""
    try:
        viz = get_visualizer()
        if not viz.save_dir:
            return jsonify({"success": False, "error": "No save directory configured. Start training first."}), 400
        latest_step = 0
        for space in ["latent", "lora", "prompt"]:
            if viz.metadata[space]:
                latest_step = max(latest_step, viz.metadata[space][-1].get("step", 0))
        saved = viz.save_snapshot(viz.save_dir, latest_step)
        return jsonify({"success": True, "saved": saved})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/set_save_dir", methods=["POST"])
def viz_set_save_dir():
    """Set the snapshot save directory (called by trainer_service when training starts)."""
    from flask import request
    data = request.get_json() or {}
    save_dir = data.get("save_dir")
    if not save_dir:
        return jsonify({"success": False, "error": "save_dir required"}), 400
    get_visualizer().set_save_dir(save_dir)
    return jsonify({"success": True})


@viz_bp.route("/api/viz/create_animation", methods=["POST"])
def viz_create_animation():
    """Stitch saved PNGs into animated GIF + WebM. Called at training end or manually."""
    try:
        viz = get_visualizer()
        if not viz.save_dir:
            return jsonify({"success": False, "error": "No save directory configured."}), 400
        created = viz.create_animation(viz.save_dir)
        if not created:
            return jsonify({"success": False, "error": "No animation created (need >= 2 PNG frames)."}), 400
        return jsonify({"success": True, "created": created})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/toggle_auto_save", methods=["POST"])
def viz_toggle_auto_save():
    """Toggle auto-save on/off."""
    from flask import request
    data = request.get_json() or {}
    enabled = data.get("enabled", True)
    get_visualizer().auto_save = bool(enabled)
    return jsonify({"success": True, "auto_save": get_visualizer().auto_save})
```

---

### Step 6: Wire Save Directory from `trainer_service.py`

When `trainer_service.py` starts a training run, it constructs the log directory path. After launching the subprocess, set the viz save_dir:

```python
# In trainer_service.py start_training(), after subprocess launch:
viz_save_dir = os.path.join(log_dir, "viz_snapshots")
try:
    import requests
    requests.post("http://127.0.0.1:7870/api/viz/set_save_dir",
                  json={"save_dir": viz_save_dir}, timeout=2)
except Exception:
    pass  # Non-fatal
```

---

### Step 7: Trigger Animation Assembly on Training End

Two places to trigger the animation stitch:

**7a. Server-side: in `visualization_api.py` Socket.IO handler**

When the training process completes, `trainer_service.py` emits a `training_complete` event. Add a listener that triggers animation assembly:

```python
# In register_viz_socketio_handlers(socketio):
@socketio.on("training_complete", namespace="/training")
def on_training_complete_viz(data):
    """When training finishes, assemble the animation from saved PNGs."""
    try:
        viz = get_visualizer()
        if viz.save_dir:
            created = viz.create_animation(viz.save_dir)
            if created:
                socketio.emit(
                    "animation_ready",
                    {"success": True, "created": created},
                    namespace="/visualization",
                )
    except Exception as e:
        print(f"[VizAPI] Animation assembly failed: {e}")
```

**7b. Alternatively: listen in `app.py` on the existing training_complete handler**

Since `trainer_service.py` already emits `training_complete` on the `/training` namespace, we can add a cross-namespace trigger in `app.py`:

```python
@socketio.on("training_complete", namespace="/training")
def on_training_complete_trigger_animation(data):
    from backend.visualization_service import get_visualizer
    viz = get_visualizer()
    if viz.save_dir:
        try:
            created = viz.create_animation(viz.save_dir)
            if created:
                socketio.emit("animation_ready", {"created": created}, namespace="/visualization")
                print(f"[App] Animation created: {list(created.keys())}")
        except Exception as e:
            print(f"[App] Animation creation failed (non-fatal): {e}")
```

---

### Step 8: Add UI Controls to `index.html` and `visualization_panel.js`

**8a. HTML — Add buttons and auto-save toggle in the Visualization tab controls section:**

```html
<!-- In the controls <section> of the Visualization tab, after the Clear All button -->
<button id="viz-save-snapshot" class="btn btn-secondary btn-sm">Save Snapshot</button>
<button id="viz-create-animation" class="btn btn-secondary btn-sm">Create Animation</button>

<!-- In the flex-1 spacer area, before the interval selector -->
<label class="flex items-center gap-1 text-xs text-gray-500">
    <input id="viz-auto-save" type="checkbox" checked class="w-3 h-3"> Auto-save
</label>
```

**8b. JavaScript — Wire event handlers in `visualization_panel.js`:**

```javascript
// In bindEvents(), add:

const saveBtn = document.getElementById('viz-save-snapshot');
if (saveBtn) saveBtn.addEventListener('click', async () => {
    try {
        const result = await Utils.apiPost('/api/viz/save', {});
        if (result.success) {
            Utils.success('Snapshot saved: ' + Object.keys(result.saved).join(', '));
        } else {
            Utils.error(result.error || 'Save failed');
        }
    } catch (e) {
        Utils.error('Failed to save snapshot: ' + e.message);
    }
});

const animBtn = document.getElementById('viz-create-animation');
if (animBtn) animBtn.addEventListener('click', async () => {
    animBtn.disabled = true;
    animBtn.textContent = 'Creating...';
    try {
        const result = await Utils.apiPost('/api/viz/create_animation', {});
        if (result.success) {
            const formats = Object.keys(result.created).join(', ');
            Utils.success('Animation created: ' + formats);
        } else {
            Utils.error(result.error || 'Animation failed');
        }
    } catch (e) {
        Utils.error('Failed to create animation: ' + e.message);
    } finally {
        animBtn.disabled = false;
        animBtn.textContent = 'Create Animation';
    }
});

const autoSaveCheck = document.getElementById('viz-auto-save');
if (autoSaveCheck) autoSaveCheck.addEventListener('change', async () => {
    try {
        await Utils.apiPost('/api/viz/toggle_auto_save', { enabled: autoSaveCheck.checked });
    } catch (e) {}
});
```

**8c. Listen for `animation_ready` event (auto-notification when training ends):**

```javascript
// In connectSocket(), add:
this.socket.on('animation_ready', (data) => {
    if (data.created) {
        const formats = Object.keys(data.created).join(' + ');
        Utils.success(`Training animation ready! (${formats})`);
        this.setBadge('Animation Ready', 'bg-purple-900 text-purple-300');
    }
});
```

---

### Step 9: Testing & Verification

1. **Install kaleido:** `pip install kaleido`
2. **Start ACETrainer:** `python app.py`
3. **Start a training run** from the Trainer tab
4. **Switch to Visualization tab** and wait for embeddings to accumulate
5. **Verify auto-save:** Check `workspace/exps/logs/lightning_logs/<run>/viz_snapshots/` for:
   - `viz_step_N_pacmap.html` — open in browser, confirm interactive plots work
   - `viz_step_N_pacmap.png` — open in image viewer, confirm 3-panel chart renders
   - `viz_step_N_embeddings.npz` — load in Python: `data = np.load('file.npz', allow_pickle=True)`
   - `viz_latest_pacmap.html` — should be the most recent snapshot
6. **Test manual save:** Click "Save Snapshot" button, verify toast notification and file creation
7. **Let training complete** — verify:
   - `pacmap_evolution.gif` exists and plays as an animated loop in any image viewer
   - `pacmap_evolution.webm` exists and plays in browser / VLC / media player
   - Toast notification appears: "Training animation ready! (gif + webm)"
8. **Test manual animation:** Click "Create Animation" button mid-training, verify it assembles from available frames
9. **Test without kaleido:** Uninstall kaleido, verify HTML+NPZ still save, PNG+animation gracefully skipped

---

## Summary of Changes

| File | Change Type | Description |
|------|:-----------:|-------------|
| `backend/visualization_service.py` | **Modify** | Add `save_snapshot()`, `create_animation()`, `set_save_dir()`, `auto_save` flag, auto-save in `update_projections()` |
| `backend/visualization_api.py` | **Modify** | Add `POST /api/viz/save`, `/api/viz/set_save_dir`, `/api/viz/create_animation`, `/api/viz/toggle_auto_save` endpoints |
| `backend/trainer_service.py` | **Modify** | Set viz save_dir when training starts |
| `app.py` | **Modify** | Add `training_complete` listener that triggers animation assembly |
| `static/index.html` | **Modify** | Add "Save Snapshot", "Create Animation" buttons + "Auto-save" checkbox |
| `static/js/visualization_panel.js` | **Modify** | Wire save, animation, auto-save handlers + `animation_ready` listener |
| (system) | **Install** | `pip install kaleido` for PNG export |

**No new files needed.** All changes extend existing visualization infrastructure.

**Safety:** All save and animation operations are wrapped in try/except. Failure to save/animate never interrupts training or the real-time visualization pipeline. PNG export and animation gracefully degrade if kaleido or ffmpeg are missing.

**Dependencies for animation (all already installed):**
- `Pillow 12.0.0` — GIF assembly
- `ffmpeg 5.1` (system PATH) — WebM encoding
- `kaleido` — **must install** — PNG frame export from Plotly
