# Code Review Action Plan

Items selected from `ACETrainer-code-review.md` — all critical fixes, all high-reward/low-risk significant fixes, and the trivial minor wins. Each step includes the exact code change required.

---

## Task List

| # | Review ID | Description | Risk | Files |
|---|-----------|-------------|------|-------|
| 1 | C2 | Fix path traversal in `serve_audio` | Critical | `backend/dataset_api.py` |
| 2 | C3 | Sanitize upload filenames | Critical | `backend/dataset_api.py` |
| 3 | C1 | Add thread locks to `_batch_state` and `_convert_state` | Critical | `backend/captioner/service.py`, `backend/dataset_api.py` |
| 4 | S5 | Add `atexit` cleanup for TensorBoard subprocess | Significant | `backend/trainer_api.py` |
| 5 | S3 | Replace `time.sleep` with `eventlet.sleep` in `_poll_gpu` | Significant | `backend/trainer_service.py` |
| 6 | S1 | Add cache version check to `ImageBindEmbedder` | Significant | `backend/captioner/embedder.py` |
| 7 | S2 | Remove dead `num_workers` field from trainer UI | Significant | `static/index.html` |
| 8 | M5 | Replace DOM-based `escHtml` with string replacement | Minor | `static/js/dataset-editor.js` |
| 9 | M7 | Direct captioner temp files into workspace | Minor | `backend/captioner_api.py` |
| 10 | M1 | Delete phantom files from project root | Minor | root directory |

**Excluded from this plan** (not worth the risk/effort ratio):
- **S4** (CORS restriction) — Localhost-only app, restricting origins could break dev workflows. Document rather than change.
- **M2** (phase2/ directory) — Needs user decision on archiving vs deletion. Not a code change.
- **M3** (SRI hashes) — Requires fetching current hashes for each CDN resource. Low payoff for localhost tool.
- **M4** (over-reading validation) — The full content is used by the frontend carousel. Optimizing requires API restructuring beyond a quick fix.
- **M6** (PaCMAP re-creation) — Documented as a known limitation, not actionable.

---

## Step 1 — Fix Path Traversal in `serve_audio` (C2)

**File**: `backend/dataset_api.py`

Replace `send_file` with `send_from_directory` which has built-in traversal protection:

```python
# BEFORE (lines 250-257):
@dataset_bp.route("/audio/<stem>", methods=["GET"])
def serve_audio(stem):
    """Serve an audio file for playback in the browser."""
    data_dir = ds.get_data_dir()
    mp3_path = os.path.join(data_dir, f"{stem}.mp3")
    if not os.path.exists(mp3_path):
        return jsonify({"error": "Audio file not found"}), 404
    return send_file(mp3_path, mimetype="audio/mpeg")

# AFTER:
@dataset_bp.route("/audio/<stem>", methods=["GET"])
def serve_audio(stem):
    """Serve an audio file for playback in the browser."""
    data_dir = ds.get_data_dir()
    try:
        return send_from_directory(data_dir, f"{stem}.mp3", mimetype="audio/mpeg")
    except FileNotFoundError:
        return jsonify({"error": "Audio file not found"}), 404
```

Also add `send_from_directory` to the import line:

```python
# BEFORE:
from flask import Blueprint, request, jsonify, current_app, send_file

# AFTER:
from flask import Blueprint, request, jsonify, current_app, send_from_directory
```

---

## Step 2 — Sanitize Upload Filenames (C3)

**File**: `backend/dataset_api.py`

Add `secure_filename` import and apply to the `upload_files` endpoint:

```python
# Add to imports:
from werkzeug.utils import secure_filename

# BEFORE (lines 42-47):
    for key in request.files:
        file_obj = request.files[key]
        if file_obj.filename:
            dest = os.path.join(data_dir, file_obj.filename)
            file_obj.save(dest)
            saved.append(file_obj.filename)

# AFTER:
    for key in request.files:
        file_obj = request.files[key]
        if file_obj.filename:
            safe_name = secure_filename(file_obj.filename)
            if not safe_name:
                continue
            dest = os.path.join(data_dir, safe_name)
            file_obj.save(dest)
            saved.append(safe_name)
```

---

## Step 3 — Add Thread Locks to Batch State Dicts (C1)

### 3a. `backend/captioner/service.py`

Wrap `_batch_state` mutations and reads with a `threading.Lock`:

```python
# Add to imports:
import threading

# Replace the module-level state block:

# BEFORE:
_batch_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_file": "",
    "results": {},
    "error": None,
}

def get_batch_state() -> dict:
    return dict(_batch_state)

# AFTER:
_batch_lock = threading.Lock()
_batch_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_file": "",
    "results": {},
    "error": None,
}

def get_batch_state() -> dict:
    with _batch_lock:
        return dict(_batch_state)
```

Then in `caption_batch()`, wrap all state mutations:

```python
# BEFORE:
def caption_batch(stems: list, data_dir: str, cache_dir: str = None):
    ...
    global _batch_state

    _batch_state["running"] = True
    _batch_state["total"] = len(stems)
    _batch_state["completed"] = 0
    _batch_state["current_file"] = ""
    _batch_state["results"] = {}
    _batch_state["error"] = None

    try:
        for stem in stems:
            ...
            if not os.path.exists(mp3_path):
                _batch_state["completed"] += 1
                continue

            _batch_state["current_file"] = stem
            ...
            try:
                ...
                _batch_state["results"][stem] = caption
            except Exception as e:
                _batch_state["results"][stem] = f"ERROR: {e}"

            _batch_state["completed"] += 1

    except Exception as e:
        _batch_state["error"] = str(e)
    finally:
        _batch_state["running"] = False
        _batch_state["current_file"] = ""

# AFTER:
def caption_batch(stems: list, data_dir: str, cache_dir: str = None):
    ...
    with _batch_lock:
        _batch_state["running"] = True
        _batch_state["total"] = len(stems)
        _batch_state["completed"] = 0
        _batch_state["current_file"] = ""
        _batch_state["results"] = {}
        _batch_state["error"] = None

    try:
        for stem in stems:
            ...
            if not os.path.exists(mp3_path):
                with _batch_lock:
                    _batch_state["completed"] += 1
                continue

            with _batch_lock:
                _batch_state["current_file"] = stem
            ...
            try:
                ...
                with _batch_lock:
                    _batch_state["results"][stem] = caption
            except Exception as e:
                with _batch_lock:
                    _batch_state["results"][stem] = f"ERROR: {e}"

            with _batch_lock:
                _batch_state["completed"] += 1

    except Exception as e:
        with _batch_lock:
            _batch_state["error"] = str(e)
    finally:
        with _batch_lock:
            _batch_state["running"] = False
            _batch_state["current_file"] = ""
```

### 3b. `backend/dataset_api.py`

Same pattern for `_convert_state`:

```python
# Add after imports:
import threading as _threading
_convert_lock = _threading.Lock()

# Replace the state block:

# BEFORE:
_convert_state = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
}

# AFTER:
_convert_lock = _threading.Lock()
_convert_state = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
}
```

Then in `convert()` route handler, wrap the running check and state reset:

```python
# BEFORE:
    if _convert_state["running"]:
        return jsonify({"success": False, "error": "Conversion already in progress"}), 409
    ...
    _convert_state["running"] = True
    _convert_state["progress"] = "Starting conversion..."
    _convert_state["result"] = None
    _convert_state["error"] = None

# AFTER:
    with _convert_lock:
        if _convert_state["running"]:
            return jsonify({"success": False, "error": "Conversion already in progress"}), 409
        _convert_state["running"] = True
        _convert_state["progress"] = "Starting conversion..."
        _convert_state["result"] = None
        _convert_state["error"] = None
```

And in `run_conversion()` inner function:

```python
# BEFORE:
    def run_conversion():
        global _convert_state
        try:
            _convert_state["progress"] = "Reading data files..."
            result = ds.convert_to_hf_dataset_standalone(
                work_dir=work_dir,
                output_name=output_name,
                progress_callback=lambda msg: _convert_state.update({"progress": msg}),
                trigger_word=trigger_word,
            )
            _convert_state["result"] = result
            _convert_state["progress"] = "Done"
        except Exception as e:
            _convert_state["error"] = str(e)
            _convert_state["progress"] = f"Error: {e}"
        finally:
            _convert_state["running"] = False

# AFTER:
    def run_conversion():
        def _update_progress(msg):
            with _convert_lock:
                _convert_state["progress"] = msg

        try:
            _update_progress("Reading data files...")
            result = ds.convert_to_hf_dataset_standalone(
                work_dir=work_dir,
                output_name=output_name,
                progress_callback=_update_progress,
                trigger_word=trigger_word,
            )
            with _convert_lock:
                _convert_state["result"] = result
                _convert_state["progress"] = "Done"
        except Exception as e:
            with _convert_lock:
                _convert_state["error"] = str(e)
                _convert_state["progress"] = f"Error: {e}"
        finally:
            with _convert_lock:
                _convert_state["running"] = False
```

And in `convert_status()`:

```python
# BEFORE:
def convert_status():
    return jsonify({
        "running": _convert_state["running"],
        "progress": _convert_state["progress"],
        "result": _convert_state["result"],
        "error": _convert_state["error"],
    })

# AFTER:
def convert_status():
    with _convert_lock:
        return jsonify({
            "running": _convert_state["running"],
            "progress": _convert_state["progress"],
            "result": _convert_state["result"],
            "error": _convert_state["error"],
        })
```

---

## Step 4 — Add `atexit` Cleanup for TensorBoard (S5)

**File**: `backend/trainer_api.py`

```python
# Add to imports:
import atexit

# Add after `_tensorboard_process = None`:
def _cleanup_tensorboard():
    global _tensorboard_process
    if _tensorboard_process and _tensorboard_process.poll() is None:
        _tensorboard_process.terminate()

atexit.register(_cleanup_tensorboard)
```

---

## Step 5 — Replace `time.sleep` with `eventlet.sleep` in `_poll_gpu` (S3)

**File**: `backend/trainer_service.py`

```python
# BEFORE:
    def _poll_gpu(self):
        """Poll nvidia-smi every 5 seconds while training is running."""
        while self.is_running:
            try:
                stats = get_gpu_stats()
                if stats:
                    self._emit("gpu_stats", stats)
            except Exception:
                pass
            time.sleep(5)

# AFTER:
    def _poll_gpu(self):
        """Poll nvidia-smi every 5 seconds while training is running."""
        while self.is_running:
            try:
                stats = get_gpu_stats()
                if stats:
                    self._emit("gpu_stats", stats)
            except Exception:
                pass
            eventlet.sleep(5)
```

---

## Step 6 — Add Cache Version Check to `ImageBindEmbedder` (S1)

**File**: `backend/captioner/embedder.py`

Add a version fingerprint based on total candidate count so the in-memory cache is invalidated when candidates change during development:

```python
# BEFORE:
    def get_text_cache(self) -> dict:
        if not self._text_cache:
            self.precompute_text_cache()
        return self._text_cache

# AFTER:
    @staticmethod
    def _candidate_fingerprint() -> int:
        """Hash of total candidate count across all dimensions for staleness detection."""
        from backend.captioner.candidates import (
            GENRE_CANDIDATES, VOCAL_CANDIDATES, INSTRUMENT_CANDIDATES,
            MOOD_CANDIDATES, TEMPO_CANDIDATES, KEY_CANDIDATES,
            TIMBRE_CANDIDATES, ERA_CANDIDATES, PRODUCTION_CANDIDATES,
            ENERGY_CANDIDATES,
        )
        return sum(len(c) for c in [
            GENRE_CANDIDATES, VOCAL_CANDIDATES, INSTRUMENT_CANDIDATES,
            MOOD_CANDIDATES, TEMPO_CANDIDATES, KEY_CANDIDATES,
            TIMBRE_CANDIDATES, ERA_CANDIDATES, PRODUCTION_CANDIDATES,
            ENERGY_CANDIDATES,
        ])

    def get_text_cache(self) -> dict:
        expected = self._candidate_fingerprint()
        if not self._text_cache or getattr(self, '_cache_fingerprint', None) != expected:
            self.precompute_text_cache()
            self._cache_fingerprint = expected
        return self._text_cache
```

---

## Step 7 — Remove Dead `num_workers` Field from Trainer UI (S2)

**File**: `static/index.html`

Find and remove the Workers form row. The backend hardcodes `--num_workers 0` for Windows compatibility, so the UI field is misleading.

Search for the HTML containing `cfg-workers` and remove that entire form row.

Also remove the corresponding line from `static/js/trainer-ui.js` `gatherConfig()`:

```javascript
// REMOVE this line:
            num_workers: parseInt(document.getElementById('cfg-workers').value) || 4,
```

---

## Step 8 — Replace DOM-based `escHtml` with String Replacement (M5)

**File**: `static/js/dataset-editor.js`

```javascript
// BEFORE:
    escHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },

// AFTER:
    escHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    },
```

---

## Step 9 — Direct Captioner Temp Files into Workspace (M7)

**File**: `backend/captioner_api.py`

```python
# BEFORE:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_file.save(tmp.name)
    tmp.close()

# AFTER:
    tmp_dir = os.path.join(current_app.config["WORK_DIR"], "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=tmp_dir)
    audio_file.save(tmp.name)
    tmp.close()
```

---

## Step 10 — Delete Phantom Files (M1)

Remove the two broken pip output files from the project root:

```bash
del "K:\acestep15turbo\ACETrainer\=0.7.0"
del "K:\acestep15turbo\ACETrainer\=5.17.0"
```

---

## Verification Checklist

After all changes:

- [ ] App starts without import errors: `python app.py`
- [ ] Dataset editor loads samples, navigates between them
- [ ] Upload MP3 files — filenames are sanitized (no path separators)
- [ ] Audio playback works in the editor (serve_audio via send_from_directory)
- [ ] Auto-caption single file — returns 10-dimension caption
- [ ] Batch caption-all — progress polls correctly, results applied
- [ ] Dataset conversion — progress polls correctly, HF dataset created
- [ ] Start/stop training — logs stream, metrics update
- [ ] TensorBoard launch — process starts, cleanup on app exit
- [ ] GPU stats poll continuously during training without blocking UI
