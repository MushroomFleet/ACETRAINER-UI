# Changelog - v1.5.0-beta

## Audio Captioning System Overhaul

### Expanded from 6 to 10 classification dimensions

The ImageBind zero-shot audio captioner now classifies across 10 dimensions, up from the original 6. This brings coverage in line with the full ACE-Step prompt template vocabulary defined in `acestep-classifiers.txt`.

**New dimensions:**

| Dimension | Candidates | Example output |
|-----------|-----------|----------------|
| Timbre/Texture | 24 | warm, bright, crisp, lo-fi, compressed |
| Era Reference | 12 | 1980s synth-pop new wave, vintage retro, modern contemporary |
| Production Style | 14 | studio-polished, bedroom pop, analog, live recording |
| Energy Level | 10 | high, explosive, building, steady, fluctuating |

**Expanded existing dimension:**

- **Vocal** (+6 candidates): breathy, raspy, powerful belting, harmonies, call and response, ad-lib

**Caption output before:**
```
pop, female vocal, piano and synthesizer and strings, dreamy and nostalgic, 90 BPM, C minor
```

**Caption output after:**
```
pop, female vocal, piano and synthesizer and strings, dreamy and nostalgic, 90 BPM, C minor, warm, 2010s EDM trap, studio-polished, moderate
```

### Files changed

- `backend/captioner/candidates.py` -- added 4 new candidate lists (60 total new candidates), expanded vocal list
- `backend/captioner/classifier.py` -- extended `CaptionFields` dataclass to 10+1 fields, updated `to_caption_string()`, `to_detail_dict()`, and `classify_song()`
- `backend/captioner/embedder.py` -- updated `precompute_text_cache()` to import and embed all 10 candidate lists, added automatic disk cache invalidation via dimension key set comparison
- `static/js/dataset-editor.js` -- extended `formatCaptionDetails()` with conditional rendering of 4 new dimensions (backward-compatible with old API responses)

### Cache handling

The text embedding disk cache (`workspace/captioner_cache/text_embeddings_cache.pt`) auto-invalidates when it detects the stored dimension key set doesn't match the expected 10 dimensions. No manual deletion needed on upgrade.

---

## Security & Stability Fixes (Code Review Action Plan)

A comprehensive code review of the full codebase (22 files, ~5800 lines) identified 3 critical, 5 significant, and 7 minor issues. The 10 highest-value fixes were implemented:

### Critical fixes

**C2 - Path traversal in audio serving** -- `serve_audio()` used `send_file()` with unsanitized user input, allowing directory traversal via crafted stem parameters. Replaced with Flask's `send_from_directory()` which has built-in path containment.
- File: `backend/dataset_api.py`

**C3 - Unsanitized upload filenames** -- `upload_files()` saved files using the raw client-provided filename, allowing path traversal or special characters. Now applies Werkzeug's `secure_filename()` to all uploads.
- File: `backend/dataset_api.py`

**C1 - Thread-unsafe shared state** -- `_batch_state` (captioner) and `_convert_state` (dataset conversion) were read and written from multiple threads without synchronization, risking torn reads during status polling. Both now use `threading.Lock()` with fine-grained locking around all mutations and reads.
- Files: `backend/captioner/service.py`, `backend/dataset_api.py`

### Significant fixes

**S5 - TensorBoard process leak** -- The TensorBoard subprocess was only cleaned up if explicitly stopped via the UI. If the Flask app exited (Ctrl+C, crash, etc.), the process was orphaned. Added `atexit.register()` cleanup handler.
- File: `backend/trainer_api.py`

**S3 - Blocking sleep in GPU poller** -- `_poll_gpu()` used `time.sleep(5)` which blocks the entire eventlet hub, freezing all WebSocket emissions for 5 seconds per cycle. Replaced with cooperative `eventlet.sleep(5)`.
- File: `backend/trainer_service.py`

**S1 - Stale in-memory text embedding cache** -- The singleton `ImageBindEmbedder` never re-checked its in-memory cache after the first load. If candidate lists were modified during development, the running server would continue using stale embeddings. Added `_candidate_fingerprint()` based on total candidate count, checked on every `get_text_cache()` call.
- File: `backend/captioner/embedder.py`

**S2 - Dead num_workers UI field** -- The trainer UI exposed a "Num Workers" config field, but the backend hardcodes `--num_workers 0` for Windows compatibility. The field was misleading and its value was silently discarded. Removed the HTML form row and the corresponding `gatherConfig()` line.
- Files: `static/index.html`, `static/js/trainer-ui.js`

### Minor fixes

**M5 - DOM-based HTML escaping** -- `escHtml()` created a DOM element on every call just to escape HTML entities. Replaced with a pure string replacement chain (`&`, `<`, `>`, `"`). Eliminates unnecessary DOM allocations in the sample list render loop.
- File: `static/js/dataset-editor.js`

**M7 - System temp directory pollution** -- Captioner temp files (uploaded MP3 blobs) were written to the OS temp directory and could accumulate if cleanup failed. Now directed into `workspace/tmp/` for visibility and easier cleanup.
- File: `backend/captioner_api.py`

**M1 - Phantom pip output files** -- Two zero-byte files (`=0.7.0`, `=5.17.0`) in the project root, created by a malformed `pip install` command (likely `pip install eventlet=0.7.0` instead of `==`). Deleted.
- Files: `=0.7.0`, `=5.17.0` (deleted)

### Deliberately excluded

| ID | Issue | Reason |
|----|-------|--------|
| S4 | Open CORS policy | Localhost-only app, restricting origins could break dev workflows |
| M2 | phase2/ directory | Requires user decision on archiving vs deletion |
| M3 | CDN SRI hashes | Low payoff for a localhost tool |
| M4 | Over-reading validation | Requires API restructuring beyond a quick fix |
| M6 | PaCMAP re-creation | Documented limitation, not actionable without architectural change |

---

## Trainer Pipeline Verification

The full trainer pipeline was traced end-to-end after all changes:

```
gatherConfig() -> POST /api/trainer/start -> TrainerService.start_training()
  -> lora_config.json -> subprocess.Popen(trainer.py) -> eventlet.spawn_n(_stream_output)
  -> tpool.execute(readline) -> _parse_metrics() -> socketio.emit("training_log")
  -> eventlet.spawn_n(_poll_gpu) -> nvidia-smi -> socketio.emit("gpu_stats")
  -> on completion: training_complete event + viz animation assembly
```

**Result: No regressions.** All 15 config fields flow correctly from UI to backend. The `num_workers` removal is safe (backend was already ignoring the frontend value). The `eventlet.sleep` fix in `_poll_gpu` and the `atexit` TensorBoard cleanup improve reliability without altering data flow.

---

## Summary of all files modified

| File | Changes |
|------|---------|
| `backend/captioner/candidates.py` | +4 candidate lists, expanded vocals |
| `backend/captioner/classifier.py` | 10-dimension CaptionFields, updated output methods |
| `backend/captioner/embedder.py` | 10-dimension cache, auto-invalidation, fingerprint check |
| `backend/captioner/service.py` | Thread lock on `_batch_state` |
| `backend/captioner_api.py` | Temp files directed to workspace |
| `backend/dataset_api.py` | `send_from_directory`, `secure_filename`, thread lock on `_convert_state` |
| `backend/trainer_api.py` | `atexit` TensorBoard cleanup |
| `backend/trainer_service.py` | `eventlet.sleep` in GPU poller |
| `static/index.html` | Removed dead num_workers field |
| `static/js/dataset-editor.js` | New caption dimensions display, string-based escHtml |
| `static/js/trainer-ui.js` | Removed num_workers from gatherConfig |
| `=0.7.0` | Deleted |
| `=5.17.0` | Deleted |
