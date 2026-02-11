# Stage 5 — Final Fixes Status

## Fixes Completed (applied and working)

### 1. Eventlet Errno 22 on Dataset Conversion
**File:** `ACETrainer/backend/dataset_api.py`
**Problem:** `eventlet.monkey_patch()` in `app.py` replaces `threading.Thread` with a green-thread wrapper. When the `/convert` endpoint spawned a "background thread" for HuggingFace Dataset conversion, it was still running inside eventlet's cooperative I/O scheduler. HF Datasets writes memory-mapped Arrow files, which breaks under eventlet's patched file descriptors on Windows — producing `[Errno 22] Invalid argument`.
**Fix:** Imported the real, unpatched `threading.Thread` via `eventlet.patcher.original("threading").Thread` and used that (`_RealThread`) to run the conversion. The HF Datasets save now runs on a true OS thread, bypassing eventlet entirely.
**Status:** Complete. Both "Convert to HF Dataset" and "Load ZIP" work.

### 2. ModelCheckpoint crash on training start
**File:** `ACETrainer/backend/trainer_service.py`
**Problem:** The UI passed `--save_top_k 2` to `trainer.py`. ACE-Step's `trainer.py` creates `ModelCheckpoint(monitor=None, save_top_k=2)`. PyTorch Lightning 2.5.1 rejects `save_top_k` values other than `-1`, `0`, or `1` when `monitor=None` — it has no metric to rank checkpoints by, so it can't pick a "top 2".
**Fix:** Hardcoded `--save_top_k` to `-1` (keep all checkpoints) in the command built by `trainer_service.py`. LoRA adapters are small (~MBs), so keeping all is fine. The fix lives entirely in ACETrainer — ACE-Step's `trainer.py` is untouched.
**Status:** Complete. Training launches successfully.

---

## Fixes NOT Yet Applied (pending next server restart)

*None — both fixes are applied and the currently running training session launched successfully with them.*

---

## Stale Artifacts to Clean Up (non-blocking)

### 3. Leftover dataset directories
**Location:** `ACETrainer/workspace/datasets/`
**Issue:** Previous conversion attempts left behind extra datasets:
- `lora_dataset` — default-named dataset from testing
- `test-banana` — appears to be a test dataset
These don't affect anything but waste disk space. The active dataset is `sleepy-platano`.
**Action needed:** Delete `lora_dataset` and `test-banana` after training completes (not while training is live).

---

## Design Debt (no fix needed now, but worth noting)

### 4. Original plan references `bf16-mixed` — app correctly uses `bf16-true`
**Location:** `stepaudio-tuning-plan.md` (Sections 5.4, 2.3 presets table)
**Issue:** The original plan recommends `bf16-mixed` precision throughout. The actual implementation in `trainer-ui.js` presets correctly defaults to `bf16-true`, and the README/USER-GUIDE correctly warn that `bf16-mixed` breaks LoRA gradients. The plan document is outdated on this point.
**Impact:** None — the running code is correct. The plan doc is just stale.

### 5. `save_top_k` UI field is now ignored
**Location:** `ACETrainer/backend/trainer_service.py` line 112
**Issue:** The hardcoded `"--save_top_k", "-1"` means the UI's `save_top_k` config value (if one were ever exposed) would be silently ignored. Currently no UI control exists for this, so there's no user-facing impact. If a UI control is added later, the hardcoded value would need to be revisited — either by adding a `monitor` metric so PL can rank checkpoints, or by only allowing valid values (`-1`, `0`, `1`) in the UI.
**Impact:** None currently.

### 6. Original plan's `repeat_count` / duplication logic was replaced
**Location:** `stepaudio-tuning-plan.md` Section 5.6
**Issue:** The plan describes a `repeat_count` parameter that duplicates dataset rows at conversion time. The actual implementation in `dataset_service.py` does NOT duplicate — it stores one row per sample and relies on PyTorch Lightning's `epochs=-1, max_steps=N` loop to handle repetition naturally. This is the correct approach. The plan is just stale.
**Impact:** None — the running code is correct.

---

## ACE-Step Dependency Status

**`K:\acestep15turbo\ACE-Step\` — CLEAN / UNMODIFIED**

All fixes were applied exclusively within `ACETrainer/`. The ACE-Step folder is the upstream dependency and has not been altered.
