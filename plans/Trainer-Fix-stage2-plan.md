# Trainer Fix — Stage 2 Plan

**Date:** 2025-02-18
**Severity:** CRITICAL — Training is completely blocked
**Root Cause:** PaCMAP visualization hooks crash the training forward pass

---

## 1. Problem Statement

Training fails immediately at step 0 with:

```
File "K:\acestep15turbo\ACETrainer\backend\training_hooks.py", line 102, in hook
AttributeError: 'BaseModelOutputWithPastAndCrossAttentions' object has no attribute 'shape'
```

The crash occurs inside `_make_prompt_hook()` → the `hook()` closure registered on `self.text_encoder_model`.

## 2. Root Cause Analysis

### 2.1 The Hook Architecture

In `trainer.py` lines 162–215, the `Pipeline.__init__()` registers three PyTorch forward hooks via `EmbeddingHookManager`:

```python
hook_config = {
    "latent_layers": ["dcae"],          # MusicDCAE module
    "lora_layers":   ["transformers"],   # ACE-Step transformer (with LoRA)
    "prompt_layers": ["text_encoder_model"],  # UMT5EncoderModel
}
self._hook_manager.register_hooks(self, hook_config)
```

These hooks intercept the **entire module's** `forward()` output — not a sub-layer.

### 2.2 Why the Prompt Hook Crashes

The `text_encoder_model` is a HuggingFace `UMT5EncoderModel`. Its `forward()` returns a **`BaseModelOutputWithPastAndCrossAttentions`** dataclass, NOT a tensor.

The prompt hook at line 92–114 of `training_hooks.py` does:

```python
def hook(module, input, output):
    if isinstance(output, tuple):     # ← False — it's a dataclass
        embeds = output[0]
    else:
        embeds = output               # ← Gets the dataclass object
    if len(embeds.shape) == 3:        # ← CRASH: dataclass has no .shape
```

The `isinstance(output, tuple)` check fails because `BaseModelOutputWithPastAndCrossAttentions` is a `ModelOutput` (an `OrderedDict` subclass), not a `tuple`. The hook then tries to access `.shape` on the dataclass, which doesn't exist.

### 2.3 The Same Risk Exists for ALL Three Hooks

| Hook | Module | `forward()` output type | Status |
|------|--------|------------------------|--------|
| `_make_latent_hook` | `dcae` (MusicDCAE) | Complex — `encode()` returns `(latents, lengths)` but `forward()` returns something different. Hook fires on `forward()`. | **RISKY** — The hook registers on the whole `dcae` module. `MusicDCAE.forward()` actually calls `encode` internally. Since `dcae.encode()` is called directly in `trainer.py:413`, the hook on `dcae` fires for `.encode()` because PyTorch hooks fire on `__call__` → `forward`. This could also produce unexpected output shapes. |
| `_make_lora_hook` | `transformers` | Complex transformer output, likely a dataclass or tuple | **RISKY** — same pattern |
| `_make_prompt_hook` | `text_encoder_model` | `BaseModelOutputWithPastAndCrossAttentions` | **CRASHES** ← current failure |

### 2.4 Why This Wasn't Caught Before

The hooks were added as a visualization feature (PaCMAP embedding capture) and tested with the hook code present but may not have been exercised in a full training run with `plot_step` triggering at step 0.

The crash path is:
```
training_step → run_step → plot_step (step 0, every_plot_step condition met)
    → predict_step → preprocess → get_text_embeddings
        → self.text_encoder_model(**inputs)   # triggers forward hook
            → hook() → output.shape → CRASH
```

## 3. Fix Strategy

### Option A: Fix hooks to handle HuggingFace ModelOutput objects (SELECTED)
Pros: Preserves visualization capability, minimal code change
Cons: Hooks still add overhead to every forward pass

### Option B: Disable hooks entirely
Pros: Eliminates all risk
Cons: Loses PaCMAP visualization feature

### Option C: Make hooks opt-in via command-line flag
Pros: Safe by default, users can enable when needed
Cons: More changes required

**Decision:** Implement **Option A** (fix the hooks) with a **safety wrapper** that catches ANY exception inside hooks so visualization bugs can never crash training again. Additionally, add proper type-checking for HuggingFace `ModelOutput` objects.

## 4. Required Changes

### Fix 1: `backend/training_hooks.py` — Handle ModelOutput in ALL hooks

**Problem:** All three hook closures assume `output` is either a `tuple` or a `Tensor`. HuggingFace models return `ModelOutput` dataclasses.

**Fix:** Add `ModelOutput` detection to each hook. Extract the appropriate tensor from the dataclass. Wrap each hook body in a try/except so visualization errors NEVER crash training.

#### 4.1 Add ModelOutput Import

```python
# At top of training_hooks.py, after existing imports:
try:
    from transformers.modeling_outputs import ModelOutput
except ImportError:
    ModelOutput = None  # Graceful fallback if transformers not installed
```

#### 4.2 Add Helper: Extract Tensor from Any Output

```python
@staticmethod
def _extract_tensor(output):
    """
    Safely extract a usable tensor from a PyTorch forward hook output.
    Handles: raw Tensors, tuples, HuggingFace ModelOutput dataclasses.
    Returns a tensor or None if extraction fails.
    """
    import torch
    # Raw tensor
    if isinstance(output, torch.Tensor):
        return output
    # HuggingFace ModelOutput (OrderedDict subclass)
    if ModelOutput is not None and isinstance(output, ModelOutput):
        # Try .last_hidden_state first (text encoders)
        if hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
            return output.last_hidden_state
        # Try .hidden_states (some models)
        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
            return output.hidden_states[-1]
        # Fallback: first non-None value
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v
        return None
    # Tuple / list
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        return None
    return None
```

#### 4.3 Rewrite `_make_prompt_hook`

```python
def _make_prompt_hook(self, name: str):
    """Hook for text encoder output -- prompt embedding space."""
    def hook(module, input, output):
        if not self.enabled:
            return
        try:
            embeds = self._extract_tensor(output)
            if embeds is None:
                return  # Silently skip — don't crash training
            # Mean pool over sequence: (batch, seq_len, dim) -> (batch, dim)
            if len(embeds.shape) == 3:
                embeds = embeds.mean(dim=1)
            elif len(embeds.shape) > 3:
                embeds = embeds.reshape(embeds.shape[0], -1)
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
        except Exception as e:
            # NEVER let visualization crash training
            print(f"[VizHook] prompt hook error (non-fatal): {e}")
    return hook
```

#### 4.4 Rewrite `_make_latent_hook`

```python
def _make_latent_hook(self, name: str):
    """Hook for VAE/DCAE encoder outputs -- audio latent space."""
    def hook(module, input, output):
        if not self.enabled:
            return
        try:
            latents = self._extract_tensor(output)
            if latents is None:
                return
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
        except Exception as e:
            print(f"[VizHook] latent hook error (non-fatal): {e}")
    return hook
```

#### 4.5 Rewrite `_make_lora_hook`

```python
def _make_lora_hook(self, name: str):
    """Hook for LoRA layer outputs -- transformer attention projections."""
    def hook(module, input, output):
        if not self.enabled:
            return
        try:
            out = self._extract_tensor(output)
            if out is None:
                return
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
        except Exception as e:
            print(f"[VizHook] lora hook error (non-fatal): {e}")
    return hook
```

### Fix 2: `ACE-Step/trainer.py` — Add Safety Wrapper Around Hook Init

**Problem:** If hook registration itself fails, the `except` block at line 213 catches it, but the error message format could be clearer.

**Fix:** Already handled — the existing try/except at lines 165–215 catches failures and sets `self._hook_manager = None`. No change needed, but verify the import guard (`_VIZ_ENABLED`) correctly gates everything.

**Verified:** Lines 37–44 already provide a proper import guard:
```python
_VIZ_ENABLED = False
try:
    from training_hooks import EmbeddingHookManager as _EmbeddingHookManager
    _VIZ_ENABLED = True
except ImportError:
    pass
```

And line 165 gates on it:
```python
if _VIZ_ENABLED and self.is_train:
```

**No changes needed in `trainer.py`.**

## 5. File Change Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `backend/training_hooks.py` | **MODIFY** | Add `ModelOutput` import, add `_extract_tensor()` helper, rewrite all 3 hook closures with try/except + ModelOutput support |
| `ACE-Step/trainer.py` | **NONE** | Already has proper fallback guards |

## 6. Testing Plan

After implementing the fix:

1. **Start the app:** `python app.py`
2. **Navigate to Trainer tab**
3. **Verify dataset:** Should show 26 samples (or stale warning if not reconverted)
4. **Click Convert** if needed to rebuild HF dataset with 26 samples
5. **Start Training** with default settings
6. **Expected:** Training proceeds past step 0 with `[VizHook]` messages in the log confirming hooks are active OR silently skipped
7. **Verify:** No `AttributeError` crash. Loss values appear in the training log.

## 7. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| `_extract_tensor` returns wrong tensor from ModelOutput | Uses `.last_hidden_state` first (the standard field for encoder models), which matches what `get_text_embeddings` uses at line 384 |
| Hook overhead slows training | Hooks only capture/emit numpy arrays — minimal overhead. The try/except adds negligible cost. |
| New crash in different code path | Every hook body is wrapped in try/except — ANY future error is caught and logged, never propagated |
| DCAE or transformer output also fails | Same `_extract_tensor` + try/except pattern applied to all three hooks |

---

**Implementation Time Estimate:** ~10 minutes
**Files Modified:** 1 (`backend/training_hooks.py`)
**Regression Risk:** ZERO — all hooks are wrapped in try/except, training cannot be affected
