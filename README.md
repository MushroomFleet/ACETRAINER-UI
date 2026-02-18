# ACE-Step Turbo Trainer

A web-based UI for fine-tuning the [ACE-Step](https://github.com/ace-step/ACE-Step) music generation model using LoRA. Built with Flask, Socket.IO, and vanilla JavaScript.

**Current version: v1.5.0-beta** -- See [changelog-v1.5.0-beta.md](changelog-v1.5.0-beta.md) for details on the latest changes.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 16+ GB VRAM (tested on RTX 4090 24GB)
- Windows (Linux should work but is untested)
- The official ACE-Step repository cloned alongside this one
- ffmpeg on PATH (required for audio captioning -- install via `conda install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org))
- Meta's ImageBind repository cloned alongside this one (optional -- required only for auto-captioning)

## Required Directory Structure

**This app is not standalone.** It must be placed next to the official ACE-Step repository as a sibling directory:

```
your-project-folder/
├── ACE-Step/              <-- Official ACE-Step repo (clone this first)
│   ├── acestep/           <-- Model code (pipeline, transformer, scheduler, etc.)
│   ├── trainer.py         <-- Training script (launched by this app)
│   ├── config/
│   │   └── zh_rap_lora_config.json
│   └── ...
│
├── ImageBind/             <-- Meta ImageBind repo (optional, for auto-captioning)
│   ├── imagebind/
│   └── setup.py
│
└── ACETrainer/            <-- This repo
    ├── app.py             <-- Flask + Socket.IO entry point (port 7870)
    ├── backend/
    │   ├── captioner/     <-- ImageBind audio classification captioner
    │   │   ├── candidates.py      <-- Text candidate libraries (10 dimensions)
    │   │   ├── classifier.py      <-- Zero-shot cosine similarity classifier
    │   │   ├── embedder.py        <-- ImageBind model wrapper + cache
    │   │   ├── audio_preprocessor.py  <-- MP3 to 16kHz mono WAV chunker
    │   │   └── service.py         <-- Captioning orchestrator
    │   ├── dataset_api.py         <-- Dataset CRUD, upload, validation, conversion
    │   ├── dataset_service.py     <-- File storage, HF dataset conversion
    │   ├── captioner_api.py       <-- Captioner REST endpoints
    │   ├── trainer_api.py         <-- Training control, TensorBoard management
    │   ├── trainer_service.py     <-- Training subprocess, log parser, GPU poller
    │   ├── training_hooks.py      <-- PyTorch forward hooks for embedding capture
    │   ├── visualization_api.py   <-- Visualization REST + Socket.IO endpoints
    │   └── visualization_service.py  <-- PaCMAP dimensionality reduction
    ├── static/
    │   ├── index.html             <-- Single-page app
    │   ├── css/style.css          <-- Dark theme styling
    │   └── js/
    │       ├── app.js             <-- Tab navigation, Socket.IO init
    │       ├── db.js              <-- IndexedDB wrapper for client-side storage
    │       ├── dataset-editor.js  <-- Dataset editor with caption display
    │       ├── trainer-ui.js      <-- Training config, monitoring, loss chart
    │       ├── visualization_panel.js  <-- PaCMAP viz tab controller
    │       └── utils.js           <-- Toast notifications, API helpers
    └── workspace/                 <-- Created automatically on first run
        ├── data/                  <-- Raw MP3 + txt files uploaded from editor
        ├── datasets/              <-- Converted HuggingFace Arrow datasets
        ├── configs/               <-- Generated lora_config.json files
        ├── captioner_cache/       <-- Cached ImageBind text embeddings
        ├── tmp/                   <-- Captioner temp files
        └── exps/logs/             <-- Training outputs, checkpoints, TensorBoard
```

The trainer UI launches `ACE-Step/trainer.py` as a subprocess using `ACE-Step/` as the working directory. All model code, the training loop, dataset loading, and inference pipeline live in the ACE-Step package. This app provides the UI layer for dataset management, training configuration, and real-time monitoring.

## Setup

### 1. Clone ACE-Step

```bash
git clone https://github.com/ace-step/ACE-Step.git
```

Follow the ACE-Step README to install its dependencies (PyTorch, diffusers, transformers, etc.).

### 2. Clone This Repo

```bash
git clone <this-repo-url> ACETrainer
```

### 3. Install UI Dependencies

```bash
cd ACETrainer
pip install -r requirements-ui.txt
```

These are lightweight additions on top of the ACE-Step environment:
- Flask + Flask-SocketIO + Flask-CORS (web server)
- eventlet (async Socket.IO transport with cooperative green threading)
- mutagen (MP3 duration detection)
- pydub (MP3-to-WAV conversion for audio captioning)

### 3b. Install ImageBind (Optional -- for auto-captioning)

The auto-captioning feature uses Meta's ImageBind model for zero-shot audio classification. Skip this step if you plan to write all prompt tags manually.

```bash
# Clone ImageBind alongside ACE-Step and ACETrainer
git clone https://github.com/facebookresearch/ImageBind

# Install without pulling its own torch version
cd ImageBind
pip install --no-deps .

# Install ImageBind sub-dependencies
pip install "timm>=0.9.0" ftfy regex einops iopath types-regex
pip install git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

> **Note:** The first time you run a captioning request, ImageBind will download its pretrained weights (~5 GB). Subsequent runs load from the local cache.

### 4. Run

```bash
python app.py
```

Open your browser to **http://127.0.0.1:7870**

## Features

### Dataset Editor
- Upload, preview, and edit training samples (MP3 + prompt + lyrics) in the browser
- Client-side data storage in IndexedDB -- nothing is sent to the server until you explicitly upload
- Audio playback, sample navigation with keyboard shortcuts (arrow keys)
- Bulk tag operations: prepend, append, and find/replace across all prompts
- ZIP import/export for portable dataset management
- Dataset validation with detailed error reporting
- Filename sanitization on upload for security

### Auto-Captioning (10 Dimensions)
Zero-shot audio classification via ImageBind generates structured prompt tags aligned to the full ACE-Step prompt template:

| Dimension | Candidates | top_k | Example output |
|-----------|-----------|-------|----------------|
| Genre | 56 | 1 | pop, synthwave, drum and bass |
| Vocal | 18 | 1 | female vocal, rapping vocal, breathy vocal |
| Instruments | 41 | 3 | piano and synthesizer and strings orchestra |
| Mood | 35 | 2 | dreamy and nostalgic |
| Tempo | 13 | 1 | 120 BPM |
| Key | 22 | 1 | C minor |
| Timbre | 24 | 1 | warm, bright, compressed |
| Era | 12 | 1 | 1980s synth-pop new wave, modern contemporary |
| Production | 14 | 1 | studio-polished, lo-fi, analog |
| Energy | 10 | 1 | high, explosive, building |

Captioning works both per-sample ("Auto-Caption" button) and in batch ("Caption All" for the entire dataset). Text embeddings are cached to disk for instant reuse across restarts, with automatic cache invalidation when dimensions or candidate lists change.

### Trigger Word
Set a LoRA activation keyword in the Dataset Editor (e.g. `MYSTYLE`) that is automatically prepended to all captions during dataset conversion, so the model learns to associate the keyword with your training style.

### Dataset Conversion
One-click conversion from raw files (MP3 + prompt + lyrics) to HuggingFace Arrow Dataset format. Runs in a background thread with progress polling. The conversion uses real OS threads (not eventlet green threads) to avoid I/O conflicts with HF Datasets' memory-mapped Arrow files on Windows.

### Training Configuration
- **Presets** -- Conservative (~14GB, r=16), Balanced (~18GB, r=64), and Aggressive (~22GB, r=256) tuned for RTX 4090
- **LoRA parameters** -- rank, alpha, dropout, RS-LoRA toggle, target module selection (speaker_embedder, attention projections)
- **Training hyperparameters** -- learning rate, LR schedule (cosine restarts / linear decay), max steps, precision (bf16-true recommended), gradient accumulation, gradient clipping, shift
- **Checkpoint intervals** -- configurable save frequency and plot frequency
- `num_workers` is hardcoded to 0 for Windows compatibility (py3langid cannot pickle across worker processes)

### Real-Time Monitoring
- Live loss chart (Chart.js) with automatic downsampling for long runs
- GPU memory, utilization, and temperature via `nvidia-smi` polling (cooperative eventlet scheduling)
- Log streaming via Socket.IO with progress bar filtering and throttled emissions
- Step counter with global step computation across epochs
- Fallback REST polling (3s interval) in case Socket.IO events are missed
- Automatic reconnection to running training sessions on page refresh

### Checkpoint Management
- LoRA adapters saved at configurable step intervals
- Checkpoint detection from subprocess log output with real-time UI notifications
- TensorBoard integration with one-click launch and automatic cleanup on app exit

### PaCMAP Embedding Visualization
Real-time 2D projections of training embeddings via PaCMAP dimensionality reduction:
- **Latent space** -- DCAE encoder audio representations (watch for genre/style clustering)
- **LoRA outputs** -- transformer attention projection outputs colored by layer (monitor for specialization)
- **Prompt space** -- text encoder embeddings (verify genre groupings and trigger word separation)
- Automatic snapshot saving during training with animation assembly on completion
- Interactive Plotly charts with hover data

## How Training Works

The app does **not** contain any ML code. It:

1. Writes your LoRA config to `workspace/configs/lora_config.json`
2. Launches `python trainer.py` inside the `ACE-Step/` directory as a subprocess
3. Streams stdout through `eventlet.tpool` (blocking reads offloaded to OS threads)
4. Parses PyTorch Lightning log output for epoch, step, loss, and checkpoint events
5. Emits real-time updates to the browser via Socket.IO (`/training` namespace)
6. Polls GPU stats via `nvidia-smi` every 5 seconds using cooperative scheduling
7. On completion, assembles PaCMAP visualization animation from saved snapshots

Training outputs (checkpoints, LoRA adapters, TensorBoard logs) are saved to `workspace/exps/logs/`.

## Using Your Trained LoRA

After training, LoRA adapters are saved at each checkpoint interval to:

```
workspace/exps/logs/lightning_logs/<run-name>/checkpoints/epoch=N-step=N_lora/
  pytorch_lora_weights.safetensors
```

Load this adapter into the ACE-Step inference pipeline to generate music with your fine-tuned style. Include your trigger word in the prompt to activate the LoRA effect.

## Architecture Notes

### Threading Model
The app runs on eventlet with monkey-patching for cooperative green threading. Two exceptions require real OS threads:
- **HF Dataset conversion** -- Arrow memory-mapped I/O fails under eventlet's patched file descriptors on Windows (Errno 22). Uses `eventlet.patcher.original("threading").Thread`.
- **Subprocess stdout reading** -- Blocking `readline()` calls are offloaded via `eventlet.tpool.execute()` so the event loop stays responsive.

### Security
- File uploads are sanitized via Werkzeug's `secure_filename()`
- Audio serving uses `send_from_directory()` for path traversal protection
- Background job state (batch captioning, dataset conversion) is protected by `threading.Lock()` to prevent torn reads during concurrent polling
- TensorBoard subprocess is cleaned up via `atexit` handler

### Caching
- **Text embeddings** (ImageBind candidate vectors) are persisted to disk as a `.pt` file. Auto-invalidated when the dimension key set changes (e.g., adding new classification dimensions) or when the total candidate count changes (e.g., adding new candidates within a dimension).
- **Temp files** from captioner uploads are stored in `workspace/tmp/` rather than the OS temp directory for easier cleanup and visibility.

## Documentation

See [USER-GUIDE.md](USER-GUIDE.md) for detailed training guidance including:
- Auto-captioning with ImageBind (how it works, how to use it)
- Trigger word setup for LoRA activation
- Step/epoch calculations for small datasets
- Time estimates
- How to read the loss graph
- Parameter explanations
- Troubleshooting

## Known Issues

- `num_workers` is forced to 0 on Windows due to multiprocessing limitations with py3langid
- `bf16-mixed` precision breaks LoRA gradients -- always use `bf16-true`
- DDP is unavailable on Windows (no NCCL) -- single GPU only with `strategy="auto"`
- CORS is open (`*`) by design for localhost development -- not intended for public deployment
- PaCMAP visualizer creates a new instance per request (no persistence across restarts)

## Changelog

- [v1.5.0-beta](changelog-v1.5.0-beta.md) -- 10-dimension captioner, security hardening, stability fixes

## 📚 Citation

### Academic Citation

If you use this codebase in your research or project, please cite:

```bibtex
@software{acetrainer_ui,
  title = {ACETRAINER UI: Music generation LoRA Trainer},
  author = {[Drift Johnson]},
  year = {2025},
  url = {https://github.com/MushroomFleet/ACETRAINER-UI},
  version = {1.0.0}
}
```

### Donate:


[![Ko-Fi](https://cdn.ko-fi.com/cdn/kofi3.png?v=3)](https://ko-fi.com/driftjohnson)
