# ACE-Step Turbo Trainer

A web-based UI for fine-tuning the [ACE-Step](https://github.com/ace-step/ACE-Step) music generation model using LoRA. Built with Flask, Socket.IO, and vanilla JavaScript.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 16+ GB VRAM (tested on RTX 4090 24GB)
- Windows (Linux should work but is untested)
- The official ACE-Step repository cloned alongside this one
- ffmpeg on PATH (required for audio captioning — install via `conda install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org))
- Meta's ImageBind repository cloned alongside this one (optional — required only for auto-captioning)

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
    ├── app.py
    ├── backend/
    │   ├── captioner/     <-- ImageBind audio classification captioner
    │   ├── dataset_api.py
    │   ├── dataset_service.py
    │   └── captioner_api.py
    ├── static/
    └── workspace/         <-- Created automatically on first run
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
- eventlet (async Socket.IO transport)
- mutagen (MP3 duration detection)
- pydub (MP3-to-WAV conversion for audio captioning)

### 3b. Install ImageBind (Optional — for auto-captioning)

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

- **Dataset Editor** — Upload, preview, and edit training samples (MP3 + prompt + lyrics) in the browser. Data stored client-side in IndexedDB until uploaded to the server.
- **Auto-Captioning** — Zero-shot audio classification via ImageBind generates structured captions (genre, vocal type, instruments, mood, tempo, key) per sample — individually or in batch.
- **Trigger Word** — Set a LoRA activation keyword in the Dataset Editor that is automatically prepended to all captions during dataset conversion.
- **One-Click Conversion** — Converts raw files to HuggingFace Dataset format for training.
- **Training Presets** — Conservative, Balanced, and Aggressive presets tuned for RTX 4090.
- **Full Config Control** — LoRA rank/alpha, learning rate, precision, gradient accumulation, checkpoint intervals, and more.
- **Real-Time Monitoring** — Live loss graph, GPU utilization, and log streaming via Socket.IO.
- **Checkpoint Management** — LoRA adapters saved at every checkpoint interval. Full `.ckpt` files limited to the 2 most recent to save disk space.

## How Training Works

The app does **not** contain any ML code. It:

1. Writes your LoRA config to `workspace/configs/lora_config.json`
2. Launches `python trainer.py` inside the `ACE-Step/` directory as a subprocess
3. Parses stdout for loss metrics and progress
4. Streams updates to the browser via Socket.IO
5. Monitors GPU usage via `nvidia-smi`

Training outputs (checkpoints, LoRA adapters, TensorBoard logs, preview audio) are saved to `workspace/exps/logs/`.

## Using Your Trained LoRA

After training, LoRA adapters are saved at each checkpoint interval to:

```
workspace/exps/logs/lightning_logs/<run-name>/checkpoints/epoch=N-step=N_lora/
  pytorch_lora_weights.safetensors
```

Load this adapter into the ACE-Step inference pipeline to generate music with your fine-tuned style.

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
- `bf16-mixed` precision breaks LoRA gradients — always use `bf16-true`
- DDP is unavailable on Windows (no NCCL) — single GPU only with `strategy="auto"`
