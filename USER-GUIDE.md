# ACE-Step Turbo Trainer - User Guide

## What This Tool Does

ACE-Step Turbo Trainer fine-tunes the ACE-Step music generation model on your own audio samples using LoRA (Low-Rank Adaptation). You provide a handful of songs, and after training, the model learns to generate music that captures the style, timbre, or vocal characteristics of your samples.

---

## Quick Start

1. **Prepare your data**: For each song, you need three files sharing the same stem name:
   - `song_name.mp3` — the audio file
   - `song_name_prompt.txt` — a text description/tags (e.g. `male vocal, rock, energetic, electric guitar`)
   - `song_name_lyrics.txt` — the lyrics of the song

2. **Upload** your files through the Dataset Editor tab, or load a ZIP containing all files.

3. **Convert** the dataset by clicking "Convert to HF Dataset".

4. **Configure** training parameters (or use a preset).

5. **Start training** and monitor progress via the loss graph and logs.

6. **Use your LoRA** — checkpoints are saved periodically and can be loaded into ACE-Step for inference.

---

## Understanding Training Duration

### How Steps and Epochs Work

Training progress is measured in **steps** (gradient updates), not time. Each step processes one sample from your dataset.

With a small dataset, the trainer loops through your samples many times (epochs):

| Dataset Size | Steps Per Epoch | Epochs for 5,000 Steps | Epochs for 10,000 Steps |
|---|---|---|---|
| 5 songs | 5 | 1,000 | 2,000 |
| 10 songs | 10 | 500 | 1,000 |
| 25 songs | 25 | 200 | 400 |
| 50 songs | 50 | 100 | 200 |

**This is normal.** Repeating over a small dataset many times is how LoRA fine-tuning works. The model gradually learns the patterns in your audio across thousands of passes.

### How Long Will Training Take?

Each step involves:
1. Loading and encoding your audio sample
2. Computing SSL features (MERT + mHuBERT models) — this is the slowest part
3. Running the forward pass through the transformer
4. Computing loss and updating LoRA weights

On an **RTX 4090** with default settings, expect roughly:
- **~7-13 seconds per step** (varies by audio length and model loading)
- **5,000 steps** ≈ 10-18 hours
- **10,000 steps** ≈ 20-36 hours

The "Plot Every N Steps" setting also adds time — each plot runs full 60-step inference to generate a preview audio. Set this higher (e.g. 5000) if you want faster training and don't need frequent previews.

### Recommended Step Counts

| Goal | Steps | Notes |
|---|---|---|
| Quick test | 500-1,000 | Enough to see if loss is decreasing. Output quality will be rough. |
| Light fine-tune | 2,000-3,000 | Noticeable style transfer. Good for a first real test. |
| Standard training | 5,000 | Recommended starting point for most use cases. |
| Deep fine-tune | 10,000 | Recommended for very small datasets (< 10 songs). Each sample seen many times. |
| Extended | 10,000+ | Diminishing returns. Watch for overfitting (loss stops decreasing or spikes). |

---

## Checkpoints and Testing Early

You do **not** need to wait for training to complete before testing results.

### Checkpoint Schedule

LoRA checkpoints are saved at regular intervals controlled by "Save Every N Steps" (default: **500 steps**). With default settings, checkpoints are saved at:

> Step 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000

Each checkpoint is a fully usable LoRA adapter. You can:
- **Stop training early** after any checkpoint and test the result
- **Compare** checkpoints at different steps to find the sweet spot
- **Resume** training later from a checkpoint if you want more steps

Checkpoints are saved to: `workspace/exps/logs/<experiment>/checkpoints/`

### When to Stop Early

- **Loss is plateauing** — if the loss graph flattens out, more steps won't help much
- **Loss is spiking** — sudden increases may indicate overfitting; use an earlier checkpoint
- **You're happy with the preview** — the plot step generates preview audio you can listen to

---

## Training Configuration

### Presets

Three presets are provided, tuned for RTX 4090 (24GB):

| Preset | LoRA Rank | VRAM Usage | Best For |
|---|---|---|---|
| **Conservative** | r=16 | ~14 GB | Quick experiments, limited VRAM |
| **Balanced** | r=64 | ~18 GB | General purpose (recommended default) |
| **Aggressive** | r=256 | ~22 GB | Maximum quality, full VRAM usage |

### Key Parameters Explained

**LoRA Rank (r)**: Controls the expressiveness of the fine-tune. Higher rank = more parameters = better style capture, but more VRAM and slightly slower training. For 5-10 songs, r=64 is a good balance.

**LoRA Alpha**: Scaling factor for LoRA updates. The effective learning rate scales as `alpha/rank`. Default of 32 works well across all rank values when RS-LoRA is enabled.

**RS-LoRA**: Rank-Stabilized LoRA. Keeps training stable across different rank values. Leave this on.

**Learning Rate**: How aggressively the model updates. Default `1e-4` is well-tested. Lower (e.g. `5e-5`) for more conservative/stable training. Higher (e.g. `2e-4`) to learn faster but risk instability.

**Precision**: Must be `bf16-true`. The `bf16-mixed` option is known to break LoRA gradient computation and will produce a non-functional model. FP32 uses twice the VRAM for no benefit.

**Gradient Accumulation**: Simulates a larger batch size without extra VRAM. With accumulation=4, every 4 steps are combined into one update, making training more stable but 4x slower per effective step.

**Shift**: Flow-matching schedule parameter. Default 3.0 matches ACE-Step's pretrained configuration. No reason to change this.

**Gradient Clip**: Prevents exploding gradients. Default 0.5 with norm clipping is standard.

### Target Modules

These are the transformer layers that LoRA modifies. All 8 are selected by default:
- `speaker_embedder` — voice/timbre characteristics
- `linear_q`, `linear_k`, `linear_v` — self-attention (music structure)
- `to_q`, `to_k`, `to_v`, `to_out.0` — cross-attention (text-to-music alignment)

For voice cloning, `speaker_embedder` is critical. For style transfer, the attention modules matter most. When in doubt, keep all enabled.

---

## Reading the Loss Graph

During training, the UI displays a real-time loss graph. Here's how to interpret it:

- **Steadily decreasing**: Training is working. The model is learning your data.
- **Noisy but trending down**: Normal. Individual steps vary, but the trend matters.
- **Flat from the start**: Something may be wrong (check precision is `bf16-true`, not `bf16-mixed`).
- **Sudden spike then recovery**: Often happens when the trainer hits a particularly difficult sample. Usually fine.
- **Steadily increasing after initial decrease**: Overfitting. Stop training and use an earlier checkpoint.

### Typical Loss Values

- **Start of training**: Loss is usually 0.03-0.10 depending on your data
- **After convergence**: Loss typically settles around 0.01-0.03
- **Exact values don't matter** — the trend is what matters. Lower is better, but overfitting produces low loss that doesn't generalize.

---

## Dataset Tips

### Quality Over Quantity

- 5-10 high-quality, well-recorded songs of a consistent style will outperform 50 mixed-quality tracks
- Clean audio without background noise trains better
- Consistent genre/style across samples helps the model learn a coherent style

### Audio Length

- Songs are processed at 48kHz stereo internally
- Longer songs take more VRAM per step and train slower
- 2-4 minute songs are ideal
- Very short clips (< 30s) may not provide enough context for the model

### Prompt Quality

Good prompts help the model associate text descriptions with audio features:
- Be specific: `female vocal, jazz, piano, slow tempo, intimate, smoky`
- Be consistent: Use similar tag formats across all samples
- Include: genre, instruments, tempo, mood, vocal characteristics

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Loss stays flat / NaN | Wrong precision setting | Use `bf16-true`, never `bf16-mixed` |
| Out of memory (OOM) | LoRA rank too high | Reduce rank, or use gradient accumulation |
| Training very slow | Plot step too frequent | Increase "Plot Every N Steps" to 5000+ |
| Checkpoint not saving | Step count too low | Wait until you reach the "Save Every" threshold |
| Server won't start | Port 7870 in use | Kill existing process on that port |

---

## File Locations

| What | Where |
|---|---|
| Raw audio + text files | `ACETrainer/workspace/data/` |
| Converted HF datasets | `ACETrainer/workspace/datasets/` |
| Training logs + checkpoints | `ACETrainer/workspace/exps/logs/` |
| LoRA checkpoints | `.../logs/<experiment>/checkpoints/epoch=N-step=N_lora/` |
| Training config (written at launch) | `ACETrainer/workspace/configs/lora_config.json` |
