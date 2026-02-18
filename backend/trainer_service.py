"""
Trainer service — manages training subprocess, streams logs, monitors GPU.
"""

import os
import re
import sys
import json
import time
import subprocess
import threading
import eventlet


class TrainerService:
    def __init__(self):
        self.process = None
        self.is_running = False
        self.current_step = 0
        self.max_steps = 0
        self.current_epoch = 0
        self.metrics_history = []
        self.log_lines = []
        self.return_code = None
        self._socketio = None
        self._gpu_thread = None
        self._log_thread = None
        self.config = {}
        self.checkpoints = []

    def set_socketio(self, sio):
        self._socketio = sio

    def _emit(self, event, data):
        if self._socketio:
            self._socketio.emit(event, data, namespace="/training")

    def get_status(self):
        return {
            "is_running": self.is_running,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "current_epoch": self.current_epoch,
            "metrics": self.metrics_history[-1] if self.metrics_history else {},
            "return_code": self.return_code,
            "config": self.config,
            "num_log_lines": len(self.log_lines),
            "checkpoints": self.checkpoints,
        }

    def write_lora_config(self, lora_config, config_dir):
        """Write LoRA config JSON to disk."""
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "lora_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(lora_config, f, indent=2)
        return config_path

    def start_training(self, config, acestep_dir, work_dir):
        """Launch trainer.py as a subprocess."""
        if self.is_running:
            return {"success": False, "error": "Training already in progress"}

        # Reset state
        self.metrics_history = []
        self.log_lines = []
        self.current_step = 0
        self.return_code = None
        self.checkpoints = []
        self.config = config
        self.max_steps = config.get("max_steps", 5000)

        # Write LoRA config
        lora_config = {
            "r": config.get("lora_r", 64),
            "lora_alpha": config.get("lora_alpha", 32),
            "lora_dropout": config.get("lora_dropout", 0.1),
            "target_modules": config.get("target_modules", [
                "speaker_embedder", "linear_q", "linear_k", "linear_v",
                "to_q", "to_k", "to_v", "to_out.0"
            ]),
            "use_rslora": config.get("use_rslora", True),
        }
        config_dir = os.path.join(work_dir, "configs")
        lora_config_path = self.write_lora_config(lora_config, config_dir)

        # Resolve dataset path
        dataset_path = config.get("dataset_path", "")
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(work_dir, "datasets", dataset_path)

        # Logger dir
        logger_dir = config.get("logger_dir", os.path.join(work_dir, "exps", "logs"))
        os.makedirs(logger_dir, exist_ok=True)

        # Build command
        cmd = [
            sys.executable, "trainer.py",
            "--dataset_path", dataset_path,
            "--exp_name", config.get("exp_name", "lora_experiment"),
            "--learning_rate", str(config.get("learning_rate", 1e-4)),
            "--max_steps", str(self.max_steps),
            "--precision", config.get("precision", "bf16-true"),
            "--accumulate_grad_batches", str(config.get("accumulate_grad_batches", 4)),
            "--gradient_clip_val", str(config.get("gradient_clip_val", 0.5)),
            "--gradient_clip_algorithm", config.get("gradient_clip_algorithm", "norm"),
            "--lr_scheduler", config.get("lr_scheduler", "cosine_restarts"),
            "--shift", str(config.get("shift", 3.0)),
            "--num_workers", "0",  # Must be 0 on Windows (py3langid can't pickle across workers)
            "--every_n_train_steps", str(config.get("save_every", 500)),
            "--every_plot_step", str(config.get("plot_every", 1000)),
            # PL 2.5+ requires save_top_k ∈ {-1, 0, 1} when monitor=None.
            # Use -1 (keep all checkpoints) — LoRA adapters are small.
            "--save_top_k", "-1",
            "--lora_config_path", lora_config_path,
            "--devices", "1",
            "--logger_dir", logger_dir,
        ]

        checkpoint_dir = config.get("checkpoint_dir", "")
        if checkpoint_dir:
            cmd += ["--checkpoint_dir", checkpoint_dir]

        ckpt_path = config.get("ckpt_path", "")
        if ckpt_path:
            cmd += ["--ckpt_path", ckpt_path]

        try:
            env = os.environ.copy()
            # Use the stdlib subprocess (not eventlet-patched) to avoid blocking
            import subprocess as _subprocess
            self.process = _subprocess.Popen(
                cmd,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=acestep_dir,
                env=env,
            )
            self.is_running = True
            self.config["logger_dir"] = logger_dir

            # Set visualization snapshot save directory
            try:
                from backend.visualization_service import get_visualizer
                exp_name = config.get("exp_name", "lora_experiment")
                # Match the PL log dir pattern: lightning_logs/<timestamp><exp_name>
                import datetime
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                viz_save_dir = os.path.join(logger_dir, "lightning_logs", f"{ts}{exp_name}", "viz_snapshots")
                get_visualizer().set_save_dir(viz_save_dir)
            except Exception as e:
                print(f"[TrainerService] Could not set viz save dir (non-fatal): {e}")

            # Use eventlet greenlets for log streaming and GPU polling
            # _stream_output uses tpool for the blocking readline, then yields
            eventlet.spawn_n(self._stream_output)
            eventlet.spawn_n(self._poll_gpu)

            return {"success": True, "pid": self.process.pid}

        except Exception as e:
            self.is_running = False
            return {"success": False, "error": str(e)}

    def stop_training(self):
        """Terminate the training subprocess."""
        if self.process and self.is_running:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.is_running = False
            self.return_code = -1
            self._emit("training_complete", {"return_code": -1, "stopped": True})
            return {"success": True}
        return {"success": False, "error": "No training in progress"}

    @staticmethod
    def _is_progress_bar(line):
        """Detect HuggingFace/tqdm progress bars and download spinners."""
        # Lines that are mostly progress bar characters
        if not line or len(line) < 3:
            return True
        # tqdm-style: "  5%|█▌       | 12/240 [00:01<...]"
        # HF download: "Downloading (…)model.safetensors:  15%"
        bar_chars = set("█▌▎░▒▓╸╺━─│|\\/-=<>[] ")
        non_bar = sum(1 for c in line if c not in bar_chars)
        if non_bar < len(line) * 0.3 and len(line) > 20:
            return True
        # Lines with \r in them (carriage return updates)
        if "\r" in line:
            return True
        # Repeated download progress: "Downloading …: XX%"
        if "Downloading" in line and "%" in line:
            return True
        # Very short single-char or whitespace-only updates
        if line.strip() in ("", "|", "/", "-", "\\"):
            return True
        return False

    def _stream_output(self):
        """Read subprocess stdout line-by-line, parse metrics, emit via SocketIO.

        Uses eventlet.tpool for blocking readline so the event loop stays responsive.
        Filters out progress bar spam to keep logs manageable.
        """
        from eventlet import tpool

        MAX_LOG_LINES = 10000  # Cap stored lines to avoid memory bloat
        emit_count = 0

        try:
            while True:
                # Read one line in a real OS thread to avoid blocking eventlet
                line = tpool.execute(self.process.stdout.readline)
                if not line:
                    break  # EOF — process has closed stdout
                line = line.rstrip("\n\r")
                if not line:
                    continue

                # Filter out progress bar noise
                if self._is_progress_bar(line):
                    continue

                # Cap stored log lines
                if len(self.log_lines) < MAX_LOG_LINES:
                    self.log_lines.append(line)
                elif len(self.log_lines) == MAX_LOG_LINES:
                    self.log_lines.append("[...log truncated at 10000 lines...]")

                # Parse metrics from line
                metrics = self._parse_metrics(line)

                # Throttle Socket.IO emissions — don't emit every single line during
                # rapid output (model loading). Emit important lines always.
                emit_count += 1
                is_important = metrics or "Error" in line or "Epoch" in line or "step" in line.lower()
                if is_important or emit_count % 5 == 0:
                    self._emit("training_log", {
                        "line": line,
                        "metrics": metrics,
                        "step": self.current_step,
                        "max_steps": self.max_steps,
                    })

                # Yield to the eventlet event loop
                eventlet.sleep(0)

            self.process.wait()
            self.return_code = self.process.returncode
        except Exception as e:
            self.return_code = -99
            self.log_lines.append(f"[TrainerUI] Internal error: {e}")
        finally:
            self.is_running = False

            # Build error summary for fast-fail cases (import errors, crashes)
            error_summary = None
            if self.return_code and self.return_code != 0:
                # Extract last meaningful error lines
                err_lines = [l for l in self.log_lines if "Error" in l or "error" in l.lower()
                             or "Traceback" in l or "ImportError" in l or "ModuleNotFoundError" in l]
                if err_lines:
                    error_summary = err_lines[-1][:200]
                elif self.log_lines:
                    error_summary = self.log_lines[-1][:200]

            self._emit("training_complete", {
                "return_code": self.return_code,
                "stopped": False,
                "error_summary": error_summary,
            })

            # Assemble visualization animation from saved PNG snapshots
            try:
                from backend.visualization_service import get_visualizer
                viz = get_visualizer()
                if viz.save_dir:
                    created = viz.create_animation(viz.save_dir)
                    if created and self._socketio:
                        self._socketio.emit(
                            "animation_ready",
                            {"success": True, "created": created},
                            namespace="/visualization",
                        )
                        print(f"[TrainerService] Animation created: {list(created.keys())}")
            except Exception as e:
                print(f"[TrainerService] Animation creation failed (non-fatal): {e}")

    def _parse_metrics(self, line):
        """Extract step number and loss values from PyTorch Lightning log output."""
        metrics = {}

        # Match epoch and per-epoch step from PL progress lines:
        # "Epoch 0:  40%|████| 2/5 [00:06<...]"
        # "Epoch 312: 100%|████| 5/5 [00:15<...]"
        # With small datasets, the denominator is samples-per-epoch (e.g. 5),
        # NOT max_steps. We compute global_step = epoch * steps_per_epoch + step_in_epoch.
        if "Epoch" in line:
            epoch_match = re.search(r"Epoch\s+(\d+)", line)
            step_match = re.search(r"(\d+)/(\d+)\s*\[", line)
            if epoch_match:
                self.current_epoch = int(epoch_match.group(1))
                metrics["epoch"] = self.current_epoch
            if step_match:
                step_in_epoch = int(step_match.group(1))
                steps_per_epoch = int(step_match.group(2))
                if steps_per_epoch > 0 and steps_per_epoch <= self.max_steps:
                    global_step = self.current_epoch * steps_per_epoch + step_in_epoch
                    self.current_step = min(global_step, self.max_steps)
                    metrics["step"] = self.current_step
        else:
            # Match epoch from non-progress lines (e.g. checkpoint logs)
            epoch_match = re.search(r"Epoch\s+(\d+)", line)
            if epoch_match:
                self.current_epoch = int(epoch_match.group(1))
                metrics["epoch"] = self.current_epoch

        # Match loss values: train/loss=0.0342
        loss_matches = re.findall(r"train/([\w]+)=([\d.e\-+]+)", line)
        for key, val in loss_matches:
            try:
                metrics[key] = float(val)
            except ValueError:
                pass

        # Also match standalone loss= patterns
        standalone = re.findall(r"(?<!\w)(loss|denoising_loss|learning_rate)=([\d.e\-+]+)", line)
        for key, val in standalone:
            try:
                metrics[key] = float(val)
            except ValueError:
                pass

        # Detect checkpoint saves
        if "epoch=" in line and "_lora" in line and ("Saving" in line or "saved" in line.lower()):
            ckpt_match = re.search(r"(epoch=\d+-step=\d+_lora)", line)
            if ckpt_match:
                ckpt_name = ckpt_match.group(1)
                if ckpt_name not in self.checkpoints:
                    self.checkpoints.append(ckpt_name)
                    self._emit("checkpoint_saved", {
                        "name": ckpt_name,
                        "step": self.current_step,
                    })

        if metrics:
            self.metrics_history.append({
                "timestamp": time.time(),
                **metrics,
            })

        return metrics if metrics else None

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

    def get_metrics_history(self):
        return self.metrics_history

    def get_log_lines(self, offset=0):
        return self.log_lines[offset:]

    def list_checkpoints(self):
        """Scan logger_dir for saved LoRA checkpoints."""
        logger_dir = self.config.get("logger_dir", "")
        if not logger_dir or not os.path.exists(logger_dir):
            return []

        found = []
        for root, dirs, files in os.walk(logger_dir):
            for d in dirs:
                if d.endswith("_lora"):
                    full_path = os.path.join(root, d)
                    found.append({
                        "name": d,
                        "path": full_path,
                        "size_mb": round(sum(
                            os.path.getsize(os.path.join(full_path, f))
                            for f in os.listdir(full_path)
                            if os.path.isfile(os.path.join(full_path, f))
                        ) / (1024 * 1024), 1) if os.path.exists(full_path) else 0,
                    })
        return sorted(found, key=lambda x: x["name"])


def get_gpu_stats():
    """Query nvidia-smi for GPU memory and utilization."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 5:
            return {
                "mem_used_mb": int(parts[0]),
                "mem_total_mb": int(parts[1]),
                "gpu_util_pct": int(parts[2]),
                "temp_c": int(parts[3]),
                "name": parts[4],
            }
    except Exception:
        pass
    return None


# Singleton
trainer_service = TrainerService()
