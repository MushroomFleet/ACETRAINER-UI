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
            "--precision", config.get("precision", "bf16-mixed"),
            "--accumulate_grad_batches", str(config.get("accumulate_grad_batches", 1)),
            "--gradient_clip_val", str(config.get("gradient_clip_val", 0.5)),
            "--gradient_clip_algorithm", config.get("gradient_clip_algorithm", "norm"),
            "--shift", str(config.get("shift", 3.0)),
            "--num_workers", str(config.get("num_workers", 4)),
            "--every_n_train_steps", str(config.get("save_every", 500)),
            "--every_plot_step", str(config.get("plot_every", 1000)),
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
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=acestep_dir,
                env=env,
            )
            self.is_running = True
            self.config["logger_dir"] = logger_dir

            # Start log streaming thread
            self._log_thread = threading.Thread(target=self._stream_output, daemon=True)
            self._log_thread.start()

            # Start GPU monitoring thread
            self._gpu_thread = threading.Thread(target=self._poll_gpu, daemon=True)
            self._gpu_thread.start()

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

    def _stream_output(self):
        """Read subprocess stdout line-by-line, parse metrics, emit via SocketIO."""
        try:
            for line in self.process.stdout:
                line = line.rstrip("\n\r")
                if not line:
                    continue
                self.log_lines.append(line)

                # Parse metrics from line
                metrics = self._parse_metrics(line)

                self._emit("training_log", {
                    "line": line,
                    "metrics": metrics,
                    "step": self.current_step,
                    "max_steps": self.max_steps,
                })

            self.process.wait()
            self.return_code = self.process.returncode
        except Exception as e:
            self.return_code = -99
        finally:
            self.is_running = False
            self._emit("training_complete", {
                "return_code": self.return_code,
                "stopped": False,
            })

    def _parse_metrics(self, line):
        """Extract step number and loss values from PyTorch Lightning log output."""
        metrics = {}

        # Match global_step from PL progress: "Epoch 0:  25%|... 1247/5000 ..."
        step_match = re.search(r"(\d+)/(\d+)\s", line)
        if step_match:
            self.current_step = int(step_match.group(1))
            metrics["step"] = self.current_step

        # Match epoch
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
            time.sleep(5)

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
