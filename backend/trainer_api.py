"""
Trainer API — Flask Blueprint for training control, monitoring, and TensorBoard.
"""

import os
import atexit
import subprocess
import sys
from flask import Blueprint, request, jsonify, current_app
from backend.trainer_service import trainer_service, get_gpu_stats

trainer_bp = Blueprint("trainer", __name__)

_tensorboard_process = None


def _cleanup_tensorboard():
    global _tensorboard_process
    if _tensorboard_process and _tensorboard_process.poll() is None:
        _tensorboard_process.terminate()


atexit.register(_cleanup_tensorboard)


@trainer_bp.route("/start", methods=["POST"])
def start_training():
    """Start a training run with the provided configuration."""
    config = request.get_json()
    if not config:
        return jsonify({"error": "No configuration provided"}), 400

    acestep_dir = current_app.config["ACESTEP_DIR"]
    work_dir = current_app.config["WORK_DIR"]

    # Inject socketio reference
    sio = current_app.config.get("SOCKETIO")
    if sio:
        trainer_service.set_socketio(sio)

    result = trainer_service.start_training(config, acestep_dir, work_dir)
    if result.get("success"):
        return jsonify(result)
    else:
        return jsonify(result), 400


@trainer_bp.route("/stop", methods=["POST"])
def stop_training():
    """Stop the current training run."""
    result = trainer_service.stop_training()
    return jsonify(result)


@trainer_bp.route("/status", methods=["GET"])
def training_status():
    """Get current training status."""
    return jsonify(trainer_service.get_status())


@trainer_bp.route("/metrics", methods=["GET"])
def training_metrics():
    """Get full metrics history for chart reconstruction."""
    offset = request.args.get("offset", 0, type=int)
    history = trainer_service.get_metrics_history()
    return jsonify({
        "metrics": history[offset:],
        "total": len(history),
    })


@trainer_bp.route("/logs", methods=["GET"])
def training_logs():
    """Get training log lines."""
    offset = request.args.get("offset", 0, type=int)
    lines = trainer_service.get_log_lines(offset)
    return jsonify({
        "lines": lines,
        "total": len(trainer_service.log_lines),
    })


@trainer_bp.route("/checkpoints", methods=["GET"])
def list_checkpoints():
    """List saved LoRA checkpoints."""
    checkpoints = trainer_service.list_checkpoints()
    return jsonify({"checkpoints": checkpoints})


@trainer_bp.route("/gpu", methods=["GET"])
def gpu_stats():
    """Get current GPU stats."""
    stats = get_gpu_stats()
    if stats:
        return jsonify(stats)
    return jsonify({"error": "Could not query GPU"}), 500


@trainer_bp.route("/tensorboard", methods=["POST"])
def launch_tensorboard():
    """Launch TensorBoard pointing at the experiment logs directory."""
    global _tensorboard_process

    body = request.get_json() or {}
    log_dir = body.get("log_dir", "")

    if not log_dir:
        work_dir = current_app.config["WORK_DIR"]
        log_dir = os.path.join(work_dir, "exps", "logs")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Kill existing tensorboard if running
    if _tensorboard_process and _tensorboard_process.poll() is None:
        _tensorboard_process.terminate()

    try:
        _tensorboard_process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006", "--bind_all"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return jsonify({"success": True, "url": "http://localhost:6006", "pid": _tensorboard_process.pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trainer_bp.route("/tensorboard/stop", methods=["POST"])
def stop_tensorboard():
    """Stop the TensorBoard subprocess."""
    global _tensorboard_process
    if _tensorboard_process and _tensorboard_process.poll() is None:
        _tensorboard_process.terminate()
        _tensorboard_process = None
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "TensorBoard not running"})
