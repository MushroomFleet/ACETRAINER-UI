"""
Flask/Socket.IO API for PaCMAP visualization.
Blueprint: /api/viz/*
Socket.IO events on /visualization namespace.
"""

from flask import Blueprint, jsonify, request
from flask_socketio import emit
import numpy as np
import traceback

from backend.visualization_service import get_visualizer

viz_bp = Blueprint("visualization", __name__)


# --- REST Endpoints ---


@viz_bp.route("/api/viz/status", methods=["GET"])
def viz_status():
    try:
        stats = get_visualizer().get_stats()
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/clear/<space>", methods=["POST"])
def viz_clear(space):
    try:
        viz = get_visualizer()
        if space == "all":
            viz.clear_all()
        elif space in ("latent", "lora", "prompt"):
            viz.clear_space(space)
        else:
            return jsonify({"success": False, "error": f"Invalid space: {space}"}), 400
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/update", methods=["POST"])
def viz_force_update():
    try:
        updated = get_visualizer().update_projections(force=True)
        return jsonify({"success": True, "updated": updated})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/save", methods=["POST"])
def viz_save():
    """Manually trigger a snapshot save."""
    try:
        viz = get_visualizer()
        if not viz.save_dir:
            return jsonify({"success": False, "error": "No save directory configured. Start training first."}), 400
        latest_step = 0
        for space in ["latent", "lora", "prompt"]:
            if viz.metadata[space]:
                latest_step = max(latest_step, viz.metadata[space][-1].get("step", 0))
        saved = viz.save_snapshot(viz.save_dir, latest_step)
        return jsonify({"success": True, "saved": saved})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/set_save_dir", methods=["POST"])
def viz_set_save_dir():
    """Set the snapshot save directory (called by trainer_service when training starts)."""
    data = request.get_json() or {}
    save_dir = data.get("save_dir")
    if not save_dir:
        return jsonify({"success": False, "error": "save_dir required"}), 400
    get_visualizer().set_save_dir(save_dir)
    return jsonify({"success": True})


@viz_bp.route("/api/viz/create_animation", methods=["POST"])
def viz_create_animation():
    """Stitch saved PNGs into animated GIF + WebM. Called at training end or manually."""
    try:
        viz = get_visualizer()
        if not viz.save_dir:
            return jsonify({"success": False, "error": "No save directory configured."}), 400
        created = viz.create_animation(viz.save_dir)
        if not created:
            return jsonify({"success": False, "error": "No animation created (need >= 2 PNG frames)."}), 400
        return jsonify({"success": True, "created": created})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@viz_bp.route("/api/viz/toggle_auto_save", methods=["POST"])
def viz_toggle_auto_save():
    """Toggle auto-save on/off."""
    data = request.get_json() or {}
    enabled = data.get("enabled", True)
    get_visualizer().auto_save = bool(enabled)
    return jsonify({"success": True, "auto_save": get_visualizer().auto_save})


# --- Socket.IO Handlers ---


def register_viz_socketio_handlers(socketio):
    """Register Socket.IO handlers. Call from app.py after creating socketio."""

    @socketio.on("request_visualization", namespace="/visualization")
    def handle_viz_request():
        try:
            viz = get_visualizer()
            viz.update_projections(force=False)
            figure_json = viz.generate_combined_figure()
            if figure_json:
                emit(
                    "visualization_update",
                    {"success": True, "figure": figure_json, "stats": viz.get_stats()},
                )
            else:
                emit(
                    "visualization_update",
                    {"success": False, "error": "Insufficient data (need >= 10 samples per space)"},
                )
        except Exception as e:
            print(f"[VizAPI] Error: {e}")
            traceback.print_exc()
            emit("visualization_update", {"success": False, "error": str(e)})

    @socketio.on("embedding_captured", namespace="/visualization")
    def handle_embedding_captured(data):
        """Receive embeddings from trainer subprocess via Socket.IO client."""
        try:
            space = data.get("space")
            embeddings = np.array(data.get("embeddings"))
            metadata = data.get("metadata", {})
            step = metadata.get("step", 0)

            viz = get_visualizer()

            if space == "latent":
                viz.add_latent_batch(
                    embeddings=embeddings,
                    step=step,
                    losses=metadata.get("losses"),
                    sample_ids=metadata.get("sample_ids"),
                )
            elif space == "lora":
                viz.add_lora_batch(
                    outputs=embeddings,
                    step=step,
                    layer_name=metadata.get("layer_name", "unknown"),
                    sample_ids=metadata.get("sample_ids"),
                )
            elif space == "prompt":
                viz.add_prompt_batch(
                    embeddings=embeddings,
                    step=step,
                    prompts=metadata.get("prompts", []),
                    sample_ids=metadata.get("sample_ids"),
                )
            else:
                return

            # Auto-update projections if enough samples accumulated
            updated = viz.update_projections(force=False)
            if any(updated.values()):
                figure_json = viz.generate_combined_figure()
                if figure_json:
                    emit(
                        "visualization_update",
                        {"success": True, "figure": figure_json, "stats": viz.get_stats()},
                        broadcast=True,
                        namespace="/visualization",
                    )
        except Exception as e:
            print(f"[VizAPI] Error handling embedding: {e}")
            traceback.print_exc()

    @socketio.on("connect", namespace="/visualization")
    def on_viz_connect():
        """Send current stats when a client connects to the viz namespace."""
        try:
            stats = get_visualizer().get_stats()
            emit("viz_status_update", {"stats": stats})
        except Exception:
            pass
