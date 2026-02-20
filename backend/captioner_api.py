"""
Captioner API — Flask Blueprint for ImageBind audio classification captioning.
"""

import os
import tempfile
import threading
from flask import Blueprint, request, jsonify, current_app

from backend.captioner.service import generate_caption, caption_batch, get_batch_state

try:
    from eventlet.patcher import original as _original
    _RealThread = _original("threading").Thread
except Exception:
    _RealThread = threading.Thread

captioner_bp = Blueprint("captioner", __name__)


@captioner_bp.route("/caption", methods=["POST"])
def caption_single():
    """
    Caption a single audio file. Accepts multipart form with:
    - audio: MP3 file blob
    - is_instrumental: "true" or "false" (optional, default false)

    Returns JSON with caption string, classification details, and audio info.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    is_instrumental = request.form.get("is_instrumental", "false").lower() == "true"

    # Save to temp file within the workspace
    tmp_dir = os.path.join(current_app.config["WORK_DIR"], "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=tmp_dir)
    audio_file.save(tmp.name)
    tmp.close()

    cache_dir = os.path.join(current_app.config["WORK_DIR"], "captioner_cache")

    try:
        caption, details, audio_info = generate_caption(
            tmp.name, is_instrumental=is_instrumental, cache_dir=cache_dir
        )
        return jsonify({
            "caption": caption,
            "details": details,
            "audio_info": audio_info,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@captioner_bp.route("/caption-all", methods=["POST"])
def caption_all():
    """
    Start batch captioning of all MP3 files currently on the server.
    Runs in a background thread. Poll /caption-all/status for progress.

    Expects JSON body with optional:
    - stems: list of stem names to caption (defaults to all MP3s in data dir)
    """
    state = get_batch_state()
    if state["running"]:
        return jsonify({"error": "Batch captioning already in progress"}), 409

    work_dir = current_app.config["WORK_DIR"]
    data_dir = os.path.join(work_dir, "data")
    cache_dir = os.path.join(work_dir, "captioner_cache")

    body = request.get_json() or {}
    stems = body.get("stems", None)

    if stems is None:
        # Discover all MP3s in data dir
        if os.path.exists(data_dir):
            stems = [
                f.rsplit(".", 1)[0]
                for f in os.listdir(data_dir)
                if f.lower().endswith(".mp3")
            ]
        else:
            stems = []

    if not stems:
        return jsonify({"error": "No MP3 files found to caption"}), 400

    def run():
        caption_batch(stems, data_dir, cache_dir=cache_dir)

    t = _RealThread(target=run, daemon=True)
    t.start()

    return jsonify({"success": True, "total": len(stems)})


@captioner_bp.route("/caption-all/status", methods=["GET"])
def caption_all_status():
    """Poll for batch captioning progress."""
    return jsonify(get_batch_state())
