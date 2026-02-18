"""
Dataset API — Flask Blueprint for dataset CRUD, validation, conversion, and ZIP operations.
"""

import os
import tempfile
import threading
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from backend import dataset_service as ds

# eventlet.monkey_patch() replaces threading.Thread with a green-thread version.
# HuggingFace Datasets uses memory-mapped Arrow files that break under eventlet's
# patched I/O on Windows (Errno 22).  Import the *real* OS thread class so the
# conversion runs outside eventlet's cooperative scheduler.
try:
    from eventlet.patcher import original as _original
    _RealThread = _original("threading").Thread
except Exception:
    _RealThread = threading.Thread

dataset_bp = Blueprint("dataset", __name__)

# ===== Background conversion state =====
import threading as _threading
_convert_lock = _threading.Lock()
_convert_state = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
}


@dataset_bp.route("/upload", methods=["POST"])
def upload_files():
    """Upload individual dataset files (mp3, txt) to the working data directory."""
    if not request.files:
        return jsonify({"error": "No files provided"}), 400

    data_dir = ds.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    saved = []
    for key in request.files:
        file_obj = request.files[key]
        if file_obj.filename:
            safe_name = secure_filename(file_obj.filename)
            if not safe_name:
                continue
            dest = os.path.join(data_dir, safe_name)
            file_obj.save(dest)
            saved.append(safe_name)

    return jsonify({"saved": saved, "count": len(saved)})


@dataset_bp.route("/upload-zip", methods=["POST"])
def upload_zip():
    """Upload a ZIP file containing dataset files. Extracts to data dir."""
    if "file" not in request.files:
        return jsonify({"error": "No ZIP file provided"}), 400

    zip_file = request.files["file"]
    if not zip_file.filename.lower().endswith(".zip"):
        return jsonify({"error": "File must be a .zip"}), 400

    # Save to temp location then extract
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    zip_file.save(tmp.name)
    tmp.close()

    try:
        stems = ds.extract_zip(tmp.name, clear_first=True)
        return jsonify({"stems": stems, "count": len(stems)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp.name)


@dataset_bp.route("/validate", methods=["GET"])
def validate():
    """Validate the current data directory."""
    result = ds.validate_dataset()
    return jsonify(result)


@dataset_bp.route("/convert", methods=["POST"])
def convert():
    """
    Start dataset conversion in a background thread.
    Returns immediately with status. Poll /convert-status for progress.
    """
    body = request.get_json() or {}
    output_name = body.get("output_name", "lora_dataset")
    trigger_word = body.get("trigger_word", "")

    # Capture app config values we need inside the thread
    work_dir = current_app.config["WORK_DIR"]

    with _convert_lock:
        if _convert_state["running"]:
            return jsonify({"success": False, "error": "Conversion already in progress"}), 409
        _convert_state["running"] = True
        _convert_state["progress"] = "Starting conversion..."
        _convert_state["result"] = None
        _convert_state["error"] = None

    def run_conversion():
        def _update_progress(msg):
            with _convert_lock:
                _convert_state["progress"] = msg

        try:
            _update_progress("Reading data files...")
            result = ds.convert_to_hf_dataset_standalone(
                work_dir=work_dir,
                output_name=output_name,
                progress_callback=_update_progress,
                trigger_word=trigger_word,
            )
            with _convert_lock:
                _convert_state["result"] = result
                _convert_state["progress"] = "Done"
        except Exception as e:
            with _convert_lock:
                _convert_state["error"] = str(e)
                _convert_state["progress"] = f"Error: {e}"
        finally:
            with _convert_lock:
                _convert_state["running"] = False

    # Must use a *real* OS thread (not eventlet's green-thread wrapper) so that
    # HF Datasets' memory-mapped Arrow I/O isn't routed through eventlet's patched
    # file descriptors, which causes Errno 22 on Windows.
    t = _RealThread(target=run_conversion, daemon=True)
    t.start()

    return jsonify({"success": True, "status": "started"})


@dataset_bp.route("/convert-status", methods=["GET"])
def convert_status():
    """Poll for conversion progress."""
    with _convert_lock:
        return jsonify({
            "running": _convert_state["running"],
            "progress": _convert_state["progress"],
            "result": _convert_state["result"],
            "error": _convert_state["error"],
        })


@dataset_bp.route("/info", methods=["GET"])
def dataset_info():
    """Get info about a converted HF dataset."""
    name = request.args.get("name", "lora_dataset")
    try:
        info = ds.get_dataset_info(name)
        if info is None:
            return jsonify({"found": False}), 200  # Not 404 — just not converted yet
        return jsonify({"found": True, **info})
    except Exception as e:
        return jsonify({"found": False, "error": str(e)}), 200


@dataset_bp.route("/list", methods=["GET"])
def list_datasets():
    """List all converted HF datasets."""
    try:
        datasets = ds.list_datasets()
        return jsonify({"datasets": datasets})
    except Exception as e:
        return jsonify({"datasets": [], "error": str(e)}), 500


@dataset_bp.route("/clear", methods=["DELETE"])
def clear():
    """Clear the data working directory."""
    ds.clear_data_dir()
    return jsonify({"success": True})


@dataset_bp.route("/save-sample", methods=["POST"])
def save_sample():
    """
    Save a single sample's files (mp3, prompt txt, lyrics txt) from multipart upload.
    Used by the dataset editor for individual sample uploads.
    """
    data_dir = ds.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    stem = request.form.get("stem", "")
    if not stem:
        return jsonify({"error": "Missing stem name"}), 400

    saved = []

    # Save audio file if provided
    if "audio" in request.files:
        audio = request.files["audio"]
        dest = os.path.join(data_dir, f"{stem}.mp3")
        audio.save(dest)
        saved.append(f"{stem}.mp3")

    # Save prompt text
    prompt = request.form.get("prompt", "")
    prompt_path = os.path.join(data_dir, f"{stem}_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    saved.append(f"{stem}_prompt.txt")

    # Save lyrics text
    lyrics = request.form.get("lyrics", "")
    lyrics_path = os.path.join(data_dir, f"{stem}_lyrics.txt")
    with open(lyrics_path, "w", encoding="utf-8") as f:
        f.write(lyrics)
    saved.append(f"{stem}_lyrics.txt")

    return jsonify({"saved": saved, "stem": stem})


@dataset_bp.route("/delete-sample", methods=["DELETE"])
def delete_sample():
    """Delete a sample's files (mp3 + prompt + lyrics) by stem name."""
    stem = request.args.get("stem", "")
    if not stem:
        return jsonify({"error": "Missing stem name"}), 400

    data_dir = ds.get_data_dir()
    deleted = []
    for ext in [".mp3", "_prompt.txt", "_lyrics.txt"]:
        path = os.path.join(data_dir, f"{stem}{ext}")
        if os.path.exists(path):
            os.remove(path)
            deleted.append(f"{stem}{ext}")

    return jsonify({"deleted": deleted})


@dataset_bp.route("/files", methods=["GET"])
def list_files():
    """List all files currently in the data directory."""
    data_dir = ds.get_data_dir()
    if not os.path.exists(data_dir):
        return jsonify({"files": []})

    files = []
    for f in sorted(os.listdir(data_dir)):
        fp = os.path.join(data_dir, f)
        if os.path.isfile(fp):
            files.append({
                "name": f,
                "size": os.path.getsize(fp),
            })
    return jsonify({"files": files})


@dataset_bp.route("/audio/<stem>", methods=["GET"])
def serve_audio(stem):
    """Serve an audio file for playback in the browser."""
    data_dir = ds.get_data_dir()
    try:
        return send_from_directory(data_dir, f"{stem}.mp3", mimetype="audio/mpeg")
    except FileNotFoundError:
        return jsonify({"error": "Audio file not found"}), 404
