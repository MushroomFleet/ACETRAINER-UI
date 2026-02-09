"""
Dataset API — Flask Blueprint for dataset CRUD, validation, conversion, and ZIP operations.
"""

import os
import tempfile
from flask import Blueprint, request, jsonify, current_app, send_file
from backend import dataset_service as ds

dataset_bp = Blueprint("dataset", __name__)


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
            dest = os.path.join(data_dir, file_obj.filename)
            file_obj.save(dest)
            saved.append(file_obj.filename)

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
    """Convert raw data files to HuggingFace dataset format."""
    body = request.get_json() or {}
    repeat_count = body.get("repeat_count", 2000)
    output_name = body.get("output_name", "lora_dataset")

    result = ds.convert_to_hf_dataset(repeat_count=repeat_count, output_name=output_name)
    return jsonify(result)


@dataset_bp.route("/info", methods=["GET"])
def dataset_info():
    """Get info about a converted HF dataset."""
    name = request.args.get("name", "lora_dataset")
    info = ds.get_dataset_info(name)
    if info is None:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(info)


@dataset_bp.route("/list", methods=["GET"])
def list_datasets():
    """List all converted HF datasets."""
    datasets = ds.list_datasets()
    return jsonify({"datasets": datasets})


@dataset_bp.route("/clear", methods=["DELETE"])
def clear():
    """Clear the data working directory."""
    ds.clear_data_dir()
    return jsonify({"success": True})


@dataset_bp.route("/recommend-repeat", methods=["GET"])
def recommend_repeat():
    """Get recommended repeat_count based on sample count."""
    num_samples = request.args.get("num_samples", 0, type=int)
    max_steps = request.args.get("max_steps", 5000, type=int)
    count = ds.recommend_repeat_count(num_samples, max_steps)
    return jsonify({"repeat_count": count})


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
    mp3_path = os.path.join(data_dir, f"{stem}.mp3")
    if not os.path.exists(mp3_path):
        return jsonify({"error": "Audio file not found"}), 404
    return send_file(mp3_path, mimetype="audio/mpeg")
