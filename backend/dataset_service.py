"""
Dataset service — handles file storage, validation, HF dataset conversion, and ZIP operations.
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from mutagen.mp3 import MP3


def get_work_dir():
    from flask import current_app
    return current_app.config["WORK_DIR"]


def get_acestep_dir():
    from flask import current_app
    return current_app.config["ACESTEP_DIR"]


def get_data_dir():
    return os.path.join(get_work_dir(), "data")


def get_datasets_dir():
    return os.path.join(get_work_dir(), "datasets")


def clear_data_dir():
    """Remove all files in the data working directory."""
    data_dir = get_data_dir()
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)


def save_uploaded_files(files_dict):
    """
    Save uploaded files to the data working directory.
    files_dict: dict of filename -> file storage object
    Returns list of saved filenames.
    """
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    saved = []
    for filename, file_obj in files_dict.items():
        dest = os.path.join(data_dir, filename)
        file_obj.save(dest)
        saved.append(filename)
    return saved


def save_raw_files(file_data_list):
    """
    Save raw file data from the client upload.
    file_data_list: list of dicts with 'filename' and 'data' (bytes).
    """
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    saved = []
    for item in file_data_list:
        dest = os.path.join(data_dir, item["filename"])
        with open(dest, "wb") as f:
            f.write(item["data"])
        saved.append(item["filename"])
    return saved


def extract_zip(zip_path, clear_first=True):
    """
    Extract a ZIP file containing dataset files into the data dir.
    Returns list of extracted MP3 stems found.
    """
    data_dir = get_data_dir()
    if clear_first:
        clear_data_dir()

    stems = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Flatten directory structure — extract just the filename
            basename = os.path.basename(info.filename)
            if not basename:
                continue
            # Only extract mp3, txt files
            ext = basename.lower().split(".")[-1] if "." in basename else ""
            if ext not in ("mp3", "txt"):
                continue
            # Extract to data dir
            dest = os.path.join(data_dir, basename)
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if ext == "mp3":
                stems.add(basename.rsplit(".", 1)[0])
    return list(stems)


def validate_dataset():
    """
    Validate the data directory for ACE-Step training format.
    Returns a dict with validation results.
    """
    data_dir = get_data_dir()
    if not os.path.exists(data_dir):
        return {"valid": False, "total": 0, "errors": ["Data directory does not exist"]}

    mp3_files = sorted(Path(data_dir).glob("*.mp3"))
    results = {
        "total": len(mp3_files),
        "valid_count": 0,
        "invalid_count": 0,
        "missing_prompt": [],
        "missing_lyrics": [],
        "samples": [],
        "total_duration_sec": 0,
        "errors": [],
    }

    for mp3_path in mp3_files:
        stem = mp3_path.stem
        prompt_path = mp3_path.with_name(f"{stem}_prompt.txt")
        lyrics_path = mp3_path.with_name(f"{stem}_lyrics.txt")

        has_prompt = prompt_path.exists() and prompt_path.read_text(encoding="utf-8").strip() != ""
        has_lyrics = lyrics_path.exists()

        # Get duration
        duration = 0
        try:
            audio = MP3(str(mp3_path))
            duration = audio.info.length
        except Exception:
            duration = 0

        sample_info = {
            "stem": stem,
            "has_prompt": has_prompt,
            "has_lyrics": has_lyrics,
            "duration": round(duration, 1),
        }

        prompt_text = ""
        lyrics_text = ""
        if has_prompt:
            prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        if has_lyrics:
            lyrics_text = lyrics_path.read_text(encoding="utf-8").strip()
        sample_info["prompt"] = prompt_text
        sample_info["lyrics"] = lyrics_text

        is_valid = has_prompt and has_lyrics
        sample_info["valid"] = is_valid

        if not has_prompt:
            results["missing_prompt"].append(stem)
        if not has_lyrics:
            results["missing_lyrics"].append(stem)

        if is_valid:
            results["valid_count"] += 1
        else:
            results["invalid_count"] += 1

        results["total_duration_sec"] += duration
        results["samples"].append(sample_info)

    results["valid"] = results["invalid_count"] == 0 and results["total"] > 0
    if results["total"] > 0:
        results["avg_duration_sec"] = round(results["total_duration_sec"] / results["total"], 1)
    else:
        results["avg_duration_sec"] = 0
    results["total_duration_sec"] = round(results["total_duration_sec"], 1)

    return results


def convert_to_hf_dataset(output_name="lora_dataset"):
    """
    Convert raw data files to HuggingFace dataset format (Flask context version).
    """
    work_dir = get_work_dir()
    return convert_to_hf_dataset_standalone(work_dir, output_name)


def convert_to_hf_dataset_standalone(work_dir, output_name="lora_dataset", progress_callback=None):
    """
    Convert raw data files to HuggingFace dataset format.
    Standalone version — no Flask context needed, safe for background threads.
    No duplication — stores one row per sample. Repetition is handled by
    PyTorch Lightning's epoch loop (epochs=-1, max_steps=N).
    """
    data_dir = os.path.join(work_dir, "data")
    datasets_dir = os.path.join(work_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    output_path = os.path.join(datasets_dir, output_name)

    def progress(msg):
        if progress_callback:
            progress_callback(msg)

    progress("Scanning data files...")

    # Import datasets library
    from datasets import Dataset

    data_path = Path(data_dir)
    all_examples = []
    mp3_files = sorted(data_path.glob("*.mp3"))

    if not mp3_files:
        return {"success": False, "error": "No MP3 files found in data directory"}

    progress(f"Found {len(mp3_files)} audio files, reading captions...")

    for i, song_path in enumerate(mp3_files):
        prompt_path = str(song_path).replace(".mp3", "_prompt.txt")
        lyric_path = str(song_path).replace(".mp3", "_lyrics.txt")
        try:
            if not os.path.exists(prompt_path):
                continue
            if not os.path.exists(lyric_path):
                continue
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            with open(lyric_path, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()

            keys = song_path.stem
            example = {
                "keys": keys,
                "filename": str(song_path),
                "tags": prompt.split(", "),
                "speaker_emb_path": "",
                "norm_lyrics": lyrics,
                "recaption": {},
            }
            all_examples.append(example)
        except Exception:
            continue

    if not all_examples:
        return {"success": False, "error": "No valid samples found (need .mp3 + _prompt.txt + _lyrics.txt)"}

    progress(f"Building dataset: {len(all_examples)} samples (no duplication)...")

    hf_dataset = Dataset.from_list(all_examples)

    progress(f"Saving {len(all_examples)} rows to disk...")

    # On Windows, HF Datasets memory-maps Arrow files which holds file locks,
    # so rmtree/overwrite of an existing dataset fails with Errno 22.
    # Strategy: save to a temp path, then swap via rename.
    tmp_new = output_path + "_new_tmp"
    if os.path.exists(tmp_new):
        shutil.rmtree(tmp_new, ignore_errors=True)

    hf_dataset.save_to_disk(tmp_new)

    # Swap: move old out of the way, move new into place
    tmp_old = output_path + "_old_tmp"
    if os.path.exists(tmp_old):
        shutil.rmtree(tmp_old, ignore_errors=True)
    if os.path.exists(output_path):
        try:
            os.rename(output_path, tmp_old)
        except OSError:
            # Old dir is locked — delete what we can, it will be cleaned up next time
            shutil.rmtree(output_path, ignore_errors=True)
    os.rename(tmp_new, output_path)

    # Best-effort cleanup of old dir
    if os.path.exists(tmp_old):
        shutil.rmtree(tmp_old, ignore_errors=True)

    progress("Done")

    return {
        "success": True,
        "path": output_path,
        "num_samples": len(all_examples),
        "total_rows": len(all_examples),
    }


def get_dataset_info(dataset_name="lora_dataset"):
    """Get info about an existing HF dataset."""
    dataset_path = os.path.join(get_datasets_dir(), dataset_name)
    if not os.path.exists(dataset_path):
        return None

    # Check for valid HF dataset marker files
    has_state = os.path.exists(os.path.join(dataset_path, "state.json"))
    has_info = os.path.exists(os.path.join(dataset_path, "dataset_info.json"))
    if not (has_state or has_info):
        # Empty or corrupt directory — not a valid dataset
        return None

    try:
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)
        unique_keys = set(ds["keys"])
        return {
            "path": dataset_path,
            "total_rows": len(ds),
            "unique_samples": len(unique_keys),
            "sample_keys": sorted(unique_keys),
        }
    except Exception as e:
        return {"error": str(e)}


def list_datasets():
    """List all converted HF datasets in the datasets directory."""
    datasets_dir = get_datasets_dir()
    if not os.path.exists(datasets_dir):
        return []
    datasets = []
    for name in os.listdir(datasets_dir):
        full_path = os.path.join(datasets_dir, name)
        if os.path.isdir(full_path):
            info = get_dataset_info(name)
            if info and "error" not in info:
                datasets.append({"name": name, **info})
    return datasets
