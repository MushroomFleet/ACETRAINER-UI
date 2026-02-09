"""
Dataset service — handles file storage, validation, HF dataset conversion, and ZIP operations.
"""

import os
import json
import shutil
import zipfile
import math
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


def recommend_repeat_count(num_samples, max_steps=5000):
    """Recommend a repeat_count based on number of samples."""
    if num_samples <= 0:
        return 2000
    count = max(100, math.ceil(max_steps / num_samples))
    return min(count, 10000)


def convert_to_hf_dataset(repeat_count=2000, output_name="lora_dataset"):
    """
    Convert raw data files to HuggingFace dataset format.
    Uses the same logic as ACE-Step's convert2hf_dataset.py.
    """
    data_dir = get_data_dir()
    output_path = os.path.join(get_datasets_dir(), output_name)

    # Import datasets library
    from datasets import Dataset

    data_path = Path(data_dir)
    all_examples = []

    for song_path in sorted(data_path.glob("*.mp3")):
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
        except Exception as e:
            continue

    if not all_examples:
        return {"success": False, "error": "No valid samples found in data directory"}

    ds = Dataset.from_list(all_examples * repeat_count)
    ds.save_to_disk(output_path)

    return {
        "success": True,
        "path": output_path,
        "num_samples": len(all_examples),
        "repeat_count": repeat_count,
        "total_rows": len(all_examples) * repeat_count,
    }


def get_dataset_info(dataset_name="lora_dataset"):
    """Get info about an existing HF dataset."""
    dataset_path = os.path.join(get_datasets_dir(), dataset_name)
    if not os.path.exists(dataset_path):
        return None

    try:
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)
        unique_keys = set(ds["keys"])
        return {
            "path": dataset_path,
            "total_rows": len(ds),
            "unique_samples": len(unique_keys),
            "repeat_count": len(ds) // max(len(unique_keys), 1),
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
