"""
High-level captioner service: orchestrates preprocessing, embedding, classification.
Exposes a simple generate_caption() function for the API layer.
"""

import os
from backend.captioner.audio_preprocessor import preprocess_mp3, cleanup_chunks
from backend.captioner.embedder import ImageBindEmbedder
from backend.captioner.classifier import classify_song


# Module-level state for the background batch captioning job
_batch_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_file": "",
    "results": {},  # stem -> caption string
    "error": None,
}


def get_batch_state() -> dict:
    return dict(_batch_state)


def generate_caption(
    mp3_path: str,
    is_instrumental: bool = False,
    cache_dir: str = None,
) -> tuple:
    """
    Full pipeline: preprocess -> embed -> classify -> build caption.

    Args:
        mp3_path: Path to the MP3 file on the server filesystem.
        is_instrumental: Force vocal field to "instrumental".
        cache_dir: Directory for text embedding cache persistence.

    Returns:
        (caption_string, detail_dict, audio_info_dict)
    """
    embedder = ImageBindEmbedder.get_instance(cache_dir=cache_dir)
    text_cache = embedder.get_text_cache()

    # Preprocess
    audio_info = preprocess_mp3(mp3_path)

    try:
        # Embed audio chunks
        chunk_embeddings = embedder.embed_audio_chunks(audio_info.chunk_paths)

        # Average to get song-level embedding
        song_embedding = chunk_embeddings.mean(dim=0, keepdim=True)

        # Classify
        fields = classify_song(song_embedding, text_cache, is_instrumental)

        caption = fields.to_caption_string()
        details = fields.to_detail_dict()

        audio_dict = {
            "duration_seconds": round(audio_info.duration_seconds, 1),
            "num_chunks": audio_info.num_chunks,
            "sample_rate": audio_info.sample_rate,
        }

        return caption, details, audio_dict

    finally:
        cleanup_chunks(audio_info)


def caption_batch(stems: list, data_dir: str, cache_dir: str = None):
    """
    Caption all MP3 files in the data directory matching the given stems.
    Updates _batch_state in-place for polling from the API.

    Args:
        stems: List of stem names (without .mp3 extension).
        data_dir: Path to the server data directory containing MP3 files.
        cache_dir: Directory for text embedding cache persistence.
    """
    global _batch_state

    _batch_state["running"] = True
    _batch_state["total"] = len(stems)
    _batch_state["completed"] = 0
    _batch_state["current_file"] = ""
    _batch_state["results"] = {}
    _batch_state["error"] = None

    try:
        for stem in stems:
            mp3_path = os.path.join(data_dir, f"{stem}.mp3")
            if not os.path.exists(mp3_path):
                _batch_state["completed"] += 1
                continue

            _batch_state["current_file"] = stem

            # Read the existing prompt to check instrumental status
            prompt_path = os.path.join(data_dir, f"{stem}_prompt.txt")
            is_instrumental = False
            if os.path.exists(prompt_path):
                content = open(prompt_path, "r", encoding="utf-8").read().strip().lower()
                is_instrumental = "instrumental" in content

            try:
                caption, details, audio_dict = generate_caption(
                    mp3_path, is_instrumental=is_instrumental, cache_dir=cache_dir
                )
                _batch_state["results"][stem] = caption
            except Exception as e:
                _batch_state["results"][stem] = f"ERROR: {e}"

            _batch_state["completed"] += 1

    except Exception as e:
        _batch_state["error"] = str(e)
    finally:
        _batch_state["running"] = False
        _batch_state["current_file"] = ""
