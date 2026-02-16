"""
MP3 audio preprocessing: convert to 16kHz mono WAV, chunk into 2-second segments.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import List

from pydub import AudioSegment


@dataclass
class AudioInfo:
    filepath: str
    duration_seconds: float
    sample_rate: int
    num_chunks: int
    chunk_paths: List[str] = field(default_factory=list)


def preprocess_mp3(mp3_path: str, output_dir: str = None) -> AudioInfo:
    """
    Load MP3, convert to 16kHz mono WAV, segment into 2-second chunks.
    For songs > 60s, sample chunks every 15 seconds instead of using all chunks.

    Args:
        mp3_path: Path to the MP3 file on disk.
        output_dir: Directory for temporary chunk WAVs. Auto-created if None.

    Returns:
        AudioInfo with chunk paths and metadata.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="captioner_")
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    duration_seconds = len(audio) / 1000.0
    chunk_duration_ms = 2000

    # For long songs, sample every 15 seconds instead of every 2 seconds
    if duration_seconds > 60:
        sample_interval_ms = 15000
        start_positions = list(range(0, len(audio), sample_interval_ms))
    else:
        start_positions = list(range(0, len(audio), chunk_duration_ms))

    chunk_paths = []
    for idx, start_ms in enumerate(start_positions):
        chunk = audio[start_ms:start_ms + chunk_duration_ms]

        # Pad if shorter than 2 seconds
        if len(chunk) < chunk_duration_ms:
            silence = AudioSegment.silent(
                duration=chunk_duration_ms - len(chunk),
                frame_rate=16000,
            )
            chunk = chunk + silence

        chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    return AudioInfo(
        filepath=mp3_path,
        duration_seconds=duration_seconds,
        sample_rate=16000,
        num_chunks=len(chunk_paths),
        chunk_paths=chunk_paths,
    )


def cleanup_chunks(audio_info: AudioInfo):
    """Remove temporary chunk WAV files."""
    for path in audio_info.chunk_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    # Try to remove the temp directory too
    if audio_info.chunk_paths:
        parent = os.path.dirname(audio_info.chunk_paths[0])
        try:
            os.rmdir(parent)
        except OSError:
            pass
