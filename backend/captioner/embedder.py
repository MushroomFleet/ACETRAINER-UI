"""
ImageBind model wrapper for audio and text embedding extraction.
Lazy-loads the model on first use. Caches text embeddings to disk.
"""

import os
import torch
from typing import List, Optional


class ImageBindEmbedder:
    _instance: Optional["ImageBindEmbedder"] = None

    def __init__(self, device: str = "cuda:0", cache_dir: str = None):
        self.device = device
        self.model = None
        self._text_cache = {}
        self._cache_dir = cache_dir

    @classmethod
    def get_instance(cls, cache_dir: str = None) -> "ImageBindEmbedder":
        """Singleton — only one model instance across the app."""
        if cls._instance is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            cls._instance = cls(device=device, cache_dir=cache_dir)
        return cls._instance

    def _ensure_model(self):
        """Lazy-load ImageBind model on first use."""
        if self.model is not None:
            return
        from imagebind.models import imagebind_model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

    def embed_audio_chunks(self, chunk_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Embed audio chunks in batches.
        Returns tensor of shape (num_chunks, 1024).
        """
        self._ensure_model()
        from imagebind import data
        from imagebind.models.imagebind_model import ModalityType

        all_embeddings = []
        for i in range(0, len(chunk_paths), batch_size):
            batch_paths = chunk_paths[i:i + batch_size]
            audio_input = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(
                    batch_paths, self.device
                )
            }
            with torch.no_grad():
                result = self.model(audio_input)
            all_embeddings.append(result[ModalityType.AUDIO])

        return torch.cat(all_embeddings, dim=0)

    def embed_text_candidates(self, text_list: List[str]) -> torch.Tensor:
        """
        Embed text candidates. Returns tensor of shape (num_candidates, 1024).
        """
        self._ensure_model()
        from imagebind import data
        from imagebind.models.imagebind_model import ModalityType

        text_input = {
            ModalityType.TEXT: data.load_and_transform_text(text_list, self.device)
        }
        with torch.no_grad():
            result = self.model(text_input)
        return result[ModalityType.TEXT]

    def precompute_text_cache(self):
        """
        Precompute and cache all candidate text embeddings.
        Saves to disk as .pt file for instant reuse across restarts.
        Auto-invalidates stale caches when dimensions change.
        """
        from backend.captioner.candidates import (
            GENRE_CANDIDATES, VOCAL_CANDIDATES, INSTRUMENT_CANDIDATES,
            MOOD_CANDIDATES, TEMPO_CANDIDATES, KEY_CANDIDATES,
            TIMBRE_CANDIDATES, ERA_CANDIDATES, PRODUCTION_CANDIDATES,
            ENERGY_CANDIDATES,
        )

        candidate_map = {
            "genre": GENRE_CANDIDATES,
            "vocal": VOCAL_CANDIDATES,
            "instruments": INSTRUMENT_CANDIDATES,
            "mood": MOOD_CANDIDATES,
            "tempo": TEMPO_CANDIDATES,
            "key": KEY_CANDIDATES,
            "timbre": TIMBRE_CANDIDATES,
            "era": ERA_CANDIDATES,
            "production": PRODUCTION_CANDIDATES,
            "energy": ENERGY_CANDIDATES,
        }

        expected_dims = set(candidate_map.keys())

        cache_path = None
        if self._cache_dir:
            cache_path = os.path.join(self._cache_dir, "text_embeddings_cache.pt")
            if os.path.exists(cache_path):
                loaded = torch.load(
                    cache_path, map_location=self.device, weights_only=False
                )
                # Invalidate cache if dimensions changed
                if set(loaded.keys()) == expected_dims:
                    self._text_cache = loaded
                    return
                # Stale cache — will regenerate below

        for dim_name, candidates in candidate_map.items():
            self._text_cache[dim_name] = {
                "embeddings": self.embed_text_candidates(candidates),
                "labels": candidates,
            }

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self._text_cache, cache_path)

    @staticmethod
    def _candidate_fingerprint() -> int:
        """Hash of total candidate count across all dimensions for staleness detection."""
        from backend.captioner.candidates import (
            GENRE_CANDIDATES, VOCAL_CANDIDATES, INSTRUMENT_CANDIDATES,
            MOOD_CANDIDATES, TEMPO_CANDIDATES, KEY_CANDIDATES,
            TIMBRE_CANDIDATES, ERA_CANDIDATES, PRODUCTION_CANDIDATES,
            ENERGY_CANDIDATES,
        )
        return sum(len(c) for c in [
            GENRE_CANDIDATES, VOCAL_CANDIDATES, INSTRUMENT_CANDIDATES,
            MOOD_CANDIDATES, TEMPO_CANDIDATES, KEY_CANDIDATES,
            TIMBRE_CANDIDATES, ERA_CANDIDATES, PRODUCTION_CANDIDATES,
            ENERGY_CANDIDATES,
        ])

    def get_text_cache(self) -> dict:
        expected = self._candidate_fingerprint()
        if not self._text_cache or getattr(self, '_cache_fingerprint', None) != expected:
            self.precompute_text_cache()
            self._cache_fingerprint = expected
        return self._text_cache
