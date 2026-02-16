"""
Zero-shot audio classification via cosine similarity in ImageBind's shared embedding space.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


@dataclass
class ClassificationResult:
    label: str
    confidence: float


@dataclass
class CaptionFields:
    genre: ClassificationResult
    vocal: ClassificationResult
    instruments: List[ClassificationResult]
    mood: List[ClassificationResult]
    tempo: ClassificationResult
    key: ClassificationResult
    is_instrumental: bool

    def to_caption_string(self) -> str:
        """Assemble the final comma-separated caption string."""
        genre_str = self.genre.label.replace(" music", "")

        if self.is_instrumental:
            vocal_str = "instrumental"
        else:
            vocal_str = self.vocal.label

        instruments_str = " and ".join([i.label for i in self.instruments])
        mood_str = " and ".join([m.label for m in self.mood])

        # "fast tempo around 120 BPM" -> "120 BPM"
        tempo_parts = self.tempo.label.split("around ")
        tempo_str = tempo_parts[1] if len(tempo_parts) > 1 else self.tempo.label

        # "music in the key of A minor" -> "A minor"
        key_str = self.key.label.replace("music in the key of ", "")

        return f"{genre_str}, {vocal_str}, {instruments_str}, {mood_str}, {tempo_str}, {key_str}"

    def to_detail_dict(self) -> dict:
        """Return a JSON-serializable detail breakdown with confidence scores."""
        return {
            "genre": {
                "label": self.genre.label.replace(" music", ""),
                "confidence": round(self.genre.confidence, 3),
            },
            "vocal": {
                "label": "instrumental" if self.is_instrumental else self.vocal.label,
                "confidence": 1.0 if self.is_instrumental else round(self.vocal.confidence, 3),
            },
            "instruments": [
                {"label": i.label, "confidence": round(i.confidence, 3)}
                for i in self.instruments
            ],
            "mood": [
                {"label": m.label, "confidence": round(m.confidence, 3)}
                for m in self.mood
            ],
            "tempo": {
                "label": (
                    self.tempo.label.split("around ")[-1]
                    if "around" in self.tempo.label
                    else self.tempo.label
                ),
                "confidence": round(self.tempo.confidence, 3),
            },
            "key": {
                "label": self.key.label.replace("music in the key of ", ""),
                "confidence": round(self.key.confidence, 3),
            },
        }


def classify_dimension(
    audio_embedding: torch.Tensor,
    text_embeddings: torch.Tensor,
    candidate_labels: list,
    top_k: int = 1,
    temperature: float = 100.0,
) -> List[ClassificationResult]:
    """
    Cosine similarity classification. Returns top_k results.
    """
    audio_norm = F.normalize(audio_embedding, dim=-1)
    text_norm = F.normalize(text_embeddings, dim=-1)
    similarities = (audio_norm @ text_norm.T).squeeze(0)
    probs = torch.softmax(similarities * temperature, dim=-1)

    top_values, top_indices = probs.topk(min(top_k, len(candidate_labels)))

    results = []
    for val, idx in zip(top_values, top_indices):
        results.append(ClassificationResult(
            label=candidate_labels[idx.item()],
            confidence=val.item(),
        ))
    return results


def classify_song(
    audio_embedding: torch.Tensor,
    text_cache: dict,
    is_instrumental: bool = False,
) -> CaptionFields:
    """Run classification across all 6 dimensions."""
    genre = classify_dimension(
        audio_embedding,
        text_cache["genre"]["embeddings"],
        text_cache["genre"]["labels"],
        top_k=1,
    )
    vocal = classify_dimension(
        audio_embedding,
        text_cache["vocal"]["embeddings"],
        text_cache["vocal"]["labels"],
        top_k=1,
    )
    instruments = classify_dimension(
        audio_embedding,
        text_cache["instruments"]["embeddings"],
        text_cache["instruments"]["labels"],
        top_k=3,
    )
    mood = classify_dimension(
        audio_embedding,
        text_cache["mood"]["embeddings"],
        text_cache["mood"]["labels"],
        top_k=2,
    )
    tempo = classify_dimension(
        audio_embedding,
        text_cache["tempo"]["embeddings"],
        text_cache["tempo"]["labels"],
        top_k=1,
    )
    key = classify_dimension(
        audio_embedding,
        text_cache["key"]["embeddings"],
        text_cache["key"]["labels"],
        top_k=1,
    )

    return CaptionFields(
        genre=genre[0],
        vocal=vocal[0],
        instruments=instruments,
        mood=mood,
        tempo=tempo[0],
        key=key[0],
        is_instrumental=is_instrumental,
    )
