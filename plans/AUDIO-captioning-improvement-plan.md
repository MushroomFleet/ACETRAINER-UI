# AUDIO Captioning Improvement Plan

## Assessment Summary

### Current Pipeline Architecture

```
MP3 file
  -> audio_preprocessor.py   (16kHz mono WAV, 2s chunks, 15s sampling for >60s)
  -> embedder.py              (ImageBind huge, singleton, text cache on disk)
  -> classifier.py            (cosine similarity + softmax, per-dimension top-k)
  -> service.py               (orchestrator, batch support)
  -> captioner_api.py         (Flask blueprint at /api/captioner/*)
```

**Current 6 dimensions** in `candidates.py`:

| Dimension    | Candidates | top_k | Caption role                  |
|-------------|-----------|-------|-------------------------------|
| genre        | 56         | 1     | primary style tag             |
| vocal        | 12         | 1     | vocal type or "instrumental"  |
| instruments  | 41         | 3     | up to 3 detected instruments  |
| mood         | 35         | 2     | up to 2 mood descriptors      |
| tempo        | 13         | 1     | BPM estimate                  |
| key          | 22         | 1     | musical key                   |

**Output format**: `"genre, vocal, instrument1 and instrument2 and instrument3, mood1 and mood2, BPM, key"`

### Gap Analysis vs `acestep-classifiers.txt`

The aceset source document defines **10 caption dimensions** for ACE-Step prompts. Our captioner covers 6. The missing 4 are:

| Missing Dimension   | aceset Examples                                              | Classifiable via ImageBind? |
|---------------------|--------------------------------------------------------------|-----------------------------|
| **Timbre/Texture**  | warm, bright, crisp, muddy, airy, punchy, lush, raw, polished | Yes — tonal quality in audio |
| **Era Reference**   | 80s synth-pop, 90s grunge, 2010s EDM, vintage soul, modern trap | Yes — era-specific sonic signatures |
| **Production Style**| lo-fi, high-fidelity, live recording, studio-polished, bedroom pop | Yes — recording artifacts in audio |
| **Structure Hints** | building intro, catchy chorus, dramatic bridge, fade-out ending | Partially — requires temporal analysis |

Additionally, existing dimensions have incomplete candidate coverage:
- **Vocal**: missing `breathy`, `raspy`, `powerful belting`, `harmonies` (present in aceset vocal tags)
- **Mood/Energy**: aceset separates energy level (`high energy`, `low energy`, `building energy`, `explosive`) from mood — our mood list partially covers this but not explicitly

### Files That Require Changes

| File | Changes |
|------|---------|
| `backend/captioner/candidates.py` | Add 4 new candidate lists, expand vocal candidates |
| `backend/captioner/classifier.py` | Add 4 new fields to `CaptionFields`, extend `classify_song()`, update `to_caption_string()` and `to_detail_dict()` |
| `backend/captioner/embedder.py` | Import and embed new candidate lists in `precompute_text_cache()` |
| `backend/captioner/service.py` | No structural changes needed (auto-picks up new fields) |
| `workspace/captioner_cache/` | Delete stale `text_embeddings_cache.pt` (auto-regenerates) |

---

## Implementation Steps

### Step 1 — Add New Candidate Lists to `candidates.py`

Add these four new lists after the existing `KEY_CANDIDATES`:

```python
TIMBRE_CANDIDATES = [
    "warm sound", "bright sound", "crisp sound", "muddy sound",
    "airy sound", "punchy sound", "lush sound", "raw sound",
    "polished sound", "thin sound", "thick sound", "harsh sound",
    "smooth sound", "gritty sound", "clean sound", "distorted sound",
    "hollow sound", "full sound", "dry sound", "reverberant sound",
    "saturated sound", "compressed sound", "lo-fi sound", "hi-fi sound",
]

ERA_CANDIDATES = [
    "1950s rock and roll era", "1960s psychedelic era", "1970s disco funk era",
    "1980s synth-pop new wave era", "1990s grunge alternative era",
    "2000s pop R&B era", "2010s EDM trap era", "2020s hyperpop modern era",
    "vintage retro era", "classic era", "modern contemporary era",
    "futuristic era",
]

PRODUCTION_CANDIDATES = [
    "lo-fi production", "high-fidelity production", "live recording production",
    "studio-polished production", "bedroom pop production", "overdriven production",
    "minimalist production", "maximalist production", "heavily layered production",
    "sparse production", "analog production", "digital production",
    "sample-based production", "acoustic production",
]

ENERGY_CANDIDATES = [
    "very low energy", "low energy", "moderate energy", "high energy",
    "very high energy", "explosive energy", "building energy",
    "declining energy", "steady energy", "fluctuating energy",
]
```

Also expand `VOCAL_CANDIDATES` with the missing aceset vocal tags:

```python
VOCAL_CANDIDATES = [
    "male vocal", "female vocal", "mixed vocals male and female",
    "vocal group choir", "rapping vocal", "spoken word vocal",
    "auto-tuned vocal", "whispered vocal", "falsetto vocal",
    "operatic vocal", "screaming vocal", "humming vocal",
    # --- new from aceset classifiers ---
    "breathy vocal", "raspy vocal", "powerful belting vocal",
    "harmonies vocal", "call and response vocal", "ad-lib vocal",
]
```

### Step 2 — Extend `CaptionFields` in `classifier.py`

Update the dataclass to include the 4 new dimensions:

```python
@dataclass
class CaptionFields:
    genre: ClassificationResult
    vocal: ClassificationResult
    instruments: List[ClassificationResult]
    mood: List[ClassificationResult]
    tempo: ClassificationResult
    key: ClassificationResult
    timbre: ClassificationResult          # NEW
    era: ClassificationResult             # NEW
    production: ClassificationResult      # NEW
    energy: ClassificationResult          # NEW
    is_instrumental: bool
```

### Step 3 — Update `to_caption_string()` in `classifier.py`

The caption string gains 4 new comma-separated fields. The order follows aceset's dimension priority (genre first, structural descriptors last):

```python
def to_caption_string(self) -> str:
    genre_str = self.genre.label.replace(" music", "")

    if self.is_instrumental:
        vocal_str = "instrumental"
    else:
        vocal_str = self.vocal.label

    instruments_str = " and ".join([i.label for i in self.instruments])
    mood_str = " and ".join([m.label for m in self.mood])

    tempo_parts = self.tempo.label.split("around ")
    tempo_str = tempo_parts[1] if len(tempo_parts) > 1 else self.tempo.label

    key_str = self.key.label.replace("music in the key of ", "")

    # New dimensions — strip trailing descriptor words for cleaner tags
    timbre_str = self.timbre.label.replace(" sound", "")
    era_str = self.era.label.replace(" era", "")
    production_str = self.production.label.replace(" production", "")
    energy_str = self.energy.label.replace(" energy", "")

    return (
        f"{genre_str}, {vocal_str}, {instruments_str}, {mood_str}, "
        f"{tempo_str}, {key_str}, {timbre_str}, {era_str}, "
        f"{production_str}, {energy_str}"
    )
```

### Step 4 — Update `to_detail_dict()` in `classifier.py`

Add the new dimension blocks to the returned dictionary:

```python
def to_detail_dict(self) -> dict:
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
        # --- new dimensions ---
        "timbre": {
            "label": self.timbre.label.replace(" sound", ""),
            "confidence": round(self.timbre.confidence, 3),
        },
        "era": {
            "label": self.era.label.replace(" era", ""),
            "confidence": round(self.era.confidence, 3),
        },
        "production": {
            "label": self.production.label.replace(" production", ""),
            "confidence": round(self.production.confidence, 3),
        },
        "energy": {
            "label": self.energy.label.replace(" energy", ""),
            "confidence": round(self.energy.confidence, 3),
        },
    }
```

### Step 5 — Extend `classify_song()` in `classifier.py`

Add 4 new `classify_dimension()` calls and pass them to the updated `CaptionFields`:

```python
def classify_song(
    audio_embedding: torch.Tensor,
    text_cache: dict,
    is_instrumental: bool = False,
) -> CaptionFields:
    """Run classification across all 10 dimensions."""
    genre = classify_dimension(
        audio_embedding, text_cache["genre"]["embeddings"],
        text_cache["genre"]["labels"], top_k=1,
    )
    vocal = classify_dimension(
        audio_embedding, text_cache["vocal"]["embeddings"],
        text_cache["vocal"]["labels"], top_k=1,
    )
    instruments = classify_dimension(
        audio_embedding, text_cache["instruments"]["embeddings"],
        text_cache["instruments"]["labels"], top_k=3,
    )
    mood = classify_dimension(
        audio_embedding, text_cache["mood"]["embeddings"],
        text_cache["mood"]["labels"], top_k=2,
    )
    tempo = classify_dimension(
        audio_embedding, text_cache["tempo"]["embeddings"],
        text_cache["tempo"]["labels"], top_k=1,
    )
    key = classify_dimension(
        audio_embedding, text_cache["key"]["embeddings"],
        text_cache["key"]["labels"], top_k=1,
    )
    # --- new dimensions ---
    timbre = classify_dimension(
        audio_embedding, text_cache["timbre"]["embeddings"],
        text_cache["timbre"]["labels"], top_k=1,
    )
    era = classify_dimension(
        audio_embedding, text_cache["era"]["embeddings"],
        text_cache["era"]["labels"], top_k=1,
    )
    production = classify_dimension(
        audio_embedding, text_cache["production"]["embeddings"],
        text_cache["production"]["labels"], top_k=1,
    )
    energy = classify_dimension(
        audio_embedding, text_cache["energy"]["embeddings"],
        text_cache["energy"]["labels"], top_k=1,
    )

    return CaptionFields(
        genre=genre[0],
        vocal=vocal[0],
        instruments=instruments,
        mood=mood,
        tempo=tempo[0],
        key=key[0],
        timbre=timbre[0],
        era=era[0],
        production=production[0],
        energy=energy[0],
        is_instrumental=is_instrumental,
    )
```

### Step 6 — Register New Candidates in `embedder.py`

Update the import and `candidate_map` in `precompute_text_cache()`:

```python
def precompute_text_cache(self):
    from backend.captioner.candidates import (
        GENRE_CANDIDATES, VOCAL_CANDIDATES, INSTRUMENT_CANDIDATES,
        MOOD_CANDIDATES, TEMPO_CANDIDATES, KEY_CANDIDATES,
        TIMBRE_CANDIDATES, ERA_CANDIDATES, PRODUCTION_CANDIDATES,
        ENERGY_CANDIDATES,
    )

    cache_path = None
    if self._cache_dir:
        cache_path = os.path.join(self._cache_dir, "text_embeddings_cache.pt")
        if os.path.exists(cache_path):
            loaded = torch.load(cache_path, map_location=self.device, weights_only=False)
            # Invalidate cache if dimensions changed
            expected_dims = {"genre", "vocal", "instruments", "mood", "tempo", "key",
                             "timbre", "era", "production", "energy"}
            if set(loaded.keys()) == expected_dims:
                self._text_cache = loaded
                return
            # Stale cache — will regenerate below

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

    for dim_name, candidates in candidate_map.items():
        self._text_cache[dim_name] = {
            "embeddings": self.embed_text_candidates(candidates),
            "labels": candidates,
        }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(self._text_cache, cache_path)
```

### Step 7 — Invalidate Stale Text Embedding Cache

The updated `precompute_text_cache()` above handles this automatically by checking `set(loaded.keys()) == expected_dims`. On first run after the update, it will detect the old 6-dimension cache is missing the new keys and regenerate the full 10-dimension cache.

No manual deletion needed, but if you want to force it:

```bash
del workspace\captioner_cache\text_embeddings_cache.pt
```

### Step 8 — Update Frontend Caption Details Display (Optional)

The caption details panel in `dataset-editor.js` renders whatever keys the API returns. Since `to_detail_dict()` now includes 4 new keys, they will appear automatically if the frontend iterates over the dict. Verify the details rendering code handles unknown keys gracefully — if it uses a hardcoded list of fields, add the new ones.

Check this section in `static/js/dataset-editor.js` around the caption details display logic. The detail dict is rendered as confidence bars. If the renderer iterates all keys dynamically, no changes needed. If it hardcodes the 6 original field names, add:

```javascript
// In the caption details rendering section, ensure these are included:
const dimensionOrder = [
    'genre', 'vocal', 'instruments', 'mood', 'tempo', 'key',
    'timbre', 'era', 'production', 'energy'  // new
];
```

---

## Caption Output Format — Before vs After

**Before** (6 dimensions):
```
pop, female vocal, piano and synthesizer and strings orchestra, dreamy and nostalgic, 90 BPM, C minor
```

**After** (10 dimensions):
```
pop, female vocal, piano and synthesizer and strings orchestra, dreamy and nostalgic, 90 BPM, C minor, warm, 2010s EDM trap, studio-polished, moderate
```

This matches the full tag vocabulary ACE-Step expects in its prompt conditioning, giving the model richer audio-text alignment during LoRA training.

---

## Dimensions NOT Added (And Why)

| Dimension from aceset | Reason for exclusion |
|----------------------|---------------------|
| **Structure Hints** (building intro, catchy chorus, etc.) | These describe temporal song structure. A single averaged embedding cannot classify verse vs chorus. Would require segment-level temporal classification — a much larger architectural change. Deferred to a future phase. |
| **Lyric structure tags** ([Verse], [Chorus], [Bridge]) | These belong in the lyrics text, not the audio caption prompt. The captioner only generates the `_prompt.txt` tag string. |
| **Vocal style inline tags** ([raspy vocal], [whispered]) | These are lyric-inline annotations, not prompt-level tags. Handled by the lyrics editor, not the captioner. |
| **Energy change tags** ([building energy], [explosive drop]) | These are lyric-inline dynamic markers. Different from the prompt-level energy dimension we added. |

---

## Testing Checklist

- [ ] Delete or let auto-invalidate `workspace/captioner_cache/text_embeddings_cache.pt`
- [ ] Run single-file caption via UI — verify 10-field comma-separated output
- [ ] Verify detail panel shows all 10 dimensions with confidence scores
- [ ] Run batch caption-all — verify new format applied to all samples
- [ ] Confirm `convert_to_hf_dataset` correctly splits the longer tag string into the `tags` list (it uses `prompt.split(", ")` which works for any number of comma-separated values)
- [ ] Spot-check a few songs: do timbre/era/production/energy labels sound reasonable?
