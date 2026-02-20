"""
Text candidate libraries for each caption dimension.
ImageBind embeds these strings and compares against audio embeddings.
"""

GENRE_CANDIDATES = [
    "rock music", "pop music", "hip hop music", "electronic dance music",
    "jazz music", "classical music", "country music", "R&B music",
    "soul music", "blues music", "reggae music", "metal music",
    "punk rock music", "folk music", "latin music", "world music",
    "ambient music", "house music", "techno music", "drum and bass music",
    "trap music", "lo-fi music", "synthwave music", "phonk music",
    "shoegaze music", "grunge music", "disco music", "funk music",
    "gospel music", "new wave music", "indie rock music", "alternative rock music",
    "progressive rock music", "psychedelic rock music", "hard rock music",
    "afrobeat music", "bossa nova music", "k-pop music", "j-pop music",
    "city pop music", "trot music", "enka music", "chanson music",
    "schlager music", "dancehall music", "dub music", "garage music",
    "grime music", "drill music", "dubstep music", "trance music",
    "hardstyle music", "industrial music", "noise music", "vaporwave music",
]

VOCAL_CANDIDATES = [
    "male vocal", "female vocal", "mixed vocals male and female",
    "vocal group choir", "rapping vocal", "spoken word vocal",
    "auto-tuned vocal", "whispered vocal", "falsetto vocal",
    "operatic vocal", "screaming vocal", "humming vocal",
    # --- new from aceset classifiers ---
    "breathy vocal", "raspy vocal", "powerful belting vocal",
    "harmonies vocal", "call and response vocal", "ad-lib vocal",
]

INSTRUMENT_CANDIDATES = [
    "electric guitar", "acoustic guitar", "bass guitar", "piano",
    "synthesizer", "drum kit", "drum machine", "808 bass",
    "strings orchestra", "brass section", "woodwind section",
    "violin", "cello", "flute", "saxophone", "trumpet",
    "organ", "harpsichord", "banjo", "mandolin", "ukulele",
    "sitar", "tabla", "harp", "accordion", "harmonica",
    "turntable scratching", "sampler", "sequencer",
    "steel drums", "marimba", "vibraphone", "erhu",
    "guzheng", "koto", "gayageum", "didgeridoo",
    "bagpipes", "electric bass", "upright bass", "pedal steel guitar",
]

MOOD_CANDIDATES = [
    "energetic", "melancholic", "uplifting", "dark", "dreamy",
    "aggressive", "peaceful", "romantic", "nostalgic", "euphoric",
    "mysterious", "playful", "tense", "triumphant", "somber",
    "ethereal", "gritty", "warm", "cold", "haunting",
    "joyful", "angry", "contemplative", "hypnotic", "chaotic",
    "serene", "brooding", "whimsical", "intense", "laid-back",
    "moody", "atmospheric", "cinematic", "raw", "polished",
]

TEMPO_CANDIDATES = [
    "very slow tempo around 60 BPM", "slow tempo around 70 BPM",
    "slow moderate tempo around 80 BPM", "moderate tempo around 90 BPM",
    "moderate tempo around 100 BPM", "moderate fast tempo around 110 BPM",
    "fast tempo around 120 BPM", "fast tempo around 130 BPM",
    "very fast tempo around 140 BPM", "very fast tempo around 150 BPM",
    "extremely fast tempo around 160 BPM", "extremely fast tempo around 170 BPM",
    "breakneck tempo around 180 BPM",
]

KEY_CANDIDATES = [
    "music in the key of C major", "music in the key of C minor",
    "music in the key of D major", "music in the key of D minor",
    "music in the key of E major", "music in the key of E minor",
    "music in the key of F major", "music in the key of F minor",
    "music in the key of G major", "music in the key of G minor",
    "music in the key of A major", "music in the key of A minor",
    "music in the key of B major", "music in the key of B minor",
    "music in the key of C sharp major", "music in the key of C sharp minor",
    "music in the key of D sharp minor", "music in the key of E flat major",
    "music in the key of F sharp major", "music in the key of F sharp minor",
    "music in the key of G sharp minor", "music in the key of A flat major",
    "music in the key of B flat major", "music in the key of B flat minor",
]

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
