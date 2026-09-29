"""Consolidates this dataset's 114 fine-grained genre tags into broader
families, for use as an alternate (looser) ground truth in `precision_at_k`.

Why this exists: several of the 114 tags are near-synonyms applied to the
*same physical track* (see `data.load_tracks`'s docstring — e.g. one song
tagged as `acoustic`, `j-pop`, `singer-songwriter`, and `songwriter`
simultaneously in the raw data). Exact-genre precision@k counts a
recommendation as wrong whenever it lands on a different one of those
near-synonyms, even though the two tracks may be genuinely
indistinguishable by sound. Family-level precision@k relaxes that,
without changing the recommender itself — only what counts as a "hit."

This mapping is a judgment call, not a ground truth — Spotify's own
genre taxonomy doesn't publish an official grouping, and a few
placements are genuinely debatable (`synth-pop` reads as much electronic
as pop; `ska`/`dancehall` sit at the reggae/dance boundary). Documented
inline where a choice was non-obvious.

One family, `mood_context`, is different in kind from the rest: `chill`,
`comedy`, `disney`, `happy`, `kids`/`children`, `party`, `romance`,
`sad`, `sleep`, and `study` aren't musical genres at all — they're
mood/activity playlist categories that could describe a track from *any*
actual genre. Lumping them into a musical family would be misleading, so
they get their own bucket instead. `media_soundtrack` (`anime`,
`disney`, `pop-film`, `show-tunes`) is similar: tied to where a track is
*used*, not what it sounds like.
"""

GENRE_TO_FAMILY: dict[str, str] = {
    # rock
    "alt-rock": "rock", "alternative": "rock", "emo": "rock", "garage": "rock",
    "goth": "rock", "grunge": "rock", "hard-rock": "rock", "indie": "rock",
    "j-rock": "rock", "psych-rock": "rock", "punk": "rock", "punk-rock": "rock",
    "rock": "rock", "rock-n-roll": "rock", "rockabilly": "rock",
    # metal
    "black-metal": "metal", "death-metal": "metal", "grindcore": "metal",
    "hardcore": "metal", "heavy-metal": "metal", "metal": "metal", "metalcore": "metal",
    # electronic / dance
    "ambient": "electronic_dance", "breakbeat": "electronic_dance",
    "chicago-house": "electronic_dance", "club": "electronic_dance",
    "dance": "electronic_dance", "deep-house": "electronic_dance",
    "detroit-techno": "electronic_dance", "disco": "electronic_dance",
    "drum-and-bass": "electronic_dance", "dubstep": "electronic_dance",
    "edm": "electronic_dance", "electro": "electronic_dance",
    "electronic": "electronic_dance", "hardstyle": "electronic_dance",
    "house": "electronic_dance", "idm": "electronic_dance",
    "industrial": "electronic_dance", "j-dance": "electronic_dance",
    "minimal-techno": "electronic_dance", "progressive-house": "electronic_dance",
    "techno": "electronic_dance", "trance": "electronic_dance", "trip-hop": "electronic_dance",
    # pop (synth-pop placed here, not electronic_dance — the name calls
    # out "pop" explicitly, a judgment call noted in the module docstring)
    "cantopop": "pop", "indie-pop": "pop", "j-idol": "pop", "j-pop": "pop",
    "k-pop": "pop", "mandopop": "pop", "pop": "pop", "power-pop": "pop", "synth-pop": "pop",
    # hip-hop / R&B
    "hip-hop": "hip_hop_rnb", "r-n-b": "hip_hop_rnb",
    # latin (reggaeton placed here rather than reggae_dancehall — Spotify's
    # own taxonomy treats it as a Latin genre despite the reggae roots)
    "brazil": "latin", "forro": "latin", "latin": "latin", "latino": "latin",
    "mpb": "latin", "pagode": "latin", "reggaeton": "latin", "salsa": "latin",
    "samba": "latin", "sertanejo": "latin", "spanish": "latin", "tango": "latin",
    # reggae / dancehall / ska
    "dancehall": "reggae_dancehall", "dub": "reggae_dancehall",
    "reggae": "reggae_dancehall", "ska": "reggae_dancehall",
    # world / regional
    "afrobeat": "world_regional", "british": "world_regional", "french": "world_regional",
    "german": "world_regional", "indian": "world_regional", "iranian": "world_regional",
    "malay": "world_regional", "swedish": "world_regional", "turkish": "world_regional",
    "world-music": "world_regional",
    # classical / acoustic / instrumental
    "acoustic": "classical_instrumental", "classical": "classical_instrumental",
    "guitar": "classical_instrumental", "new-age": "classical_instrumental",
    "opera": "classical_instrumental", "piano": "classical_instrumental",
    "singer-songwriter": "classical_instrumental", "songwriter": "classical_instrumental",
    # jazz / blues / soul
    "blues": "jazz_blues_soul", "funk": "jazz_blues_soul", "gospel": "jazz_blues_soul",
    "groove": "jazz_blues_soul", "jazz": "jazz_blues_soul", "soul": "jazz_blues_soul",
    # folk / country
    "bluegrass": "folk_country", "country": "folk_country", "folk": "folk_country",
    "honky-tonk": "folk_country",
    # media/soundtrack-tied tags — what a track is *used for*, not what it sounds like
    "anime": "media_soundtrack", "disney": "media_soundtrack",
    "pop-film": "media_soundtrack", "show-tunes": "media_soundtrack",
    # mood/activity playlist tags — not musical genres at all (see module docstring)
    "children": "mood_context", "chill": "mood_context", "comedy": "mood_context",
    "happy": "mood_context", "kids": "mood_context", "party": "mood_context",
    "romance": "mood_context", "sad": "mood_context", "sleep": "mood_context",
    "study": "mood_context",
}


def to_family(genre) -> object:
    """Maps one genre string (or a pandas Series of them) to its family.
    An unrecognized genre (shouldn't happen on this dataset, but a future
    Spotify API export could add new tags) maps to itself rather than
    raising, so an evaluation run degrades gracefully instead of crashing.
    """
    if hasattr(genre, "map"):
        return genre.map(lambda g: GENRE_TO_FAMILY.get(g, g))
    return GENRE_TO_FAMILY.get(genre, genre)
