GENRE_ALIASES = {
    "english business news": "EBN",
    "english business": "EBN",
    "ebn": "EBN",

    "hindi business news": "HBN v1",
    "hindi business": "HBN v1",
    "hbn": "HBN v1",

    "english news": "English News",
    "hindi news": "HSM",
    "hsm": "HSM",
}

def resolve_genre(text: str | None):
    if not text:
        return None

    t = text.lower()
    for key, genre in GENRE_ALIASES.items():
        if key in t:
            return genre

    return None
