import hashlib

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def assert_prompt_unchanged(name: str, text: str, expected_hash: str):
    actual = hash_text(text)
    if actual != expected_hash:
        raise RuntimeError(
            f"Prompt drift detected: {name}\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual}"
        )
