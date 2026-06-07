import hashlib


def render_text_hash(text: str, digits: int | None = 12) -> str:
    """Generate a SHA-256 hash for the given text.

    This is the single hashing entry point for the entire codebase.
    All modules that need to derive a hash from text should use this function
    instead of calling ``hashlib`` directly.

    Args:
        text: The text to hash.
        digits: Number of hex digits to return (default: 12).
            Pass ``None`` to return the full 64-character hex digest.

    Returns:
        A hex string hash of the text.
    """
    digest = hashlib.sha256(text.encode()).hexdigest()
    return digest[:digits] if digits is not None else digest
