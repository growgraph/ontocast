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


def render_bytes_hash(payload: bytes, digits: int | None = 12) -> str:
    """Generate a SHA-256 hash for raw bytes.

    Binary inputs (PDFs, office documents) must be hashed directly rather than
    decoded to text first: a lossy ``decode(errors="ignore")`` discards most of
    a binary payload before hashing, so the resulting key rests on whichever
    bytes happen to form valid UTF-8.

    Args:
        payload: The bytes to hash.
        digits: Number of hex digits to return (default: 12).
            Pass ``None`` to return the full 64-character hex digest.

    Returns:
        A hex string hash of the bytes.
    """
    digest = hashlib.sha256(payload).hexdigest()
    return digest[:digits] if digits is not None else digest
