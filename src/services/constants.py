"""
Shared constants for all service modules (rate limiting, retry behaviour, script guidance).
"""

# Rate limiting and retry configuration
PAGE_DELAY_SECONDS: float = 3.0  # Delay between pages to prevent content filter triggers
MAX_RETRIES: int = 10            # Maximum retries for content filter / transient errors
BASE_RETRY_DELAY: float = 3.0   # Base delay (seconds) for exponential backoff

# Per-language script guidance for OCR prompts (transcribe command)
OCR_SCRIPT_GUIDANCE: dict[str, str] = {
    'Chinese': (
        "The text uses Chinese characters (hanzi/漢字). "
        "Transcribe each character exactly as it appears."
    ),
    'Simplified Chinese': (
        "The text uses Simplified Chinese characters (简体字). "
        "Transcribe each character exactly in its simplified form — "
        "do NOT convert to or substitute traditional variants."
    ),
    'Traditional Chinese': (
        "The text uses Traditional Chinese characters (繁體字). "
        "Transcribe each character exactly in its traditional form — "
        "do NOT convert to or substitute simplified variants."
    ),
    'Japanese': (
        "The text uses Japanese script, which combines kanji (Chinese-derived characters), "
        "hiragana, katakana, and possibly rōmaji. "
        "Reproduce all scripts exactly as written. "
        "Some kanji may be Japanese-specific forms (kokuji) not found in standard Chinese — "
        "transcribe them faithfully and do NOT substitute simplified or traditional Chinese variants. "
        "Hiragana printed at very small sizes may be omitted only if completely illegible."
    ),
    'Korean': (
        "The text uses Korean script (hangul/한글), possibly mixed with hanja (漢字) or Latin text. "
        "Transcribe all scripts exactly as they appear."
    ),
    'English': (
        "The text uses the Latin alphabet."
    ),
}

# Per-language script guidance for combined OCR + translation prompts (translate command on images).
# Uses "source text" phrasing and includes translation-context notes (e.g. kanji disambiguation).
IMAGE_TRANSLATION_SCRIPT_GUIDANCE: dict[str, str] = {
    'Chinese': (
        "The source text uses Chinese characters (hanzi/漢字). "
        "Transcribe each character exactly as it appears."
    ),
    'Simplified Chinese': (
        "The source text uses Simplified Chinese characters (简体字). "
        "Transcribe each character exactly in its simplified form — "
        "do NOT convert to or substitute traditional variants."
    ),
    'Traditional Chinese': (
        "The source text uses Traditional Chinese characters (繁體字). "
        "Transcribe each character exactly in its traditional form — "
        "do NOT convert to or substitute simplified variants."
    ),
    'Japanese': (
        "The source text uses Japanese script, which combines kanji (Chinese-derived characters), "
        "hiragana, katakana, and possibly rōmaji. "
        "Reproduce all scripts exactly as written. "
        "Some kanji may be Japanese-specific forms (kokuji) not found in standard Chinese — "
        "transcribe them faithfully and do NOT substitute simplified or traditional Chinese variants. "
        "Use kanji ambiguity resolution via translation context before committing to a transcript."
    ),
    'Korean': (
        "The source text uses Korean script (hangul/한글), possibly mixed with hanja (漢字) or Latin text. "
        "Transcribe all scripts exactly as they appear."
    ),
    'English': (
        "The source text uses the Latin alphabet."
    ),
}
