"""
tts.py
------
Text-to-speech using gTTS (Google Text-to-Speech).
Free, no API key needed, supports all 6 target Indian languages.

Limitations of gTTS:
- Requires internet connection (calls Google TTS API)
- No offline mode
- Limited voice customization
For offline fallback, pyttsx3 can be used but Indian language support is poor.
"""

import os
import tempfile
from gtts import gTTS

# gTTS language codes for supported Indian languages
GTTS_LANG_MAP = {
    "hi": "hi",   # Hindi
    "bn": "bn",   # Bengali
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "ml": "ml",   # Malayalam
    "kn": "kn",   # Kannada
    "en": "en",   # English
}

# Fallback to English if language not supported
DEFAULT_LANG = "en"


def synthesize(text: str, language_code: str = "en", output_path: str = None) -> str:
    """
    Convert text to speech and save as MP3.

    Parameters
    ----------
    text          : str   Text to convert (in the target language)
    language_code : str   Language code e.g. 'hi', 'ta', 'en'
    output_path   : str   Optional path to save the MP3.
                          If None, saves to a temp file.

    Returns
    -------
    str   Path to the generated MP3 file, or None if failed.
    """
    if not text or not text.strip():
        return None

    lang = GTTS_LANG_MAP.get(language_code, DEFAULT_LANG)

    # Truncate very long texts to avoid slow generation
    # gTTS can handle long text but it's slow; 1000 chars is ~1 min of speech
    if len(text) > 1500:
        text = text[:1500] + "..."

    try:
        tts = gTTS(text=text, lang=lang, slow=False)

        if output_path is None:
            # Create a temp file that persists until explicitly deleted
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp3", prefix="tts_"
            )
            output_path = tmp.name
            tmp.close()

        tts.save(output_path)
        print(f"[tts] Audio saved to: {output_path} (lang={lang})")
        return output_path

    except Exception as e:
        print(f"[tts] gTTS error: {e}")
        return None


def cleanup(audio_path: str):
    """Delete a temp audio file after it's been served."""
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception:
            pass
