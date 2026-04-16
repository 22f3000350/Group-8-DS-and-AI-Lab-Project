"""
asr.py
------
Speech-to-text using OpenAI Whisper.
Accepts an audio file path (from Gradio microphone input)
and returns the transcribed text + detected language.

Model: whisper-small  (good balance for Indian languages on HF free tier)
       Switch to whisper-medium if accuracy needs improvement,
       but it requires more RAM.
"""

import os
import whisper

_model = None

def _get_model(model_size: str = "small"):
    """Lazy-load Whisper model (singleton to avoid reloading on every call)."""
    global _model
    if _model is None:
        print(f"[asr] Loading Whisper {model_size} model...")
        _model = whisper.load_model(model_size)
        print("[asr] Whisper model loaded.")
    return _model


def _check_audio_validity(audio_path: str) -> tuple[bool, str]:
    """Check if audio file is valid and has content."""
    if not os.path.exists(audio_path):
        return False, "File does not exist"
    
    file_size = os.path.getsize(audio_path)
    if file_size < 10000:  # Less than 10KB is likely empty/silence
        return False, f"Audio file too small ({file_size} bytes) - likely silence or not recorded"
    
    return True, f"Audio file valid ({file_size} bytes)"


def transcribe(audio_path: str, model_size: str = "small") -> dict:
    """
    Transcribe audio to text and detect language.

    Parameters
    ----------
    audio_path : str   Path to audio file (wav/mp3/m4a — Gradio saves as wav)
    model_size : str   'tiny', 'small', 'medium' — use 'small' for HF free tier

    Returns
    -------
    dict with keys:
        text           : str   Transcribed text
        language       : str   Detected language name e.g. 'hindi', 'tamil'
        language_code  : str   ISO code e.g. 'hi', 'ta'
        success        : bool
    """
    if not audio_path or not os.path.exists(audio_path):
        return {"text": "", "language": "english", "language_code": "en", "success": False}

    # Validate audio file
    is_valid, msg = _check_audio_validity(audio_path)
    print(f"[asr] Audio validation: {msg}")
    if not is_valid:
        return {"text": "", "language": "english", "language_code": "en", "success": False}

    try:
        model = _get_model(model_size)

        # Whisper auto-detects language when language=None
        result = model.transcribe(audio_path, language=None, task="transcribe")

        text      = result.get("text", "").strip()
        lang_name = result.get("language", "english").lower()

        # Map Whisper language names to our codes
        # Whisper returns full language names like 'hindi', 'tamil', etc.
        lang_code_map = {
            "hindi":     "hi",
            "bengali":   "bn",
            "tamil":     "ta",
            "telugu":    "te",
            "malayalam": "ml",
            "kannada":   "kn",
            "english":   "en",
            # Whisper sometimes returns these
            "marathi":   "mr",
            "gujarati":  "gu",
            "punjabi":   "pa",
        }
        lang_code = lang_code_map.get(lang_name, "en")

        print(f"[asr] Transcribed: '{text[:80]}...' | Language: {lang_name} ({lang_code})")

        return {
            "text":          text,
            "language":      lang_name,
            "language_code": lang_code,
            "success":       bool(text),
        }

    except Exception as e:
        print(f"[asr] Transcription error: {e}")
        return {"text": "", "language": "english", "language_code": "en", "success": False}
