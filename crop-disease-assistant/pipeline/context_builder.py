"""
context_builder.py
------------------
Takes the model's prediction dict and the user's input,
and constructs a structured query object for the RAG retriever.
"""

# Maps detected language names (from Whisper) to language codes
# used throughout the pipeline.
LANGUAGE_MAP = {
    # Whisper detected names
    "hindi":     "hi",
    "bengali":   "bn",
    "tamil":     "ta",
    "telugu":    "te",
    "malayalam": "ml",
    "kannada":   "kn",
    "english":   "en",
    # gTTS / display names (same mapping, for UI dropdown)
    "Hindi":     "hi",
    "Bengali":   "bn",
    "Tamil":     "ta",
    "Telugu":    "te",
    "Malayalam": "ml",
    "Kannada":   "kn",
    "English":   "en",
}

LANGUAGE_DISPLAY = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "en": "English",
}


def build_context(
    prediction: dict,
    user_query: str = "",
    user_language: str = "en",
) -> dict:
    """
    Build a structured context object from the model prediction.

    Parameters
    ----------
    prediction   : dict   Output from inference.predict()
    user_query   : str    User's text/voice question (already transcribed)
    user_language: str    Language code e.g. 'ta', 'hi', 'en'
                          or full name e.g. 'Tamil' — both handled

    Returns
    -------
    dict with keys:
        crop           : str
        disease        : str
        confidence     : float
        low_confidence : bool
        user_query     : str
        language_code  : str   e.g. 'ta'
        language_name  : str   e.g. 'Tamil'
        retrieval_query: str   query string for ChromaDB
        llm_context    : str   summary string for LLM prompt header
    """
    # Normalise language input
    lang_code = LANGUAGE_MAP.get(user_language, user_language[:2].lower())
    lang_name = LANGUAGE_DISPLAY.get(lang_code, "English")

    crop    = prediction["crop"].lower()
    disease = prediction["disease"].lower()
    conf    = prediction["confidence"]
    low_conf = prediction["low_confidence"]

    # Build the retrieval query — combines disease + crop + user intent
    if user_query.strip():
        retrieval_query = (
            f"{disease} in {crop}: symptoms treatment prevention. "
            f"Farmer query: {user_query}"
        )
    else:
        retrieval_query = (
            f"symptoms treatment prevention management of {disease} in {crop} crop"
        )

    # Summary line passed to LLM as context header
    if low_conf:
        llm_context = (
            f"The image analysis was inconclusive (confidence: {conf:.0%}). "
            f"Possible disease: {disease} in {crop}. "
            f"Advise the farmer to consult a local agricultural officer or KVK."
        )
    else:
        llm_context = (
            f"Disease detected: {disease} in {crop} crop (confidence: {conf:.0%})."
        )

    return {
        "crop":            crop,
        "disease":         disease,
        "confidence":      conf,
        "low_confidence":  low_conf,
        "user_query":      user_query.strip(),
        "language_code":   lang_code,
        "language_name":   lang_name,
        "retrieval_query": retrieval_query,
        "llm_context":     llm_context,
    }
