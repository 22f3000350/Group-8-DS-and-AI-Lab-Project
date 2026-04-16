"""
generator.py
------------
Calls the Groq API with the retrieved context and generates
a farmer-friendly disease advisory response in the user's language.

API key: Set as environment variable GROQ_API_KEY
         On HF Spaces: add under Settings → Secrets
"""

import os
from groq import Groq

# ── Model selection ──────────────────────────────────────────────────────────
# llama-3.1-8b-instant  → fastest, good for simple queries
# llama-3.3-70b-versatile → better multilingual quality, slower
# gemma2-9b-it          → good multilingual balance
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Language instructions for the prompt ─────────────────────────────────────
LANGUAGE_INSTRUCTIONS = {
    "hi": "Respond entirely in Hindi (Devanagari script). Use simple vocabulary suitable for a farmer.",
    "bn": "Respond entirely in Bengali (Bengali script). Use simple vocabulary suitable for a farmer.",
    "ta": "Respond entirely in Tamil (Tamil script). Use simple vocabulary suitable for a farmer.",
    "te": "Respond entirely in Telugu (Telugu script). Use simple vocabulary suitable for a farmer.",
    "ml": "Respond entirely in Malayalam (Malayalam script). Use simple vocabulary suitable for a farmer.",
    "kn": "Respond entirely in Kannada (Kannada script). Use simple vocabulary suitable for a farmer.",
    "en": "Respond in simple English. Use clear, practical language suitable for a farmer.",
}

NO_CONTEXT_RESPONSE = {
    "hi": "इस बीमारी के बारे में हमारे पास विस्तृत जानकारी उपलब्ध नहीं है। कृपया अपने स्थानीय कृषि अधिकारी या KVK से संपर्क करें।",
    "bn": "এই রোগ সম্পর্কে আমাদের কাছে বিস্তারিত তথ্য নেই। দয়া করে আপনার স্থানীয় কৃষি অফিসার বা KVK-এর সাথে যোগাযোগ করুন।",
    "ta": "இந்த நோய் பற்றிய விரிவான தகவல்கள் எங்களிடம் இல்லை. உங்கள் உள்ளூர் விவசாய அதிகாரி அல்லது KVK ஐ தொடர்பு கொள்ளுங்கள்.",
    "te": "ఈ వ్యాధి గురించి మాకు వివరమైన సమాచారం లేదు. దయచేసి మీ స్థానిక వ్యవసాయ అధికారి లేదా KVK ని సంప్రదించండి.",
    "ml": "ഈ രോഗത്തെക്കുറിച്ച് ഞങ്ങളുടെ പക്കൽ വിശദമായ വിവരങ്ങൾ ഇല്ല. ദയവായി നിങ്ങളുടെ പ്രാദേശിക കൃഷി ഓഫീസർ അല്ലെങ്കിൽ KVK-ഉമായി ബന്ധപ്പെടുക.",
    "kn": "ಈ ರೋಗದ ಬಗ್ಗೆ ನಮ್ಮ ಬಳಿ ವಿವರವಾದ ಮಾಹಿತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿ ಅಥವಾ KVK ಅನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    "en": "We don't have detailed information about this disease in our knowledge base. Please consult your local agricultural officer or Krishi Vigyan Kendra (KVK).",
}


def build_system_prompt(language_code: str) -> str:
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language_code, LANGUAGE_INSTRUCTIONS["en"])
    return f"""You are an expert agricultural advisor helping Indian farmers diagnose and manage crop diseases.

{lang_instruction}

When given a disease diagnosis and context from agricultural guidelines:
1. Briefly confirm the disease and what it does to the crop (1-2 sentences)
2. Explain the main symptoms the farmer might see (2-3 points)
3. Provide practical treatment steps (2-3 actionable steps)
4. Suggest prevention for next season (1-2 points)
5. If relevant, mention when to contact a local KVK or agriculture officer

Rules:
- Prioritize answering the farmer's specific question using the provided context.
- Base your advice on the provided agricultural guidelines. Do not invent chemical names, dosages, or treatments not mentioned.
- If the context does not contain information to answer the farmer's specific question, say so clearly and recommend consulting a local expert.
- Keep the response concise and practical — farmers need actionable advice, not academic text.
- Never provide dosages for pesticides unless explicitly stated in the context.
"""


def generate(context: dict, retrieved_context: str) -> str:
    """
    Generate a response using Groq LLM.

    Parameters
    ----------
    context          : dict   Output from context_builder.build_context()
    retrieved_context: str    Output from retriever.retrieve()

    Returns
    -------
    str   The LLM's response in the user's language.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "Error: GROQ_API_KEY not set. Add it as an environment variable or HF Space Secret."

    lang_code = context.get("language_code", "en")

    # If no context retrieved, return a pre-translated fallback message
    if not retrieved_context.strip():
        print("[generator] No context retrieved. Returning fallback response.")
        return NO_CONTEXT_RESPONSE.get(lang_code, NO_CONTEXT_RESPONSE["en"])

    # Build the user message
    user_message = f"""
{context['llm_context']}

AGRICULTURAL GUIDELINES FROM VERIFIED SOURCES:
{retrieved_context}

FARMER'S QUESTION: {context['user_query'] if context['user_query'] else 'What is this disease and how should I treat it?'}

Please provide advice based strictly on the above guidelines.
"""

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt(lang_code)},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=600,
            temperature=0.3,  # Low temperature = more factual, less creative
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[generator] Groq API error: {e}")
        return f"Sorry, could not generate a response at this time. Please try again. (Error: {e})"
