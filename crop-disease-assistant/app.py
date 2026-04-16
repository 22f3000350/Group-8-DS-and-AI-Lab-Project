"""
app.py
------
Redesigned Gradio UI for the Multimodal AI Assistant for Smart Agriculture.
Professional two-panel layout with:
  - Instruction page shown on first visit (and via 'i' button)
  - Tabbed results panel (Detection / Advisory / Audio)
  - Confidence progress bar
  - Semantic disease/crop/severity badges
  - Preset example questions
  - Custom green theme (see theme.py)

Pipeline:
  Image → MobileNet inference → Context Builder
                                      ↓
  Audio/Text → Whisper ASR → Retriever (ChromaDB)
                                      ↓
                               Groq LLM Generator
                                      ↓
                               gTTS → Audio output
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.inference       import load_model, predict
from pipeline.context_builder import build_context
from pipeline.generator       import generate
from rag.retriever            import retrieve
from voice.asr                import transcribe
from voice.tts                import synthesize, cleanup
from ui.theme                    import get_theme
from ui.combined_html            import COMBINED_HTML

# ── Load model at startup ────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "model", "mobilenet.pth")
MODEL_NAME    = "mobilenet_v3_large"
MODEL_DROPOUT = 0.4704167012838658

print("[app] Loading MobileNet model...")
try:
    model = load_model(MODEL_PATH, model_name=MODEL_NAME, dropout=MODEL_DROPOUT)
    print("[app] Model loaded successfully.")
except Exception as e:
    print(f"[app] WARNING: Could not load model: {e}")
    model = None

from voice.asr import _get_model as load_whisper
print("[app] Loading Whisper model...")
try:
    load_whisper("small")
    print("[app] Whisper loaded successfully.")
except Exception as e:
    print(f"[app] WARNING: Could not load Whisper: {e}")

# ── Language config ───────────────────────────────────────────────────────────
LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Bengali", "Malayalam", "Kannada"]
LANG_CODE  = {
    "English": "en", "Hindi": "hi", "Tamil": "ta",
    "Telugu": "te", "Bengali": "bn", "Malayalam": "ml", "Kannada": "kn",
}

# ── Result formatting helpers ─────────────────────────────────────────────────
def _confidence_bar_html(confidence: float, low: bool) -> str:
    pct   = int(confidence * 100)
    color = "#E24B4A" if low else ("#EF9F27" if pct < 75 else "#1D9E75")
    label_color = "#A32D2D" if low else ("#854F0B" if pct < 75 else "#0F6E56")
    return f"""
<div class="conf-bar-wrap">
  <div class="conf-bar-track">
    <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
  </div>
  <div class="conf-bar-label" style="color:{label_color};">{pct}% confidence</div>
</div>"""


def _disease_card_html(prediction: dict) -> str:
    crop     = prediction.get("crop", "Unknown")
    disease  = prediction.get("disease", "Unknown")
    conf     = prediction.get("confidence", 0.0)
    low      = prediction.get("low_confidence", False)

    severity = (
        "Low confidence" if low else
        ("Moderate" if conf < 0.85 else "High confidence")
    )

    warn_html = ""
    if low:
        warn_html = """
        <div class="badge badge-warn result-warning">
          ⚠ Low confidence — please consult your local KVK
        </div>
        """

    bar = _confidence_bar_html(conf, low)

    return f"""
<div class="result-card">
  {warn_html}

  <div class="result-header">
    <div class="result-icon">🌿</div>

    <div class="result-text">
      <div class="result-title">{disease}</div>
      <div class="result-subtitle">Detected in {crop}</div>
    </div>
  </div>

  <div class="result-badges">
    <span class="badge badge-crop">{crop}</span>
    <span class="badge badge-disease">Fungal</span>
    <span class="badge badge-severity">{severity}</span>
  </div>

  {bar}
</div>
"""


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(leaf_image, audio_input, text_input, language_choice):
    """Full pipeline: image + (audio or text) → detection card + advisory + audio."""

    # -- Resolve audio path --
    audio_path = None
    if audio_input is not None:
        audio_path = audio_input[0] if isinstance(audio_input, tuple) else audio_input
    print(f"[debug] audio_path: {audio_path} | text: {text_input}")

    # -- Validate image --
    if leaf_image is None:
        err = "<div class='result-card'>⚠️ Please upload a leaf image first.</div>"
        return err, "", None

    if model is None:
        err = "<div class='result-card'>⚠️ Model not loaded — check MODEL_PATH in app.py.</div>"
        return err, "", None

    # -- Step 1: Inference --
    try:
        pil_img    = Image.fromarray(leaf_image) if not isinstance(leaf_image, Image.Image) else leaf_image
        prediction = predict(model, pil_img)
    except Exception as e:
        return (f"<div class='result-card'>Error during inference: {e}</div>", "", None)

    # -- Step 2: ASR or text --
    user_query    = ""
    detected_lang = LANG_CODE.get(language_choice, "en")

    if audio_path and os.path.exists(audio_path):
        asr = transcribe(audio_path)
        if asr["success"]:
            user_query = asr["text"]
            if language_choice == "English" and asr.get("language_code", "en") != "en":
                detected_lang = asr["language_code"]
        else:
            card = _disease_card_html(prediction)
            msg  = (
                "⚠️ Voice recording was empty or unclear.\n\n"
                "Please check your microphone and try again, or use the text input instead."
            )
            return card, msg, None
    elif text_input and text_input.strip():
        user_query = text_input.strip()

    # -- Step 3: Context --
    context = build_context(prediction, user_query, detected_lang)

    # -- Step 4: RAG retrieval --
    try:
        retrieved = retrieve(context, top_k=4)
    except FileNotFoundError:
        retrieved = ""
        print("[app] ChromaDB not found — proceeding without RAG.")
    except Exception as e:
        retrieved = ""
        print(f"[debug] Retrieval error: {e}")

    # -- Step 5: Generate --
    try:
        llm_response = generate(context, retrieved)
    except Exception as e:
        llm_response = f"Error generating response: {e}"

    # -- Step 6: TTS --
    out_audio = synthesize(llm_response, detected_lang)

    # -- Build outputs --
    disease_card = _disease_card_html(prediction)
    return disease_card, llm_response, out_audio


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Crop Disease AI Assistant",
) as demo:

    # ── Inject styles + modal + header ────────────────────────────────
    gr.HTML(COMBINED_HTML)

    # ── Two-column main layout ───────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left: Inputs ─────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.HTML("<div class='panel-label'>Leaf image</div>")
            image_input = gr.Image(
                label="",
                type="pil",
                height=200,
                elem_classes=["upload-zone"],
            )

            gr.HTML("<div class='panel-label' style='margin-top:14px;'>Response language</div>")
            language_input = gr.Dropdown(
                choices=LANGUAGES,
                value="English",
                label="",
                container=False,
            )

            gr.HTML("<div class='panel-label' style='margin-top:14px;'>Ask a question</div>")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Voice (optional)",
                show_label=True,
            )
            text_input = gr.Textbox(
                placeholder="Or type your question here…",
                label="Text (optional)",
                show_label=True,
                lines=2,
            )

            submit_btn = gr.Button(
                "Analyse",
                variant="primary",
                size="lg",
                elem_id="analyse-btn",
            )

            # ── Preset example questions ──────────────────────────────────────
            gr.HTML("<div class='panel-label' style='margin-top:16px;'>Quick questions</div>")
            with gr.Row():
                ex1 = gr.Button("What treatment should I use?",    size="sm", variant="secondary")
                ex2 = gr.Button("Is this disease spreading?",       size="sm", variant="secondary")
            with gr.Row():
                ex3 = gr.Button("How to prevent next season?",      size="sm", variant="secondary")
                ex4 = gr.Button("Which pesticide is safe?",         size="sm", variant="secondary")

            def _set_q(q): return q
            ex1.click(fn=lambda: "What treatment should I use for this disease?",     outputs=text_input)
            ex2.click(fn=lambda: "Is this disease spreading to nearby plants?",        outputs=text_input)
            ex3.click(fn=lambda: "How can I prevent this disease next season?",        outputs=text_input)
            ex4.click(fn=lambda: "Which pesticide is safe and affordable for this?",   outputs=text_input)

        # ── Right: Outputs ───────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            # ── Detection ─────────────────────────────
            gr.HTML("<div class='panel-label'>Disease detection result</div>")
            disease_output = gr.HTML(
                value="<div class='result-card' style='color:#80807A;font-size:13px;'>"
                    "Upload a leaf image and click Analyse to see results.</div>",
                elem_id="results-section"
            )

            # ── Advisory ─────────────────────────────
            gr.HTML("<div class='panel-label' style='margin-top:16px;'>Expert advisory</div>")
            response_output = gr.Textbox(
                label="",
                lines=10,
                interactive=False,
                placeholder="The advisory will appear here after analysis…",
                elem_classes=["advisory-box"],
            )

            # ── Audio ─────────────────────────────
            gr.HTML("<div class='panel-label' style='margin-top:16px;'>Listen to advisory</div>")
            audio_output = gr.Audio(
                label="",
                type="filepath",
                autoplay=False,
            )

    # ── Wire up analyse button ───────────────────────────────────────────────
    submit_btn.click(
        fn=run_pipeline,
        inputs=[image_input, audio_input, text_input, language_input],
        outputs=[disease_output, response_output, audio_output],
    )

if __name__ == "__main__":
    demo.launch(
    theme=get_theme(),
    css="""
    .gradio-container { 
        padding: 0 !important;
        max-width: 100% !important;   
        width: 100% !important;
        margin: auto !important; 
    }
    footer { display: none !important; }
    """,
    js="""
    function attachHandlers() {
        const infoBtn = document.querySelector('#info-btn');
        const modal = document.querySelector('#modal-backdrop');
        const closeBtn = document.querySelector('#modal-close');
        const ctaBtn = document.querySelector('#modal-cta-btn');

        if (!infoBtn || !modal) return;

        infoBtn.onclick = () => modal.classList.add('open');

        if (closeBtn) closeBtn.onclick = () => modal.classList.remove('open');
        if (ctaBtn) ctaBtn.onclick = () => modal.classList.remove('open');

        modal.onclick = (e) => {
            if (e.target === modal) modal.classList.remove('open');
        };

        console.log("Modal handlers attached");
    }
    
    function setTextInput(value) {
        const textbox = document.querySelector('textarea');

        if (textbox) {
            textbox.value = value;
            textbox.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
    
    document.addEventListener('click', function(e) {
        if (e.target && e.target.id === 'modal-cta-btn') {

            if (selectedQuestion) {
                setTextInput(selectedQuestion);
            }

            // close modal
            const modal = document.querySelector('#modal-backdrop');
            if (modal) modal.classList.remove('open');
        }
    });

    let attached = false;

    const interval = setInterval(() => {
        const infoBtn = document.querySelector('#info-btn');

        if (!attached && infoBtn) {
            attachHandlers();
            attached = true;
            clearInterval(interval); 
        }
    }, 300);
    
    // Auto-open modal on first load
    setTimeout(() => {
        const modal = document.querySelector('#modal-backdrop');
        if (modal) {
            modal.classList.add('open');
        }
    }, 500);
    
    let selectedQuestion = "";

    // handle chip click
    document.addEventListener('click', function(e) {
        const chip = e.target.closest('.modal-example-chip');

        if (chip) {
            // remove active from all
            document.querySelectorAll('.modal-example-chip').forEach(c => {
                c.classList.remove('active');
            });

            // activate clicked
            chip.classList.add('active');

            // store question
            selectedQuestion = chip.getAttribute('data-question');
        }
    });
    
    
    
    """
)