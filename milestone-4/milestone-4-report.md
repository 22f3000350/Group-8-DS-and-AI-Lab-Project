# Milestone 4 Report
### RAG Pipeline, System Integration & Deployment

---

App live on Hugging Face: [LINK](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant)

## 1. System Overview

This report covers the full integration of the crop disease detection model with a Retrieval Augmented Generation (RAG) advisory pipeline, multimodal voice interface, and deployment on Hugging Face Spaces.

### 1.1 What Was Built

| Component | Description |
|-----------|-------------|
| Model Integration | MobileNet V3 Large inference module with confidence thresholding |
| RAG Pipeline | Document ingestion, ChromaDB vector store, metadata-filtered retrieval |
| LLM Advisory | Groq API (Llama 3.3 70B) with verified-source grounding |
| Voice Input | Whisper ASR — speech-to-text in 6 Indian languages |
| Voice Output | gTTS — text-to-speech in 6 Indian languages |
| UI | Gradio 6.x interface with image, voice, and text input |
| Deployment | Hugging Face Spaces (free tier) |

### 1.2 Project Structure

```
crop-disease-assistant/
├── app.py                        ← Gradio entry point
├── requirements.txt
├── model/
│   └── mobilenet.pth             ← Trained MobileNet V3 Large
├── pipeline/
│   ├── inference.py              ← Model prediction
│   ├── context_builder.py        ← Label → structured query
│   └── generator.py              ← Groq LLM response
├── rag/
│   ├── ingest.py                 ← One-time document indexing
│   ├── retriever.py              ← ChromaDB query logic
│   ├── docs/                     ← Agricultural PDFs
│   └── chroma_db/                ← Persisted vector index
└── voice/
    ├── asr.py                    ← Whisper transcription
    └── tts.py                    ← gTTS synthesis
```

---

## 2. Model Integration

### 2.1 Model Details

The best model from Milestone 3 Optuna hyperparameter search was used directly:

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNet V3 Large |
| Trial | #4 (best) |
| Optimizer | AdamW |
| Learning Rate | 0.000428 |
| Dropout | 0.4704 |
| Batch Size | 16 |
| Test Accuracy | 92.1% |
| Test Macro F1 | 90.0% |
| Val Macro F1 | 92.5% |
| Test Loss | 0.207 |

### 2.2 Inference Pipeline

The inference module (`pipeline/inference.py`) loads the saved `.pth` file and applies the exact same transform pipeline used during training:

```
Input Image
    → RGB Conversion (safety step)
    → Resize to (224, 224) with LANCZOS resampling
    → ToTensor (pixel values 0.0–1.0)
    → Normalize (ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    → MobileNet V3 Large forward pass
    → Softmax
    → (class_label, confidence_score)
```

**Why LANCZOS resize instead of Resize(256) → CenterCrop(224):**
Training images were preprocessed to exactly 224×224 using LANCZOS in the M2 pipeline. Using the standard ImageNet eval convention (Resize 256 → CenterCrop 224) would introduce a distribution mismatch at inference. The transform was aligned to match training exactly.

### 2.3 Confidence Thresholding

A confidence threshold of 0.60 is applied at inference:

- **Above 0.60** → disease label + crop passed to RAG pipeline normally
- **Below 0.60** → `low_confidence` flag set → retrieval broadens to crop-level only → LLM generates a cautious response advising KVK consultation

This prevents confidently wrong advice reaching a farmer when the model is uncertain — for example on blurry, partially damaged, or non-leaf images.

### 2.4 Class Label Parsing

The 17 class labels follow the format `Crop___Disease` (three underscores) or `Sugarcane__Disease` (two underscores). The inference module normalises both formats:

```python
parts   = label.replace("__", "___").split("___")
crop    = parts[0]           # "Corn"
disease = parts[1].replace("_", " ")  # "Gray Leaf Spot"
```

Both are lowercased before being passed to the retriever to match ChromaDB metadata.

---

## 3. RAG Pipeline

### 3.1 What is RAG

Retrieval Augmented Generation (RAG) grounds LLM responses in verified external documents rather than the model's training memory. The LLM is instructed to answer only from the retrieved context — preventing hallucinated chemical names, wrong dosages, or fabricated treatment protocols from reaching a farmer.

### 3.2 Knowledge Base

Agricultural documents were collected from the following verified sources:

| Source | Coverage |
|--------|----------|
| ICAR (Indian Council of Agricultural Research) | Crop disease management, field advisories |
| TNAU Agritech Portal | Disease identification, chemical control |
| FAO (Food and Agriculture Organization) | Crop protection manuals |

Documents follow the naming convention `{crop}_{disease}_{source}.pdf` (e.g. `rice_brown_spot_icar.pdf`) so crop and disease metadata can be auto-parsed at ingest time without manual tagging.

**Total indexed:** 34 chunks across all documents

### 3.3 Ingestion Pipeline (`rag/ingest.py`)

Run once locally to build the ChromaDB index. Steps:

1. **Text Extraction** — PyMuPDF (`fitz`) reads each PDF page and extracts raw text
2. **Metadata Parsing** — crop, disease, and source are parsed from the filename
3. **Smart Chunking** — text is split at natural section boundaries (Symptoms, Management, Treatment, Prevention headings) before falling back to fixed 400-token windows with 50-token overlap
4. **Embedding** — each chunk is embedded using `intfloat/multilingual-e5-large`
5. **Storage** — chunks, embeddings, and metadata stored in ChromaDB at `rag/chroma_db/`

### 3.4 Embedding Model

**Model:** `intfloat/multilingual-e5-large`

| Property | Value |
|----------|-------|
| Parameters | 560M |
| Languages | 100+ including Hindi, Tamil, Telugu, Bengali, Malayalam, Kannada |
| Embedding Dimension | 1024 |
| Similarity Metric | Cosine |

**Why multilingual-e5-large over English-only models:**
A farmer querying in Tamil must be semantically matched to English agricultural documents without a translation step. English-only embedding models (e.g. OpenAI ada-002) would produce poor or random similarity scores for non-English queries. multilingual-e5-large was trained on parallel corpora across 100 languages — Indian language queries embed into the same vector space as English text.

### 3.5 Retrieval Strategy (`rag/retriever.py`)

Three-stage fallback retrieval:

```
Stage 1: Metadata filter → crop == "corn" AND disease == "gray leaf spot"
         + semantic similarity ranking within filtered set
              ↓ (if < 2 results)
Stage 2: Metadata filter → crop == "corn" only
         + semantic similarity ranking within filtered set
              ↓ (if < 2 results)
Stage 3: Unfiltered semantic search across all 34 chunks
```

**Why metadata pre-filtering matters:**
Rust diseases across different crops (corn rust, wheat rust) use very similar vocabulary in agricultural literature. Without the crop + disease filter, a query about corn rust would retrieve wheat rust chunks because they're semantically similar. Pre-filtering pins retrieval to the exact disease category before semantic ranking.

Returns top 4 chunks joined as a single context string.

### 3.6 Context Builder (`pipeline/context_builder.py`)

Bridges the model output and the retriever. Takes the raw prediction dict and constructs:

| Output Field | Example Value |
|-------------|---------------|
| `crop` | `"corn"` |
| `disease` | `"gray leaf spot"` |
| `retrieval_query` | `"treatment and prevention of gray leaf spot in corn crop"` |
| `llm_context` | `"Disease detected: Gray Leaf Spot in Corn (confidence: 91%)"` |
| `language_code` | `"ta"` |
| `low_confidence` | `False` |

---

## 4. LLM Advisory Generation

### 4.1 Model and API

**LLM:** Llama 3.3 70B  
**Provider:** Groq API (free tier)  
**Latency:** < 1 second on Groq LPU hardware

**Why Groq over OpenAI / Hugging Face Inference API:**

| Provider | Cost | Latency | Indian Language Quality |
|----------|------|---------|------------------------|
| Groq (Llama 3.3 70B) | Free | < 1s | Good |
| OpenAI GPT-4o | Paid | 2–5s | Excellent |
| HF Inference API | Free (limited) | 5–30s | Variable |
| Local Ollama | Free | 10–60s (CPU) | Good |

Groq provides production-grade latency at zero cost — critical for a demo environment.

### 4.2 Prompt Design

The LLM receives a system prompt and a user message on every call.

**System prompt instructs the model to:**
- Answer only from the provided context — never from training memory
- Structure responses as: diagnosis → symptoms → treatment → prevention
- Use simple farmer-friendly vocabulary — no academic language
- Respond entirely in the user's specified language
- Never provide chemical dosages unless explicitly stated in the retrieved context
- Recommend KVK consultation when uncertain

**Temperature:** 0.3 — conservative and factual, stays close to retrieved context

**Max tokens:** 600 — sufficient for a complete advisory without being verbose

### 4.3 Fallback Handling

| Condition | Behaviour |
|-----------|-----------|
| No context retrieved | Pre-written KVK referral message returned in user's language — LLM not called |
| Low confidence prediction | LLM instructed to express uncertainty and recommend professional consultation |
| Groq API error | Error message returned, user prompted to retry |

Pre-written fallback messages are hardcoded in all 6 Indian languages in `generator.py` — ensuring a meaningful response even with zero retrieved context and zero LLM dependency.

---

## 5. Voice Interface

### 5.1 Speech-to-Text — Whisper ASR

**Model:** `openai/whisper-small`  
**Parameters:** 244M  
**Module:** `voice/asr.py`

**Why whisper-small over whisper-medium or whisper-large:**
HF Spaces free tier provides 16GB RAM shared across MobileNet, the embedding model, and Whisper. whisper-small fits within this budget while maintaining acceptable accuracy on clear speech in Indian languages. whisper-large would exceed RAM limits.

**Processing pipeline:**

```
Audio file (WAV from Gradio microphone)
    → Whisper encoder (audio spectrogram → feature representation)
    → Language detection (probability distribution over 99 languages)
    → Whisper decoder (autoregressive text generation)
    → {text: "...", language: "tamil"}
    → Language code mapping: "tamil" → "ta"
```

**Language auto-detection:** Whisper detects the spoken language automatically. If a farmer speaks Tamil without changing the UI dropdown, Whisper returns `language_code = "ta"` which overrides the English default and flows through the entire pipeline — retrieval, LLM generation, and TTS all respond in Tamil automatically.

### 5.2 Text-to-Speech — gTTS

**Library:** gTTS (Google Text-to-Speech)  
**Module:** `voice/tts.py`

**Why gTTS over alternatives:**

| Option | Cost | Indian Languages | Offline |
|--------|------|-----------------|---------|
| gTTS | Free | All 6 supported | No |
| pyttsx3 | Free | Poor support | Yes |
| ElevenLabs | Paid | Limited | No |
| Azure TTS | Paid | All supported | No |

gTTS wraps Google's TTS API — the same engine as Google Translate audio. No API key required. Supports all 6 target languages natively.

**Processing pipeline:**

```
LLM response text + language_code
    → Text truncation to 1500 characters (≈ 1 min audio)
    → gTTS API call to Google servers
    → MP3 file saved to /tmp/tts_random.mp3
    → File path returned to Gradio audio player
```

**Supported languages:**

| Language | gTTS Code |
|----------|-----------|
| Hindi | `hi` |
| Bengali | `bn` |
| Tamil | `ta` |
| Telugu | `te` |
| Malayalam | `ml` |
| Kannada | `kn` |
| English | `en` |

---

## 6. User Interface

### 6.1 Gradio Application (`app.py`)

**Framework:** Gradio 6.10.0  
**Hosting:** Hugging Face Spaces (Gradio SDK)

**Input components:**
- `gr.Image` — leaf image upload (type=pil)
- `gr.Audio` — microphone or file upload (type=filepath)
- `gr.Textbox` — text question input
- `gr.Dropdown` — language selection (7 options)

**Output components:**
- `gr.Textbox` — disease detection result with crop, disease, confidence
- `gr.Textbox` — LLM advisory response
- `gr.Audio` — TTS audio playback

### 6.2 End-to-End Request Flow

```
User uploads leaf image
    → inference.py → (crop, disease, confidence, low_confidence)
         ↓
User provides audio or text question
    → asr.py → (transcribed_text, language_code)   [if audio]
         ↓
context_builder.py → structured context dict
         ↓
retriever.py → top 4 relevant document chunks
         ↓
generator.py (Groq API) → advisory text in user's language
         ↓
tts.py → MP3 audio file
         ↓
Gradio outputs: disease label + advisory text + audio player
```

---

## 7. Deployment

### 7.1 Infrastructure Stack

| Component | Service | Cost |
|-----------|---------|------|
| App Hosting | Hugging Face Spaces | Free |
| LLM Inference | Groq API | Free |
| Vector Store | ChromaDB (local) | Free |
| ASR | Whisper (local) | Free |
| TTS | gTTS | Free |
| CI/CD | GitHub (manual push) | Free |
| Model Storage | HF Spaces Files | Free |

**Total infrastructure cost: $0**

### 7.2 Hardware (HF Spaces Free Tier)

| Resource | Available | Usage |
|----------|-----------|-------|
| RAM | 16GB | MobileNet + Whisper-small + e5-large ≈ 8–10GB |
| CPU | 2 vCPU | Inference runs on CPU |
| Storage | 50GB | Model + ChromaDB + app code |
| GPU | None (free tier) | CPU inference only |

### 7.3 Model and Index Deployment

The model file (`mobilenet.pth`, 16MB) and ChromaDB index (`rag/chroma_db/`) are excluded from git due to binary file size restrictions. Both are uploaded directly via the HF Spaces file browser and persist permanently across deployments.

Application code is pushed to HF Spaces manually via:

```bash
git push hf main --force
```

---

## 8. Key Design Decisions

### 8.1 Why RAG over Pure LLM

A pure LLM (without retrieval) would generate agricultural advice from its training data — which may be outdated, geographically incorrect for Indian conditions, or simply hallucinated. RAG ensures every piece of advice is traceable to a specific ICAR, TNAU, or FAO document. If a document does not mention a treatment, the system does not recommend it.

### 8.2 Why MobileNet over ResNet-50

MobileNet V3 Large uses depthwise separable convolutions — architecturally designed for efficient inference with fewer parameters (~5.4M trainable) compared to ResNet-50 (~25.6M). For a deployment target of rural environments with CPU-only inference, this results in significantly faster prediction times with only marginal accuracy trade-off. The achieved 92.1% accuracy and 90.0% macro F1 on 17 classes validates this choice.

### 8.3 Why Macro F1 as Primary Metric

The dataset has a 14.9x class imbalance. Accuracy is misleading under imbalance — a model predicting only majority classes could achieve high accuracy while failing entirely on Sugarcane (100 images/class). Macro F1 weights all 17 classes equally, penalising poor performance on minority classes. A 90.0% macro F1 indicates the model performs well across all classes including the severely underrepresented Sugarcane classes.

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

| Limitation | Impact |
|-----------|--------|
| 34 document chunks in knowledge base | Narrow advisory coverage |
| Whisper-small accuracy on heavy accents | Occasional transcription errors |
| gTTS requires internet connection | Cannot work fully offline |
| CPU-only inference on HF free tier | ~2–3 second prediction latency |
| No severity classification | Cannot distinguish mild vs severe disease |

### 9.2 Future Extensions

- **Severity classification** (mild / moderate / severe) — additional output head on the classification model
- **Larger knowledge base** — add state agriculture university bulletins, KVK field guides
- **Multimodal RAG** — extract and caption figures from agricultural PDFs using a vision LLM (LLaVA / Gemini Flash) so image-heavy documents are fully indexed
- **Offline mode** — quantized local LLM (Llama 3.2 3B GGUF) + FAISS for fully offline rural deployment
- **More crops** — extend beyond 5 crops to cover additional Indian staples (Soybean, Groundnut, Cotton)
- **Severity-aware retrieval** — tag document chunks with severity level metadata and filter by predicted severity

---

## 10. Results Summary

| Metric | Value |
|--------|-------|
| Test Accuracy | 92.1% |
| Test Macro F1 | 90.0% |
| Test Micro F1 | 92.1% |
| Val Macro F1 | 92.5% |
| Test Loss | 0.207 |
| Classes | 17 |
| Crops | 5 |
| Languages Supported | 7 (6 Indian + English) |
| Knowledge Base Chunks | 34 |
| Infrastructure Cost | $0 |
| Deployment | Live on Hugging Face Spaces |
