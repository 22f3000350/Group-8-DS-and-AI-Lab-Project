# Developer Guide

## 1. Project Overview

This project is a multimodal crop-disease assistant for Indian farming use cases. It combines:

- Leaf-image classification with a trained MobileNet model
- Retrieval-augmented generation (RAG) over crop-disease documents
- Voice input using Whisper ASR
- Multilingual advisory generation using Groq
- Audio playback using gTTS
- A Gradio frontend for local use and Hugging Face Spaces deployment

The current repository contains two app variants:

- Root app: the current primary UI with custom theming and richer layout
- `space/` app: a mirrored Hugging Face Space-oriented copy of the project

## 2. System Architecture

The main runtime flow is:

1. User uploads a leaf image.
2. `pipeline/inference.py` predicts crop, disease, and confidence.
3. Optional text or recorded audio is accepted.
4. `voice/asr.py` transcribes audio with Whisper.
5. `pipeline/context_builder.py` builds a retrieval query and LLM context.
6. `rag/retriever.py` loads ChromaDB and retrieves relevant disease-management chunks.
7. `pipeline/generator.py` calls Groq to produce a farmer-friendly advisory.
8. `voice/tts.py` converts the advisory to MP3 using gTTS.
9. `app.py` renders the result card, text response, and audio output in Gradio.

## 3. Repository Layout

```text
crop-disease-assistant/
├── app.py
├── README.md
├── DEVELOPER_GUIDE.md
├── requirements.txt
├── .env.example
├── model/
│   └── mobilenet.pth
├── pipeline/
│   ├── __init__.py
│   ├── inference.py
│   ├── context_builder.py
│   └── generator.py
├── rag/
│   ├── __init__.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── docs/
│   └── chroma_db/
├── voice/
│   ├── __init__.py
│   ├── asr.py
│   └── tts.py
├── ui/
│   ├── combined_html.py
│   └── theme.py
└── space/
    ├── app.py
    ├── README.md
    ├── requirements.txt
    ├── pipeline/
    ├── rag/
    └── voice/
```

## 4. Development Environment

### Recommended baseline

- OS: Linux preferred for local development
- Python: 3.10 or 3.11 recommended
- `ffmpeg`: required for reliable Whisper audio decoding
- Optional GPU: supported automatically through PyTorch if available

### Create the environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment variables

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Required variable:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Notes:

- `app.py` loads environment variables with `python-dotenv`
- On Hugging Face Spaces, set `GROQ_API_KEY` in Space Secrets instead of committing `.env`

## 5. Dependencies

The authoritative pinned dependency lists are:

- [`requirements.txt`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/requirements.txt)
- [`space/requirements.txt`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/space/requirements.txt)

Important runtime packages used by the system:

- `gradio`: web UI
- `torch`, `torchvision`: model loading and inference
- `chromadb`: vector database
- `sentence-transformers`: embedding model support for retrieval
- `groq`: LLM inference API client
- `openai-whisper`: speech-to-text
- `gTTS`: text-to-speech
- `PyMuPDF`: PDF text extraction
- `python-dotenv`: environment loading
- `Pillow`: image handling

Important implementation note:

- The pinned requirements currently include CUDA-specific packages. They are useful on GPU-enabled systems but can be heavier than necessary on CPU-only deployments.

## 6. Required Project Assets

### 6.1 Model file

Expected location:

```text
model/mobilenet.pth
```

The current root app is configured for:

- `MODEL_NAME = "mobilenet_v3_large"`
- `MODEL_DROPOUT = 0.4704167012838658`

If a different training artifact is used, update those values in [`app.py`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/app.py) and optionally in [`space/app.py`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/space/app.py).

### 6.2 Knowledge-base documents

RAG source documents live in:

```text
rag/docs/
```

Recommended naming convention:

```text
{crop}_{disease}_{source}.pdf
```

Examples:

- `rice_brown_spot_icar.pdf`
- `corn_gray_leaf_spot_tnau.pdf`
- `wheat_yellow_rust_fao.pdf`
- `potato_late_blight_icar.pdf`
- `sugarcane_red_rot_icar.pdf`

This naming pattern matters because `rag/ingest.py` parses crop, disease, and source metadata from filenames.

### 6.3 Chroma vector index

Persistent index location:

```text
rag/chroma_db/
```

The index is already present in this repository. To rebuild it:

```bash
python rag/ingest.py
```

## 7. Setup Procedure From Scratch

### Step 1: Clone and enter the repo

```bash
git clone <your-repo-url>
cd crop-disease-assistant
```

### Step 2: Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Add environment configuration

```bash
cp .env.example .env
```

Set:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Verify required assets

Check that these exist:

- `model/mobilenet.pth`
- `rag/docs/`
- `rag/chroma_db/` or the source documents needed to rebuild it

### Step 6: Rebuild retrieval index if needed

```bash
python rag/ingest.py
```

### Step 7: Run the application

```bash
python app.py
```

## 8. Script-by-Script Reference

### Root application

#### `app.py`

Main entry point for the current local app.

Responsibilities:

- Loads `.env`
- Loads MobileNet model at startup
- Preloads Whisper `small`
- Defines supported UI languages
- Runs the end-to-end pipeline in `run_pipeline(...)`
- Builds the custom Gradio interface
- Launches the app with custom CSS and JavaScript

Key configuration inside the file:

- `MODEL_PATH`
- `MODEL_NAME`
- `MODEL_DROPOUT`
- `LANGUAGES`
- `LANG_CODE`

#### `README.md`

High-level project overview and quick-start instructions intended for users or evaluators.

### Inference pipeline

#### `pipeline/inference.py`

Loads the trained classifier and performs image prediction.

Key behaviors:

- Supports multiple checkpoint styles:
  - Full PyTorch model in `.pth`
  - Checkpoint dict containing `model_state`
  - Raw `state_dict`
  - Joblib `.pkl` state dict
- Rebuilds either `mobilenet_v2` or `mobilenet_v3_large`
- Applies ImageNet normalization and resizing to `224 x 224`
- Returns crop, disease, confidence, and top probabilities

Important constants:

- `CLASS_NAMES`
- `CONFIDENCE_THRESHOLD = 0.60`
- `DEVICE = cuda if available else cpu`

#### `pipeline/context_builder.py`

Converts prediction output and user input into a structured retrieval/generation context.

Responsibilities:

- Normalizes language names and codes
- Creates the Chroma retrieval query
- Builds a confidence-aware LLM context header

#### `pipeline/generator.py`

Creates the multilingual advisory using the Groq API.

Responsibilities:

- Reads `GROQ_API_KEY`
- Chooses model `llama-3.3-70b-versatile`
- Applies language-specific prompt instructions
- Falls back to a translated canned response if no RAG context is retrieved

Important notes:

- Responses are intentionally constrained to avoid inventing pesticide dosages
- If `GROQ_API_KEY` is missing, generation returns an explicit error string

### Retrieval pipeline

#### `rag/ingest.py`

Builds the ChromaDB vector store from local PDF, TXT, or Markdown documents.

Responsibilities:

- Scans `rag/docs/`
- Extracts text from PDFs with PyMuPDF
- Parses metadata from filenames
- Splits content into chunks with section-aware heuristics
- Embeds chunks using `intfloat/multilingual-e5-large`
- Stores them in `rag/chroma_db/`

Important constants:

- `COLLECTION = "crop_disease_knowledge"`
- `EMBED_MODEL = "intfloat/multilingual-e5-large"`
- `CHUNK_SIZE = 400`
- `CHUNK_OVERLAP = 50`

Implementation note:

- The script currently calls `collection.upsert(...)` twice per batch. The second call writes the actual text chunks, so the final stored documents are correct, but the first call is redundant.

#### `rag/retriever.py`

Queries the ChromaDB store at runtime.

Retrieval strategy:

1. Crop + disease metadata filter
2. Crop-only fallback
3. Unfiltered semantic search fallback

Special case:

- Low-confidence predictions use broader retrieval behavior

### Voice modules

#### `voice/asr.py`

Speech-to-text using Whisper.

Responsibilities:

- Lazy-loads the Whisper model
- Validates recorded audio size
- Auto-detects language
- Returns both transcript text and language code

Default runtime choice:

- `whisper-small`

#### `voice/tts.py`

Text-to-speech using gTTS.

Responsibilities:

- Maps supported languages to gTTS codes
- Writes MP3 output to a temp file
- Truncates very long responses before synthesis

Important note:

- gTTS requires internet access during synthesis

### UI modules

#### `ui/combined_html.py`

Contains a large embedded HTML/CSS block used by the root app for:

- Header styling
- Modal behavior
- Upload zone styling
- Result-card visuals
- Responsive layout polish

#### `ui/theme.py`

Defines the custom Gradio theme.

Responsibilities:

- Creates custom green and neutral color ramps
- Uses `DM Sans` and `DM Mono`
- Applies only theme properties supported by the installed Gradio version

### Mirrored Hugging Face Space copy

#### `space/app.py`

Space-oriented copy of the app with a simpler UI than the root `app.py`.

#### `space/pipeline/*`

Mirrored pipeline modules for the Space deployment variant.

#### `space/rag/*`

Mirrored retrieval modules and document set for the Space variant.

#### `space/voice/*`

Mirrored ASR/TTS modules for the Space variant.

#### `space/README.md`

Deployment-oriented README for the Space copy.

## 9. Runtime Configuration Summary

The most important values another developer may need to adjust are:

- In [`app.py`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/app.py):
  - `MODEL_PATH`
  - `MODEL_NAME`
  - `MODEL_DROPOUT`
- In [`pipeline/generator.py`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/pipeline/generator.py):
  - `GROQ_MODEL`
- In [`rag/ingest.py`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/rag/ingest.py) and [`rag/retriever.py`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/rag/retriever.py):
  - `EMBED_MODEL`
  - Chroma collection name and storage path
- In [`voice/asr.py`](/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/voice/asr.py):
  - Whisper model size

## 10. Reproducing Similar Results

To get results similar to the current project behavior, keep the following aligned:

- Use the same trained MobileNet checkpoint and matching architecture/dropout settings
- Keep the same `CLASS_NAMES` ordering in `pipeline/inference.py`
- Use the same document corpus and file naming convention in `rag/docs/`
- Rebuild Chroma using the same embedding model: `intfloat/multilingual-e5-large`
- Keep Groq generation on the same model or expect different wording and quality
- Keep Whisper on `small` if you want memory usage similar to the current deployment target

If any of these change, advisory quality and disease matching behavior may shift noticeably.

## 11. Local Validation Checklist

After setup, verify the following:

1. `python app.py` starts without import errors.
2. The model loads successfully at startup.
3. Whisper loads successfully at startup.
4. Image-only inference produces a disease result.
5. Text query plus image returns a Groq-generated advisory.
6. Audio query transcribes correctly and returns an advisory.
7. Audio playback MP3 is generated successfully.

## 12. Hugging Face Spaces Deployment

For deployment using the repository root README flow:

1. Create a new Gradio Space.
2. Push the project to the Space or connect the Git repository.
3. Add `GROQ_API_KEY` in Space Secrets.
4. Ensure `model/mobilenet.pth` and `rag/chroma_db/` are present, or regenerate the index during startup.

Deployment notes:

- `whisper-small` is a practical memory compromise for Spaces
- The prebuilt Chroma directory avoids indexing at boot time
- If using the `space/` mirror instead of the root app, keep both code paths synchronized manually

## 13. Known Implementation Notes

- The repository currently includes both root and `space/` copies of the codebase, so changes may need to be applied twice.
- `gTTS` requires network access, even if the rest of the app can run mostly locally.
- Whisper generally depends on `ffmpeg` being available on the host system.
- If the Chroma index is missing, the app still runs, but advisory generation falls back to no-context behavior.
- Low-confidence image predictions are intentionally surfaced as tentative diagnoses rather than hard labels.

## 14. Suggested Future Improvements

- Remove duplicated code by making the root and `space/` apps share a single source tree
- Split CPU and GPU requirement files to reduce install overhead
- Add automated tests for model loading, retrieval, and pipeline smoke checks
- Add a startup health-check script for verifying model, Chroma, Whisper, and Groq connectivity
- Replace the committed model artifact and vector index flow with a documented artifact download step if repository size becomes a concern
