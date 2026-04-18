# Developer Guide
## Model Development Pipeline and Final Application

---

This combined guide contains both parts of the developer workflow, one after another:

1. The notebook-based model-development pipeline used to build the classifier for the app
2. The final multimodal application stack that loads and uses the trained model

## Table of Contents

[Part A. Model Development Guide](#part-a-model-development-guide)
- [A.1 Purpose](#1-purpose)
- [A.2 Repository Files Used](#2-repository-files-used)
- [A.3 Recommended Runtime](#3-recommended-runtime)
- [A.4 Dependencies](#4-dependencies)
- [ A.5 Dataset Source](#5-dataset-source)
- [A.6 End-to-End Workflow](#6-end-to-end-workflow)
- [A.7 Reproduction Instructions](#7-reproduction-instructions)
- [A.8 Important Implementation Notes](#8-important-implementation-notes)
- [A.9 Minimal Developer Checklist](#9-minimal-developer-checklist)
- [A.10 Summary](#10-summary)

[Part B. Final Application Developer Guide](#part-b-final-application-developer-guide)
- [B.1 Project Overview](#1-project-overview)
- [B.2 System Architecture](#2-system-architecture)
- [B.3 Repository Layout](#3-repository-layout)
- [B.4 Development Environment](#4-development-environment)
- [B.5 Dependencies](#5-dependencies-1)
- [B.6 Required Project Assets](#6-required-project-assets)
- [B.7 Setup Procedure From Scratch](#7-setup-procedure-from-scratch)
- [B.8 Script-by-Script Reference](#8-script-by-script-reference)
- [B.9 Runtime Configuration Summary](#9-runtime-configuration-summary)
- [B.10 Reproducing Similar Results](#10-reproducing-similar-results)
- [B.11 Local Validation Checklist](#11-local-validation-checklist)
- [B.12 Hugging Face Spaces Deployment](#12-hugging-face-spaces-deployment)
- [B.13 Known Implementation Notes](#13-known-implementation-notes)
- [B.14 Suggested Future Improvements](#14-suggested-future-improvements)

---

# Part A. Model Development Guide
## Crop Disease Detection Pipeline for the App


## 1. Purpose

This section documents the exact model-development pipeline currently used by the application.

For the deployed app workflow, only these two notebooks are part of the image-classification pipeline:

- `notebooks/eda-and-preprocessing-pipeline.ipynb`
- `notebooks/mobilenet.ipynb`

The process is:

1. Run the EDA and preprocessing notebook to download, inspect, clean, resize, split, and augment the dataset.
2. Run the MobileNet notebook to train and evaluate the classifier on the processed dataset.

The other model notebooks in `notebooks/` are exploratory alternatives and are not the primary training path used for the app.

---

## 2. Repository Files Used

| File | Role |
|------|------|
| `notebooks/eda-and-preprocessing-pipeline.ipynb` | Dataset download, EDA, cleaning, duplicate removal, resizing, stratified split, augmentation, processed dataset export |
| `notebooks/mobilenet.ipynb` | MobileNet transfer learning, Optuna hyperparameter tuning, validation selection, final test evaluation, model artifact export |

---

## 3. Recommended Runtime

These notebooks are written for **Google Colab** and assume access to:

- Google Drive
- Kaggle dataset download
- Python 3.x
- PyTorch / torchvision

Training can run on CPU, but a GPU-enabled Colab runtime is strongly recommended for acceptable training time.

---

## 4. Dependencies

Install the following Python packages before running the notebooks.

```bash
pip install imagehash Pillow tqdm scikit-learn matplotlib seaborn kagglehub kaggle
pip install torch torchvision optuna joblib
```

### Imported libraries across the two notebooks

- `os`
- `json`
- `math`
- `time`
- `random`
- `hashlib`
- `shutil`
- `warnings`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `imagehash`
- `PIL`
- `tqdm`
- `torch`
- `torchvision`
- `optuna`
- `joblib`
- `sklearn`

### External services / credentials

- **Kaggle dataset access**
  - Either `kagglehub.dataset_download(...)`
  - Or Kaggle API with `kaggle.json`
- **Google Drive**
  - Used to persist the processed dataset and training outputs

---

## 5. Dataset Source

The preprocessing notebook downloads this dataset:

- **Kaggle dataset:** `kamal01/top-agriculture-crop-disease`

Expected raw dataset root inside Colab after download:

```text
/content/agri_dataset/Crop Diseases
```

The dataset contains 17 classes across 5 crops:

- Corn
- Potato
- Rice
- Wheat
- Sugarcane

---

## 6. End-to-End Workflow

### Step 1: Run `eda-and-preprocessing-pipeline.ipynb`

This notebook performs both EDA and dataset preparation.

#### 6.1 Notebook setup behavior

The notebook:

- installs notebook-local dependencies
- downloads the Kaggle dataset
- mounts Google Drive
- defines preprocessing configuration
- scans all image folders into a master dataframe
- performs EDA visualizations and data-quality checks
- removes duplicates
- resizes images to `224x224`
- creates stratified `train / val / test` splits
- augments minority classes in the training set
- builds a final ImageFolder-style directory structure

#### 6.2 Key configuration block

The notebook uses the following preprocessing settings:

```python
DATASET_PATH = '/content/agri_dataset/Crop Diseases'
PROCESSED_PATH = '/content/processed'
DRIVE_SAVE_PATH = '/content/drive/MyDrive/CropDisease/processed'

TARGET_SIZE = (224, 224)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_STATE = 42

AUG_THRESHOLD = 600
AUG_TARGET = 700
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
```

#### 6.3 EDA and quality checks performed

The notebook verifies:

- class distribution
- crop-level distribution
- image width, height, aspect ratio, and channels
- sampled RGB intensity statistics
- corrupt files using `PIL.Image.verify()`
- exact duplicates using MD5 hashing
- near-duplicates using perceptual hash (`pHash`)

#### 6.4 Preprocessing logic

The notebook then applies:

1. Exact duplicate removal
2. Full-dataset near-duplicate removal using `pHash`
3. RGB conversion
4. Resize to `224x224` with `Image.LANCZOS`
5. Stratified split into train, validation, and test
6. Minority-class augmentation on the training split only
7. Copy into an `ImageFolder` directory structure

#### 6.5 Augmentation strategy

The training augmentation function includes:

- horizontal flip
- vertical flip
- rotation from `{0, 90, 180, 270}`
- brightness adjustment
- contrast adjustment
- saturation adjustment
- center zoom crop and resize back

Augmentation is applied only to classes with fewer than `600` training images, and those classes are expanded toward a target of `700` training images.

#### 6.6 Output structure expected by training

The final processed dataset must look like:

```text
processed/
  train/
    Class_A/
    Class_B/
    ...
  val/
    Class_A/
    Class_B/
    ...
  test/
    Class_A/
    Class_B/
    ...
```

This is required because the training notebook loads data using `torchvision.datasets.ImageFolder`.

#### 6.7 Important note about persistence

In the current notebook, the section that copies the processed dataset to Google Drive is commented out. Before relying on the output in a new environment, you should either:

- uncomment and use the Drive save block, or
- manually copy `/content/processed` to your chosen persistent location

The commented section also includes optional manifest generation for traceability.

---

### Step 2: Run `mobilenet.ipynb`

This notebook trains and evaluates the classifier on the processed dataset produced by Step 1.

#### 6.8 Training data path configuration

The current notebook defines:

```python
DATA_ROOT = '/content/drive/MyDrive/DSAI Lab/DASI Lab Project/dataset/processed'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

OUT_DIR = '/content/drive/MyDrive/CropDisease/outputs/mobilenet'
```

#### 6.9 Important replication warning

The preprocessing notebook and the MobileNet notebook currently use different Drive paths:

- preprocessing notebook save target:
  - `/content/drive/MyDrive/CropDisease/processed`
- MobileNet notebook load target:
  - `/content/drive/MyDrive/DSAI Lab/DASI Lab Project/dataset/processed`

Before running the training notebook, you must make these paths consistent. The easiest fix is to choose one Drive location and use it in both notebooks.

#### 6.10 Image transforms used for training

The notebook uses ImageNet normalization:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

Training transform:

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
```

Evaluation transform:

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
```

#### 6.11 Model options searched

The notebook runs Optuna-based hyperparameter optimization across:

- `mobilenet_v2`
- `mobilenet_v3_large`

Transfer-learning setup:

- pretrained ImageNet weights are loaded
- all backbone layers are frozen
- only the classifier head is trained
- dropout is tuned by Optuna
- final classifier output dimension is set to the number of disease classes

#### 6.12 Hyperparameters searched

Optuna searches:

- model architecture
- dropout
- learning rate
- weight decay
- optimizer: `Adam`, `AdamW`, `SGD`
- batch size: `16` or `32`

The notebook uses:

- `CrossEntropyLoss`
- validation Macro F1 as the optimization target
- `MedianPruner` for early pruning
- `10` epochs per trial
- `10` trials in the current configuration

#### 6.13 Evaluation and exported artifacts

After selecting the best trial, the notebook:

- reloads the best checkpoint
- evaluates on the test set
- saves summary metrics
- generates a confusion matrix
- generates class-wise F1 plots
- generates training and validation curves
- saves the best model in PyTorch formats

Expected output artifacts include:

- `all_trials_log.jsonl`
- `optuna_all_trials_results.csv`
- `best_metrics.json`
- `confusion_matrix.png`
- `classwise_f1.png`
- `metric_summary.png`
- `best_loss_curves.png`
- `best_macro_f1_curves.png`
- `all_trials_macro_f1_curves.png`
- `best_mobilenet_full_model.pth`
- `best_mobilenet_weights.pkl`

---

## 7. Reproduction Instructions

To replicate the same setup from scratch:

1. Open `notebooks/eda-and-preprocessing-pipeline.ipynb` in Google Colab.
2. Install dependencies when prompted by the notebook.
3. Authenticate Kaggle access using either `kagglehub` or `kaggle.json`.
4. Mount Google Drive.
5. Set `DATASET_PATH`, `PROCESSED_PATH`, and `DRIVE_SAVE_PATH`.
6. Run the notebook fully to generate the processed `train/val/test` dataset.
7. Confirm the processed dataset is saved in the same Drive location that the training notebook will read.
8. Open `notebooks/mobilenet.ipynb`.
9. Update `DATA_ROOT` and `OUT_DIR`.
10. Run the notebook fully to perform Optuna search, save the best checkpoint, and evaluate on the test split.

---

## 8. Important Implementation Notes

### 8.1 Colab-specific code

Both notebooks contain Google Colab specific sections such as:

- `from google.colab import drive`
- `drive.mount('/content/drive')`
- `from google.colab import files`
- `files.upload()`

If you want to run the pipeline outside Colab, these sections must be replaced with local filesystem paths and standard credential handling.

### 8.2 Class folder names must remain unchanged

Because the training notebook uses `ImageFolder`, the class labels are derived directly from folder names. Do not rename class directories unless you also update all dependent code and downstream app label mappings.

### 8.3 The preprocessing output is the handoff contract

The `mobilenet.ipynb` notebook does not perform raw-data cleaning. It assumes the EDA notebook has already produced a clean, resized, stratified dataset. If preprocessing is skipped or altered, model results will not be comparable.

### 8.4 Random seed and reproducibility

Both notebooks set the random seed to `42`. This improves reproducibility, but exact reproducibility can still vary across runtime type, PyTorch version, and hardware.

### 8.5 CPU versus GPU

The training notebook automatically selects:

```python
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

The pipeline works on CPU, but training and Optuna sweeps will take much longer.

### 8.6 Augmentation policy

Augmentation is intentionally limited to the training split. Do not augment validation or test data, or reported metrics will no longer reflect real model generalization.

### 8.7 Current notebooks mix experimentation and production steps

The EDA notebook includes:

- two Kaggle download approaches
- visualization-heavy analysis
- commented-out persistence code

For long-term maintainability, a future improvement would be to separate this into:

- a pure preprocessing notebook or script
- a training notebook
- a shared config file for paths and constants

---

## 9. Minimal Developer Checklist

- Confirm Kaggle access works
- Confirm Google Drive is mounted
- Confirm preprocessing and training notebooks use the same dataset path
- Confirm processed dataset contains `train`, `val`, and `test`
- Confirm class folder names match exactly across splits
- Confirm MobileNet notebook can read all splits via `ImageFolder`
- Confirm output artifacts are written to `OUT_DIR`

---

## 10. Summary

To reproduce the app's image-classification model, use the notebooks in this order:

1. `notebooks/eda-and-preprocessing-pipeline.ipynb`
2. `notebooks/mobilenet.ipynb`

The first notebook creates the cleaned and augmented dataset. The second notebook trains and evaluates the MobileNet-based classifier using transfer learning and Optuna tuning. The most important setup requirement is that both notebooks must point to the same processed dataset location.

---

# Part B. Final Application Developer Guide
## Multimodal Crop Disease Assistant

Clone the application hosted on Hugging Face Spaces: [crop-disease-assistant](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant)

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

- `/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/requirements.txt`
- `/home/harish/Harish/IITM/dsai-lab/Project/HF/crop-disease-assistant/space/requirements.txt`

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

If a different training artifact is used, update those values in `app.py` and optionally in `space/app.py`.

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

- In `app.py`:
  - `MODEL_PATH`
  - `MODEL_NAME`
  - `MODEL_DROPOUT`
- In `pipeline/generator.py`:
  - `GROQ_MODEL`
- In `rag/ingest.py` and `rag/retriever.py`:
  - `EMBED_MODEL`
  - Chroma collection name and storage path
- In `voice/asr.py`:
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

---

### Team Member Signatures

The following team members have reviwed and approved the contents of report/file:
- [x] Harish Sahadev M
- [x] Sai Naman
- [ ] Ayushi Dixit
- [x] Allanki Saketh Kumar
- [ ] Manas Rastogi
