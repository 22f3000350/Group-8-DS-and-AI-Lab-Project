# Multimodal AI Assistant for Smart Agriculture

`Group-8-DS-and-AI-Lab-Project`

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant)
[![Gradio](https://img.shields.io/badge/Gradio-App-F97316?logo=gradio&logoColor=fff)](crop-disease-assistant/README.md)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=fff)](crop-disease-assistant/requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=fff)](models/model_details.md)

This project builds a multimodal AI assistant for crop disease support in Indian farming contexts. It combines crop leaf image classification, retrieval-based agricultural guidance, and multilingual interaction through text and voice.

The system is designed around five major crops: corn, potato, rice, wheat, and sugarcane. A farmer can upload a leaf image, ask a question, and receive grounded guidance based on the detected crop disease.

## What This Project Includes

- A crop disease classification pipeline for leaf images
- A RAG-based advisory system using curated agriculture documents
- Multilingual support for text and voice interaction
- A Gradio app for local use and Hugging Face deployment
- Training notebooks, reports, and milestone documentation

## Core Stack

[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG%20Store-6E44FF)](crop-disease-assistant/rag/)
[![Whisper](https://img.shields.io/badge/Whisper-Speech%20to%20Text-111827)](crop-disease-assistant/voice/asr.py)
[![gTTS](https://img.shields.io/badge/gTTS-Text%20to%20Speech-34A853?logo=google&logoColor=fff)](crop-disease-assistant/voice/tts.py)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Generation-000000)](crop-disease-assistant/pipeline/generator.py)

Main tools and dependencies used in this project include Python, Gradio, PyTorch, ChromaDB, OpenAI Whisper, gTTS, and Groq-powered response generation.

## Project Structure

```text
.
├── crop-disease-assistant/   # Main application
├── data/                     # Dataset details and references
├── models/                   # Model summary and results
├── notebooks/                # Training and experimentation notebooks
└── docs/                     # Milestone reports and project documents
```

## Quick Start

Run the demo app locally:

Clone the Hugging Face Space repository:

```bash
git clone https://huggingface.co/spaces/harishsahadev/crop-disease-assistant
cd crop-disease-assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open the Gradio link shown in the terminal after the app starts.

For a more detailed setup and usage guide, see [crop-disease-assistant/README.md](crop-disease-assistant/README.md).

Before running the full assistant, make sure these files are available:

- `model/mobilenet.pth` for disease prediction
- `rag/chroma_db/` for document retrieval
- `GROQ_API_KEY` set in your environment for response generation

Example:

```bash
export GROQ_API_KEY=your_api_key_here
```

## Architecture

The assistant follows a multimodal pipeline: leaf image input goes through disease detection, user text or speech is processed into a query, relevant agricultural documents are retrieved, and a grounded response is generated back in text or audio.

![Project Architecture](docs/milestone-3/architecture.png)

## Model and Dataset

- Dataset details: [data/dataset_details.md](data/dataset_details.md)
- Model summary: [models/model_details.md](models/model_details.md)

## App and Reports

- Try live demo: [Hugging Face Space](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant)
- App guide: [crop-disease-assistant/README.md](crop-disease-assistant/README.md)
- Final report: [docs/milestone-6/final_project_report.md](docs/milestone-6/final_project_report.md)
- User guide: [docs/milestone-6/user_guide.md](docs/milestone-6/user_guide.md)
