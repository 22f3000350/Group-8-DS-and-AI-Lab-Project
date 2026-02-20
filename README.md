# Multimodal AI Assistant for Smart Agriculture

`Group-8-DS-and-AI-Lab-Project`

## Problem Statement

[DL; Vision; GenAI; Speech]

Timely detection of crop diseases and access to reliable agricultural guidance remain critical challenges for farmers in India. Small and medium-scale farmers often rely on local Indian languages and voice-based communication, while many existing agricultural applications are English-centric, text-heavy, or require expert interpretation. In addition, rural regions often face connectivity constraints, making high-compute, fully online solutions impractical.

This project proposes the development of a **Multimodal AI Assistant for Smart Agriculture**, tailored to Indian farming conditions. The system will focus on **major harvested crops such as rice, wheat, sugarcane, potato, and corn**, targeting commonly occurring leaf diseases including **leaf blight, rust, early/late blight (potato), smut (sugarcane), and mildew-type infections**. The vision module will primarily process **leaf-based images**, while being designed to handle real-world variability such as different lighting conditions, varied camera quality, cluttered backgrounds, and partially damaged leaves.

The proposed system will include:
- A **deep learning-based crop disease detection model** trained on publicly available crop disease datasets, evaluated using accuracy and F1-score.

- Optional **disease severity classification** (mild, moderate, severe) to assist early intervention.

- A **multilingual chatbot supporting Hindi, Bengali, Tamil, Telugu, Malayalam, and Kannada** to provide agriculture-related guidance.

- Integrated **voice and text interaction**, using speech-to-text (ASR) and text-to-speech (TTS) for real-time communication in supported languages.

- A **retrieval-augmented generation (RAG)** pipeline to ensure that recommendations are grounded in verified agricultural advisories and reliable knowledge sources.

- A lightweight, modular architecture optimized for **low-bandwidth rural environments**.

## System Workflow
When a farmer uploads a leaf image and submits a query (via text or voice):

1. The image is processed by the disease detection model.

2. The predicted disease (and severity level) is passed as structured input to the conversational module.

3. Relevant guidance is retrieved from curated agricultural resources.

4. A grounded response is generated in the selected Indian language.

5. If voice mode is used, the system performs real-time speech recognition and speech synthesis.


## Context-Aware Recommendations
Recommendations will consider:
- Crop type
- Detected disease
- Season
- Optional region input for location-specific advisories


## Key Differentiation
The proposed system uniquely integrates:
- Real-time disease detection for major Indian crops
- Multilingual conversational support in local Indian languages
- Grounded, retrieval-based agricultural recommendations
- Voice-enabled interaction suitable for rural users
- Optimization for low-connectivity environments


By combining computer vision, generative AI, and multilingual speech interaction, the system aims to deliver accessible, reliable, and context-aware agricultural assistance tailored to Indian farmers.
