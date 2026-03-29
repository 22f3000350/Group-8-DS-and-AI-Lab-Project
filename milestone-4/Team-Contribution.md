# Contributions (Milestone 3):

## Harish Sahadev M
### 21f1005856
**Model Training**
- Ran MobileNet V2 and V3 Large Optuna hyperparameter search (6 hyperparameters — architecture, optimizer, learning rate, dropout, weight decay, batch size) on Google Colab T4 GPU 
- Evaluated trial results, identified best configuration (MobileNet V3 Large + AdamW, Trial #4) achieving 92.1% test accuracy and 90.0% macro F1

**System Integration & Deployment**
- Built end-to-end inference module with confidence thresholding and class label parsing
- Designed and implemented the RAG pipeline — document ingestion using PyMuPDF, section-aware chunking, multilingual embedding with `intfloat/multilingual-e5-large`, and two-stage metadata-filtered retrieval using ChromaDB
- Integrated Groq API (Llama 3.3 70B) with verified-source grounding and multilingual prompt design
- Implemented voice interface — Whisper ASR for speech-to-text with automatic Indian language detection, gTTS for text-to-speech in 6 Indian languages
- Built and deployed [Gradio application](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant) on Hugging Face Spaces (zero cost, public URL)

**Documentation & Presentation**
- Authored [milestone report](/milestone-4/milestone-4-report.md) and [mobilenet training report](/milestone-4/milestone-4-mobilenet--training-report.md) covering dataset documentation, preprocessing rationale, model architecture, training configuration, hyperparameter experiments, and results analysis
- Prepared [presentation deck](/milestone-4/crop_disease_milestone_4_ppt.pdf) covering end-to-end system design, architecture diagram, training results, RAG pipeline explanation, deployment stack, and tech stack justifications


## Sai Naman
### 22f3000350
- Implemented training pipelines for [ResNet](/milestone-4/notebooks/resnet.ipynb), [EfficientNet](/milestone-4/notebooks/efficientnet.ipynb), and [MobileNet](/milestone-4/notebooks/mobilenet.ipynb) models using transfer learning.
- Performed hyperparameter tuning by varying learning rate, optimizer, and batch size.
- Integrated preprocessing pipeline with model training and ensured smooth end-to-end execution.
- Implemented evaluation using Macro F1, Micro F1, and Accuracy for performance analysis.
- Developed an inference pipeline to load trained models and generate predictions on new data.


## Allanki Saketh Kumar
### 21f3002277
- I contributed to the RAG module by collecting and curating official agricultural documents from trusted sources such as ICAR, FAO, and IRRI. I processed these documents by extracting, cleaning, and structuring disease-related information into chunked text suitable for embedding and retrieval. This ensured that the system generates accurate and grounded responses based on verified agricultural knowledge.

## Ayushi Dixit
### 22f3000082
- Did not contribute


## Manas Rastogi
### 22f3001477 
- Did not contribute
