# Contributions (Milestone 3):

## Harish Sahadev M
### 21f1005856
- Authored [milestone report](/milestone-5/milestone-5-report.md) covering model evaluation setup, performance metrics, quantitative results, visualizations, qualitative analysis, error analysis, and key insights on model limitations and improvements.
- Ran MobileNet V2 and V3 Large Optuna hyperparameter search (6 hyperparameters — architecture, optimizer, learning rate, dropout, weight decay, batch size) on Google Colab T4 GPU 
- Evaluated trial results, identified best configuration (MobileNet V3 Large + AdamW, Trial #4) achieving 92.1% test accuracy and 90.0% macro F1
- Designed and implemented the RAG pipeline — document ingestion using PyMuPDF, section-aware chunking, multilingual embedding with `intfloat/multilingual-e5-large`, and two-stage metadata-filtered retrieval using ChromaDB
- Integrated Groq API (Llama 3.3 70B) with verified-source grounding and multilingual prompt design
- Implemented voice interface — Whisper ASR for speech-to-text with automatic Indian language detection, gTTS for text-to-speech in 6 Indian languages
- Built and deployed Gradio application on Hugging Face Spaces


## Sai Naman
### 22f3000350
- Created the complete model evaluation notebook, including test data pipeline, preprocessing, and implementation of all evaluation metrics.
- Performed quantitative analysis by comparing model performance across metrics (Macro F1, accuracy, loss) and different Optuna trials/architectures.
- Conducted qualitative analysis by examining sample model outputs, including both correct predictions and failure cases.
- Performed error analysis, identifying key patterns in misclassifications (e.g., rice disease confusion, data scarcity, real-world noise) and analysing their causes.
- Analysed and documented key observations, limitations, and anomalies in model performance, and created the Milestone 5 PPT presenting all results and insights.
