# Final Contribution Summary

## Harish Sahadev M  
### Roll No: 21f1005856

I contributed across all stages of the project, covering **research, system design, model development, integration, deployment, and documentation**, while also **coordinating team progress** and ensuring timely milestone completion.

For **Milestone 1**, I structured and drafted the report, including **problem definition, objectives, and evaluation strategy**. I conducted a comprehensive **literature review** (computer vision, RAG, multilingual NLP, speech systems), defined the **disease scope and benchmarks**, and ensured overall **clarity, coherence, and academic formatting**.

For **Milestone 2**, I reviewed dataset preparation requirements, explored **EDA and preprocessing strategies**, and contributed to understanding dataset structuring for the modeling phase.

For **Milestone 3**, I authored the report, designed the **model architecture and data flow diagrams**, and validated the **ResNet-based pipeline setup**.

For **Milestone 4**, I performed **hyperparameter tuning (Optuna)** on MobileNet models and identified the best configuration achieving **92.1% accuracy and 90.0% Macro F1**. I built the **inference pipeline** and led system development, including:
- **RAG pipeline** (document processing, chunking, multilingual embeddings, ChromaDB retrieval)  
- **LLM integration** (Llama 3.3 70B via Groq with grounded responses)  
- **Voice pipeline** (Whisper ASR + gTTS for multilingual interaction)  

I also handled **deployment** by building and hosting the Gradio application on **Hugging Face Spaces**, and prepared the **training report and presentation**.

For **Milestone 5**, I designed the **evaluation pipeline**, conducted **quantitative, qualitative, and error analysis**, and documented key insights, limitations, and improvements. I also refined the **RAG, ASR, and deployment components**.

For **Milestone 6**, I led **final deployment and system integration**, prepared the **final presentation (PPT)**, refined and updated the **developer guide**, and reviewed all reports. I ensured **system stability, documentation quality, and submission readiness**, while coordinating the team throughout.

Overall, I served as:
- **Technical Lead** — model training, RAG pipeline, ASR/TTS, deployment  
- **Documentation Lead** — reports, evaluation, developer guide  
- **System Integrator** — end-to-end pipeline (Vision + RAG + LLM + Speech)  
- **Team Coordinator** — ensured milestone progress and completion  

---

# Sai Naman  
### Roll No: 22f3000350  

For the project, I contributed across **model development, training, experimentation, and documentation**. My key contributions include:

- Researched and analyzed existing agri-AI startups (e.g., KissanAI, Agent Crop, Farmonaut) to identify gaps such as lack of multimodal integration and limited conversational capabilities  
- Prepared structured presentations explaining dataset selection, EDA findings, and preprocessing pipeline  
- Implemented image classification models using PyTorch (**ResNet, EfficientNet, MobileNet**) with transfer learning  
- Integrated dataset with PyTorch DataLoader and applied preprocessing (224×224 resizing, ImageNet normalization, augmentation)  
- Designed and built end-to-end training pipelines from preprocessing to prediction  
- Performed experimentation across multiple architectures to compare performance  
- Conducted hyperparameter tuning (learning rate, optimizer: Adam/AdamW/SGD, batch size)  
- Implemented evaluation metrics (**Accuracy, Macro F1, Micro F1**) for robust performance analysis  
- Developed inference pipelines for generating predictions on new data  
- Integrated preprocessing and training into a reproducible workflow  
- Conducted quantitative analysis across models and configurations  
- Performed error analysis (e.g., class imbalance, rice disease confusion)  
- Documented model behavior, observations, and limitations  
- Contributed to report writing and prepared presentations on training and results  

---

# Manas Rastogi  
### Roll No: 22f3001477  

For the project, I contributed across project planning, environment setup, data pipeline development, model experimentation, and end-user documentation. My key contributions include:

- Led the initial project topic selection and helped finalize the technical direction of the system, scoping it toward a multimodal AI assistant for smart agriculture.  
- Researched and curated key references, candidate datasets, and supporting content used in the Milestone 1 report and presentation deck.  
- Performed early feasibility analysis of datasets, candidate model families, and system architecture options to guide project planning decisions.  
- Oversaw the final review and consolidation of the Milestone 1 report to ensure consistency and completeness across sections.  
- Set up the Google Colab environment and integrated the Kaggle API to establish a reliable dataset download pipeline for the team.  
- Assisted in implementing the image scanning and master dataframe building logic across all 17 disease class folders.  
- Contributed to writing the class distribution and crop-level visualization cells in the EDA notebook.  
- Helped implement the MD5 hashing and perceptual hash (pHash) logic used for exact and near-duplicate detection during data quality checks.  
- Collaborated on debugging and testing the augmentation function across multiple minority classes to ensure balanced training data.  
- Assisted in verifying the stratified 80/10/10 train/validation/test folder structure and the cross-split leakage check to prevent data contamination.  
- Researched and experimented with multiple architecture families — including ResNet and EfficientNet (nano and small variants) — during the model selection phase.  
- Contributed to early design discussions on the end-to-end application architecture, covering the integration of vision, RAG, and voice modules.  
- Participated in team reviews of training experiments, evaluation outcomes, and system integration decisions, supporting intermediate validation checkpoints.  
- Authored the UserGuide for the deployed Crop Disease AI Assistant, translating the technical system into clear, accessible instructions for non-technical end users such as farmers and field officers.  
- Ensured the end-user documentation aligned with the application's actual interface, feature set, and safety guardrails, including confidence thresholds and KVK referral guidance.  
- Contributed to report writing and documentation across milestones, supporting clear communication of the project's progress, methodology, and final deliverables.  

---

## Ayushi Dixit
### 22f3000082

---

## Allanki Saketh Kumar
### 21f3002277

---

### Team Member Signatures

The following team members have reviwed and approved the contents of report/file:
- [x] Harish Sahadev M
- [ ] Sai Naman
- [ ] Ayushi Dixit
- [ ] Allanki Saketh Kumar
- [ ] Manas Rastogi