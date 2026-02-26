# Milestone 1 Report

# 1. Problem Definition

## 1.1 Background and Key Issues

Agriculture remains a primary livelihood source for a significant portion of India’s population, particularly small and medium-scale farmers. Timely detection of crop diseases is critical to preventing yield loss and economic damage. Leaf-based diseases such as blight, rust, mildew, and smut can spread rapidly if not identified at early stages.

Currently, farmers often rely on manual visual inspection or delayed consultation with agricultural experts. This process is time-consuming, subjective, and prone to inconsistencies. Although digital agricultural platforms exist, they present several limitations:

- Many are English-centric and text-heavy
- Image-based models often fail in real-world field conditions
- AI advisory systems may generate unverified or hallucinated recommendations
- Most systems lack multilingual voice interaction
- Cloud-heavy solutions are unsuitable for low-connectivity rural environments

These challenges highlight the need for an integrated, accessible, and reliable AI-driven system tailored to Indian farming conditions.

---

## 1.2 Problem Statement

There is currently no unified AI system that integrates:

- Real-time crop disease detection from leaf images
- Multilingual conversational support
- Voice-based interaction
- Retrieval-grounded agricultural recommendations
- Optimization for low-bandwidth rural settings

This project aims to develop a **Multimodal AI Assistant for Smart Agriculture** that combines computer vision, generative AI, and speech technologies to provide context-aware and reliable support to farmers.

## 1.3 Objectives

### Primary Objective

To design and develop a deep learning-based multimodal system that automatically detects crop diseases from leaf images and provides grounded, multilingual, and voice-enabled agricultural guidance.

### Secondary Objectives

1. Achieve competitive classification performance (target accuracy ≥ 85–90% and strong F1-score).
2. Support multiple crops such as rice, wheat, sugarcane, potato, and corn.
3. Optionally classify disease severity (mild, moderate, severe).
4. Implement Retrieval-Augmented Generation (RAG) to reduce hallucination.
5. Enable multilingual support (Hindi, Bengali, Tamil, Telugu, Malayalam, Kannada).
6. Integrate speech-to-text (ASR) and text-to-speech (TTS).
7. Design a lightweight architecture optimized for rural deployment.

# **2. Literature Review**

This project combines research from four major domains:

1. Crop disease detection using deep learning

2. Disease severity estimation

3. Retrieval-Augmented Generation (RAG)

4. Multilingual and speech-based AI systems


### **2.1 Deep Learning for Crop Disease Detection**

Early plant disease detection relied on handcrafted features such as
color histograms and texture descriptors. With the advancement of deep
learning, Convolutional Neural Networks (CNNs) have become the dominant
approach. The PlantVillage Dataset enabled large-scale supervised
benchmarking for leaf classification tasks[[1]](#1).

Architectures such as ResNet[[2]](#2), MobileNet[[3]](#3), and EfficientNet[[4]](#4)
reported accuracies above 95% under controlled conditions. However,
studies show performance drops in real-world field conditions due to
lighting variation and background clutter[[5]](#5). Recent research focuses
on transfer learning and data augmentation for robustness[[6]](#6). While
CNN-based transfer learning forms a strong baseline, generalization
remains a key challenge.

### **2.2 Disease Severity Estimation**

While prior research explores severity estimation using multi-class classification and segmentation-based lesion quantification[[7]](#7)[[8]](#8), most public datasets provide limited fine-grained severity annotations.

In our project, we will implement severity estimation as a multi-class classification extension, categorizing each detected disease into mild, moderate, or severe stages based on visual lesion spread. Instead of performing full pixel-level segmentation (which requires dense annotation), we will leverage:

- Data augmentation to simulate progression patterns

- Class re-weighting to handle imbalance

- Focal loss to improve minority class detection

We will evaluate severity prediction using macro-F1 score to ensure balanced performance across stages.

This allows us to provide early-intervention guidance, rather than only binary disease presence detection.

### **2.3 Retrieval-Augmented Generation (RAG) for Grounded AI**

Instead of using a direct LLM response (which may hallucinate pesticide recommendations), we will implement a Retrieval-Augmented Generation pipeline based on the architecture proposed in Lewis et al.[[9]](#9).

Specifically, we will:

1. Curate verified agricultural documents (ICAR guidelines, state advisories).

2. Store these in a vector database using sentence embeddings.

3. Retrieve top-k relevant passages based on:
    - Detected disease
    - Crop type
    - Optional region and season input

4. Pass retrieved content to the generator model to produce grounded responses.

This ensures that when a farmer asks:

“How do I treat early blight in potato?”

The system:
- Detects “Early Blight”
- Retrieves verified treatment guidelines
- Generates response grounded only in those documents

We will compare hallucination rate between:
- Direct LLM response
- RAG-grounded response

### **2.4 Multilingual Language Models**

Rather than relying on general multilingual capability, we will fine-tune multilingual transformer models (such as mT5[[11]](#11) or IndicBERT[[12]](#12)) on agriculture-specific corpora.

We plan to:
- Use crop advisory documents translated into Hindi, Tamil, Bengali, etc.
- Fine-tune on domain-specific QA pairs.
- Handle code-mixed inputs using transliteration preprocessing.

This ensures the chatbot is not only multilingual but agriculture-domain aware.

### **2.5 Speech Recognition and Synthesis**

We will integrate Whisper[[14]](#14) for speech-to-text conversion and a transformer-based TTS model for speech synthesis.

To handle rural constraints:
- We will use smaller Whisper variants (base or small model).
- We will measure Word Error Rate (WER) on agricultural vocabulary.
- We will evaluate latency under simulated low-bandwidth settings.

Noise robustness will be tested by augmenting input audio with background noise samples.

### **2.6 Key Research Insights**

Existing research establishes CNN-based transfer learning as an effective baseline but highlights generalization challenges in field conditions. Severity estimation remains underexplored. RAG improves factual grounding in domain-specific AI systems, while multilingual and speech models require domain adaptation and deployment optimization. These insights directly guide the design of the proposed multimodal agricultural assistant.

# 3. Existing Solutions and Comparison with Proposed Approach

| Solution | What They Do | Limitations | Comparison with Proposed System | Key Differences |
| --- | --- | --- | --- | --- |
| **Plantix** ([Plantix](https://www.plantix.net/)) | AI-based leaf image disease detection with treatment suggestions via mobile app | Primarily classification-focused; limited conversational depth; no clear retrieval grounding; limited voice integration | Strong on vision-based diagnosis but lacks multimodal conversational integration | Proposed system integrates vision + RAG + multilingual chatbot + voice interaction |
| **KissanAI** ([KissanAI](https://kissan.ai/)) | Voice-first multilingual agricultural chatbot | No integrated image-based disease detection; advisory-focused | Strong on conversational AI but lacks vision component | Proposed system combines real-time disease detection with grounded advisory |
| **Agent Crop** ([AgentCrop](https://agentcrop.com/)) | AI-based crop disease detection using smartphone images | No multilingual chatbot; no voice interface; limited contextual reasoning | Similar vision component but lacks multimodal integration | Proposed system adds contextual RAG-based responses and speech support |
| **AgroStar** ([AgroStar](https://corporate.agrostar.in/)) | Digital advisory and agri-input marketplace | Often human-assisted; limited AI-driven disease detection | Advisory platform but not fully automated multimodal AI | Proposed system provides automated diagnosis + contextual AI advisory |
| **Farmonaut** ([Farmonaut](https://farmonaut.com/)) | Satellite-based crop monitoring and predictive analytics | Not leaf-level; not conversational; no voice interaction | Operates at macro monitoring level | Proposed system focuses on farmer-level leaf diagnosis and interaction |

---

Across the current agritech landscape, individual components such as vision-based disease detection, conversational AI platforms, and precision agriculture analytics are already available. However, these solutions operate largely in isolation. There is no existing system that unifies leaf-level disease detection with multilingual conversational AI, retrieval-grounded generative recommendations, voice interaction, and lightweight rural deployment into a single end-to-end multimodal architecture. This integration gap presents a clear opportunity for the proposed system.

# 4. Identify Gaps and Opportunities

## 4.1 Identified Gaps and Limitations in Existing Solutions

### 1. Fragmented Multimodal Integration

Existing agritech solutions typically focus on either image-based disease detection (e.g., classification models) or conversational advisory systems. These systems operate independently and do not exchange structured outputs. For example, disease detection systems often output only a label, without integrating that output into a contextual advisory engine.

This lack of integration prevents automated end-to-end pipelines where image prediction directly informs multilingual, grounded recommendations.

### 2. Limited Real-World Robustness

Although CNN models achieve 90–99% accuracy on controlled datasets such as PlantVillage, studies show performance degradation of 10–25% under real-world field conditions due to:
- Lighting variability
- Background clutter
- Partial occlusion
-Low-resolution camera inputs

Most deployed systems do not explicitly evaluate robustness under such perturbations, limiting their practical reliability.

### 3. Limited Disease-Specific Context Modeling

We focus on four diseases:

- Rice Leaf Blight
- Wheat Rust
- Potato Early Blight
- Corn Leaf Spot

These diseases often exhibit visually overlapping symptoms in early stages (yellowing, spotting, mild necrosis), making them difficult to differentiate with the naked eye. Misclassification may lead to incorrect pesticide usage or delayed intervention.

Existing systems rarely incorporate severity staging or structured disease metadata into advisory generation.

### 4. Absence of Retrieval-Grounded Advisory
Most AI-based agricultural assistants rely on direct LLM outputs or rule-based systems. Without retrieval grounding, generative responses may hallucinate pesticide dosages or treatment steps. In agriculture, such hallucinations can cause economic loss or crop damage.

Few systems publicly document their grounding or validation pipeline.

### 5. Incomplete Multilingual and Voice Integration

Although multilingual chatbots exist, seamless integration of Image-based disease detection, Regional language generation, Real-time speech-to-text and text-to-speech into a single pipeline remains limited. Additionally, agricultural vocabulary in regional languages is often domain-specific and code-mixed.

### 6. Rural Deployment Constraints

Many AI systems depend on high-bandwidth cloud inference and large model sizes. Rural environments often experience:

- Intermittent connectivity
- Limited device capability
- Background environmental noise

Systems not optimized for these constraints may fail in practical use.

## 4.2 Opportunities for Improvement

The above gaps reveal clear technical opportunities.

1. Develop a unified multimodal architecture where disease predictions are structured outputs that directly inform advisory generation.

2. Improve real-world robustness by simulating field conditions using augmentation techniques such as brightness shifts (±40%), blur, random occlusion, and background variation.

3. Introduce severity-level prediction (mild, moderate, severe) to enable early-intervention recommendations rather than binary classification.

4. Implement a Retrieval-Augmented Generation pipeline using verified agricultural documents (ICAR/state advisories) stored in a vector database to ensure grounded responses.

5. Fine-tune multilingual transformer models on agriculture-specific corpora to handle regional and code-mixed queries.

6. Optimize deployment using lightweight CNN architectures and smaller ASR variants to ensure usability in low-bandwidth rural settings.

## 4.3 How the Proposed Project Addresses These Gaps

To address these limitations, we propose a structured end-to-end multimodal pipeline.

1. Robust Vision Module
We will train MobileNetV2 and ResNet50 using transfer learning on selected crop datasets. To improve robustness, we will apply controlled augmentation simulating real-world lighting, blur, and occlusion. Performance degradation under perturbation will be explicitly measured.

2. Structured Disease Representation
Instead of outputting only a class label, the model will generate structured outputs including:
    - Crop type
    - Disease name
    - Severity level

This structured metadata will be passed to the advisory module.

3. Retrieval-Augmented Advisory Engine
We will build a vector database of verified agricultural guidelines. For each query, the system will retrieve top-k relevant passages based on disease, crop, and region before generation. This ensures grounded, context-aware responses and reduces hallucination.

4. Multilingual and Speech Integration
We will fine-tune multilingual models on agriculture-specific data and integrate Whisper (base/small variant) for speech recognition. We will measure Word Error Rate (WER) under simulated rural noise conditions.

5. Deployment Optimization
The architecture will be modular, allowing lightweight inference for vision and speech components. Latency will be measured to maintain practical usability (below 3–4 seconds total response time).

# 5. Baseline Metrics and Evaluation Strategy

## 5.1 Vision Baseline  

For the vision component, we will focus on four major diseases:

- Rice Leaf Blight  
- Wheat Rust  
- Potato Early Blight  
- Corn Leaf Spot  

These diseases often exhibit visually similar symptoms such as yellowing, spotting, or necrotic lesions, particularly in early stages, making them difficult to reliably differentiate with the naked eye under field conditions. Early misidentification can lead to incorrect pesticide usage.

### Research Benchmarks  

On the PlantVillage dataset:

- ResNet50 achieved ~96–99% accuracy (He et al., 2016)  
- MobileNetV2 achieved ~92–96% accuracy (Howard et al., 2017)  
- EfficientNet variants reported ~95–98% accuracy (Tan & Le, 2019)  

However, Barbedo (2018) demonstrated that real-field performance drops by approximately 10–25% due to lighting variation, background clutter, and image noise.

### Our Baseline Definition  

We will implement transfer learning-based CNN models (MobileNetV2 and ResNet50) trained on a combined dataset of selected crop diseases.

Baseline expectation:

- Controlled validation accuracy: ~90–95%  
- Real-world robustness target: ≥85% F1-score  

**Evaluation metrics** includes accuracy, precision, recall, macro & weighted F1-score and onfusion Matrix  

We will simulate field conditions using:
- Brightness variation (±40%)  
- Gaussian blur  
- Random cropping  
- Background noise  

Performance degradation under these perturbations will be measured to quantify robustness improvement over standard baselines.

## 5.2 RAG Evaluation  

For the advisory module, the baseline will be a direct LLM response without retrieval grounding. Literature shows that non-grounded LLMs can generate hallucinated domain-specific recommendations, especially in specialized fields such as agriculture.

### Baseline Comparison  

We will compare a direct LLM response (without retrieval grounding) against a Retrieval-Augmented Generation (RAG) pipeline. In the RAG setup, verified agricultural documents such as ICAR and state advisory guidelines will be embedded and stored in a vector database. For each user query, the system will retrieve the top-k most relevant passages based on crop type, detected disease, severity level, and optional region input, and inject this retrieved context into the generation process to produce grounded and context-aware responses.

The evaluation will measure factual consistency through manual expert assessment, hallucination rate defined as the percentage of unsupported or fabricated claims, and response relevance scored on a 1–5 scale. Our target is to reduce hallucination rate by at least 50% compared to the direct LLM baseline while significantly improving factual consistency and contextual relevance of recommendations.

## 5.3 Speech Evaluation  

For speech interaction, we will use Whisper (base or small variant) as the baseline automatic speech recognition (ASR) model. Reported benchmarks indicate that Whisper-small achieves approximately 5–10% Word Error Rate (WER) on clean speech datasets, though performance typically degrades under noisy or real-world conditions. Since rural environments often involve background noise and informal speech patterns, we will explicitly evaluate robustness under such constraints.

The speech module will be evaluated using Word Error Rate (WER) and end-to-end latency measured in seconds. Testing will include simulated background noise and code-mixed agricultural queries to reflect realistic usage scenarios. Our target is to maintain WER ≤15% for agriculture-specific vocabulary while ensuring total response latency remains below 3–4 seconds to support practical real-time interaction.

## 5.4 End-to-End Evaluation  

The complete multimodal system will be evaluated as an integrated pipeline rather than independent modules.

Evaluation criteria:

- Total response time (image + query → final voice output)  
- Contextual accuracy of generated recommendations  
- Severity-stage correctness  
- User-level usability testing (structured feedback form)  

We will compare:
- Vision-only system  
- Vision + direct LLM  
- Full multimodal (Vision + RAG + Multilingual + Speech)  

The goal is to demonstrate measurable improvements in **reliability, robustness, context-awareness and accessibility**.

---
## References

<a id="1">[1]</a> : Hughes, D. & Salathé, M. (2015). PlantVillage Dataset.
    arXiv:1511.08060.

<a id="2">[2]</a> : He, K. et al. (2016). Deep Residual Learning for Image
    Recognition. arXiv:1512.03385.

<a id="3">[3]</a> : Howard, A. et al. (2017). MobileNets: Efficient Convolutional
    Neural Networks for Mobile Vision Applications. arXiv:1704.04861.

<a id="4">[4]</a> : Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks. arXiv:1905.11946.

<a id="5">[5]</a> : Barbedo, J. (2018). Impact of dataset variability on deep learning
    for plant disease detection. Computers and Electronics in
    Agriculture.

<a id="6">[6]</a> : Too, E. et al. (2019). A comparative study of fine-tuning deep
    learning models for plant disease identification. Agronomy.

<a id="7">[7]</a> : Ronneberger, O. et al. (2015). U-Net: Convolutional Networks for
    Biomedical Image Segmentation. arXiv:1505.04597.

<a id="8">[8]</a> : Mohanty, S. et al. (2016). Using Deep Learning for Image-Based
    Plant Disease Detection. arXiv:1604.03169.

<a id="9">[9]</a> : Lewis, P. et al. (2020). Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks. arXiv:2005.11401.

<a id="10">[10]</a> : Gao, Y. et al. (2023). Retrieval-Augmented Generation for Large
    Language Models: A Survey. arXiv:2312.10997.

<a id="11">[11]</a> : Xue, L. et al. (2021). mT5: A Massively Multilingual Pre-trained
    Text-to-Text Transformer. arXiv:2010.11934.

<a id="12">[12]</a> : Kakwani, D. et al. (2020). IndicBERT: A Multilingual ALBERT Model
    for Indian Languages. arXiv:2009.00318.

<a id="14">[14]</a> : Radford, A. et al. (2022). Whisper: Robust Speech Recognition via
    Large-Scale Weak Supervision. arXiv:2212.04356.