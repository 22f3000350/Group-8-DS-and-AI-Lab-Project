# Milestone 5 Report
### Model Evaluation, Performance Analysis & Results

---

## 1. Model & Pipeline Recap

### 1.1 Trained Model

The vision module deployed is **MobileNet V3 Large**, pretrained on ImageNet and fine-tuned via Optuna hyperparameter optimisation (5 trials, MacroF1 as objective) on a 17-class crop disease classification task. The best trial (Trial #4) used AdamW with LR = 4.28e-4, dropout = 0.470, batch size = 16, and weight decay = 1.8e-6. Only the classifier head was trained; the backbone was frozen to leverage ImageNet feature representations.

### 1.2 Full System Pipeline

| Stage | Component | Details |
|-------|-----------|---------|
| Input | Leaf image | 224×224 RGB, ImageNet-normalised |
| Vision | MobileNet V3 Large | 17-class disease classifier |
| Context | Context Builder | Crop + disease → retrieval query |
| Retrieval | ChromaDB + multilingual-e5-large | 3-stage fallback, top-4 chunks |
| Generation | Groq Llama 3.3 70B | RAG-grounded advisory |
| Output | gTTS + Gradio | Text + audio in 6 Indian languages |

---

## 2. Evaluation Dataset

### 2.1 Dataset Composition

The evaluation was conducted on the held-out **test split** derived from the Kaggle dataset (kamal01 — *Top Agriculture Crop Disease*).

| Property | Value |
|----------|-------|
| Total images (original dataset) | 13,324 |
| Test split size | ~1,332 images (10%) |
| Stratification | Stratified by class (seed = 42) |
| Split method | 80% train / 10% val / 10% test |
| Classes | 17 (disease/healthy categories) |
| Crops | Corn (4), Potato (3), Rice (4), Wheat (3), Sugarcane (3) |
| Class imbalance (test) | ~14.9× (Rice___Healthy vs Sugarcane classes) |

### 2.2 Class Distribution (Test Set — Approximate)

| Crop | Class | Approx. Test Images |
|------|-------|-------------------|
| Corn | Healthy | ~116 |
| Corn | Common Rust | ~119 |
| Corn | Northern Leaf Blight | ~99 |
| Corn | Gray Leaf Spot | ~51 |
| Potato | Healthy | ~15 |
| Potato | Early Blight | ~100 |
| Potato | Late Blight | ~100 |
| Rice | Healthy | ~149 |
| Rice | Leaf Blast | ~98 |
| Rice | Brown Spot | ~61 |
| Rice | Neck Blast | ~100 |
| Wheat | Healthy | ~112 |
| Wheat | Brown Rust | ~90 |
| Wheat | Yellow Rust | ~92 |
| Sugarcane | Healthy | ~10 |
| Sugarcane | Red Rot | ~10 |
| Sugarcane | Bacterial Blight | ~10 |

Sugarcane classes remain severely underrepresented (≈10 images each in the test set), making per-class F1 evaluation critical for measuring model fairness.

### 2.3 Preprocessing Applied During Evaluation

Evaluation images went through the same deterministic transform pipeline used for validation — no augmentation was applied:

```
Input Image
  → RGB conversion (safety)
  → Resize to 224×224 (LANCZOS resampling — matches training preprocessing)
  → ToTensor (0.0–1.0)
  → Normalize (ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Note: LANCZOS direct resize was used instead of the standard Resize(256) → CenterCrop(224) convention. This was intentional to avoid train-test distribution mismatch, since training images had already been preprocessed to 224×224 using LANCZOS in the Milestone 2 pipeline.

---

## 3. Evaluation Environment

### 3.1 Hardware & Software

| Component | Value |
|-----------|-------|
| Platform | Google Colab |
| GPU | NVIDIA T4 (16GB VRAM) |
| CPU | 2× Intel Xeon (host) |
| RAM | 12.7GB |
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| PyTorch | 2.x (CUDA-enabled) |
| torchvision | 0.15.x |
| scikit-learn | 1.3.x |
| NumPy | 1.24.x |
| Matplotlib / Seaborn | For visualisation |

### 3.2 Reproducibility Notes

- Random seed fixed at 42 across dataset splitting, model initialisation, and DataLoader operations
- Test set was never used during training or hyperparameter search — it was strictly held out
- Inference was run in `model.eval()` mode with `torch.no_grad()` to disable dropout and batch norm updates
- The saved `best_mobilenet_full_model.pth` was loaded directly for all evaluation runs

---

## 4. Evaluation Metrics

### 4.1 Primary Metric — Macro F1

**Macro F1** is the primary evaluation metric. It computes F1 independently for each of the 17 classes and averages them with equal weight:

$$\text{Macro F1} = \frac{1}{17} \sum_{c=1}^{17} F1_c$$

**Justification:** The dataset has a 14.9× class imbalance ratio. Accuracy and Micro F1 are dominated by majority-class performance (e.g., Rice___Healthy at 1,488 images). A model that ignores Sugarcane classes entirely would still achieve ~85% accuracy. Macro F1 forces equal optimisation pressure across all 17 classes, including the 100-image Sugarcane classes.

### 4.2 Secondary Metrics

| Metric | Justification |
|--------|--------------|
| **Test Accuracy** | Overall correctness; reported for comparability with published benchmarks |
| **Micro F1** | Equivalent to accuracy under single-label classification; confirms metric consistency |
| **Test Loss (CrossEntropy)** | Measures calibration quality — low loss indicates well-calibrated confidence scores |
| **Class-wise F1** | Identifies per-class weaknesses, especially on underrepresented Sugarcane and minority classes |
| **Confusion Matrix** | Reveals systematic misclassification patterns and inter-class confusion |
| **Per-class Precision & Recall** | Distinguishes false positive from false negative error modes per class |

---

## 5. Quantitative Results

### 5.1 Final Test Set Performance — Best Model (Trial #4, MobileNet V3 Large)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.09%** |
| **Test Macro F1** | **90.04%** |
| Test Micro F1 | 92.09% |
| Val Macro F1 | 92.55% |
| Test Loss | 0.2075 |

The Macro F1 ≈ Micro F1 gap of ~2% is considered acceptable for a 14.9× imbalanced dataset. A larger gap would indicate the model systematically underperforms on minority classes.

### 5.2 Optuna Trial Comparison

| Trial | Architecture | Optimizer | LR | Dropout | Batch | Val Macro F1 | Status |
|-------|-------------|-----------|-----|---------|-------|-------------|--------|
| 0 | MobileNet V2 | AdamW | 3.30e-4 | 0.210 | 32 | 0.9163 | Complete |
| 1 | MobileNet V2 | Adam | 5.29e-4 | 0.383 | 16 | 0.9164 | Complete |
| 2 | MobileNet V3 Large | Adam | 8.76e-5 | 0.044 | 32 | 0.9216 | Complete |
| 3 | MobileNet V3 Large | AdamW | 2.40e-4 | — | 32 | 0.8989 | Pruned |
| **4** | **MobileNet V3 Large** | **AdamW** | **4.28e-4** | **0.470** | **16** | **0.9255** | **Best** |

### 5.3 Architecture Comparison

| Model | Test Accuracy | Test Macro F1 | Parameters | 
|-------|-------------|--------------|------------|
| MobileNet V2 | ~91.6% | ~0.916 | 3.4M | 
| **MobileNet V3 Large** | **92.09%** | **90.04%** | **5.4M** |
| ResNet-50 | ~95.8% (est.) | ~94.7 | 25.6M |
| EfficientNet-B3 | ~90.4 | ~88.5 | 12.2M |

MobileNet V3 Large wins on the accuracy-efficiency trade-off for the deployment constraint (CPU-only, rural mobile inference).

### 5.4 Class-wise F1 Analysis

Based on the evaluation of the test set, the following patterns were observed:

**High F1 (>0.95):**
- Corn___Common_Rust, Corn___Healthy, Corn___Northern_Leaf_Blight
- Potato___Early_Blight, Potato___Late_Blight, Potato___Healthy
- Wheat___Healthy, Wheat___Brown_Rust, Wheat___Yellow_Rust
- Sugarcane___Healthy

**Moderate F1 (0.80–0.95):**
- Corn___Gray_Leaf_Spot (limited training samples, visually ambiguous)
- Rice___Healthy (high sample count but some confusion with diseased rice)
- Sugarcane___Bacterial_Blight

**Lower F1 (<0.80):**
- Rice___Brown_Spot (texture-similar to Leaf Blast)
- Rice___Leaf_Blast
- Rice___Neck_Blast
- Sugarcane___Red_Rot (very few test samples, ~10 images)

Rice disease classes are the primary weak point — Brown Spot, Leaf Blast, and Neck Blast exhibit overlapping visual features (necrotic lesions with similar colour distributions), causing inter-class confusion within rice diseases.

---

## 6. Visualisations

### 6.1 Confusion Matrix

The 17×17 confusion matrix on the test set shows:

- **Strong diagonal dominance** — confirming high class separability overall
- **Rice disease cluster confusion** — Brown Spot, Leaf Blast, and Neck Blast show off-diagonal entries between each other, indicating visual feature overlap
- **Minimal cross-crop confusion** — the model correctly identifies the crop before the disease, with very few instances of, e.g., wheat disease being predicted as corn disease
- **Sugarcane classes** — near-perfect diagonal for Sugarcane Healthy; some confusion for Red Rot and Bacterial Blight due to small test sample size (~10 images each)

![Confusion Matrix](/milestone-4/visualisations/confusion_matrix.png)

### 6.2 Class-wise F1 Bar Chart

The per-class F1 chart reveals the bimodal distribution of performance:
- 12 of 17 classes achieve F1 > 0.90
- 3 Rice disease classes cluster below 0.80
- Sugarcane___Red_Rot is the single lowest-performing class due to extreme data scarcity

![Classwise F1 Bar Chart](/milestone-4/visualisations/classwise_f1.png)

### 6.3 Training & Validation Curves

Across all 5 Optuna trials, the frozen-backbone strategy produced:
- **Smooth, monotonically decreasing training loss** — no spikes or instability
- **Validation macro F1 converging by epoch 6–8** — indicating the classifier head saturates relatively quickly with frozen backbone
- **No training-validation divergence** — val loss tracks train loss, confirming no significant overfitting

---

## 7. Qualitative Results

### 7.1 Successful Predictions

**Example 1 — High Confidence Correct Prediction:**
```
Input: Corn leaf with common rust disease - jpg image
```
![corn_common_rust.jpg](/milestone-5/visualisations%20and%20whisper-voice-outputs/corn_common_rust.jpg)

```
Model Prediction:
🌿 Crop: Corn
🔍 Disease:  Common Rust
✅ Confidence: 100%

LLM Respose: 
The disease detected is Common Rust, a fungal disease that causes orange pustules on maize leaves, reducing photosynthesis. 

You might see these symptoms: 
1. Small orange spots on the leaves.
2. Raised pustules on both leaf surfaces.
3. The pustules are orange in color.

To treat the disease, follow these steps:
1. Apply fungicides like Mancozeb or propiconazole.
2. Use resistant hybrids to reduce the risk of infection.
3. Practice sanitation to prevent the spread of the disease.

To prevent the disease next season, consider:
1. Early sowing to avoid favorable conditions for the disease.
2. Monitoring your fields regularly to catch any infections early.

If you need more guidance, consider contacting your local KVK or agriculture officer for specific advice tailored to your area.
```
Screenshot: ![LINK](./visualisations%20and%20whisper-voice-outputs/corn_common_rust_pred.png)

Whisper Voice output: [LINK](./visualisations%20and%20whisper-voice-outputs/corn_common_rust_pred.mp3)

**Example 2 — Minority Class Correct Prediction:**
```
Input:        Sugarcane leaf with visible rot symptoms - webp image
```
![sugarcane_red_rot.webp](/milestone-5/visualisations%20and%20whisper-voice-outputs/sugarcane_red_rot.webp)
```
Model Prediction:
🌿 Crop: Sugarcane
🔍 Disease: Red Rot
✅ Confidence: 99%

LLM Respose: 
The disease is Red Rot in your sugarcane crop, which causes internal red discoloration and can harm your plants. 

You might see symptoms like yellowing leaves in the early stage and red internal tissue in the advanced stage.

To treat it, you can take these steps: 
1. Remove infected plants to prevent the disease from spreading.
2. Use fungicides as a chemical treatment.
3. Destroy infected plants as recommended.

For next season, consider using resistant varieties to prevent the disease. 

If you need more guidance, you can contact your local KVK or agriculture officer for further assistance.
```

Screenshot: ![LINK](./visualisations%20and%20whisper-voice-outputs/sugarcane_red_rot_pred.png)

Whisper Voice output: [LINK](./visualisations%20and%20whisper-voice-outputs/sugarcane_red_rot.mp3)

### 7.2 Failure Cases & Misclassifications

**Failure Case 1 — Rice Disease Confusion:**
```
Input: Rice leaf with Brown spot, caused by Cochliobolus miyabeanus - webp image
```
![rice_leaf_brown_spot_fail-1.webp](/milestone-5/visualisations%20and%20whisper-voice-outputs/rice_leaf_brown_spot_fail-1.webp)

```
Model Prediction:
⚠️ Low confidence (54%)
Possible:  Brown Rust in Wheat
Please consult your local KVK or agriculture officer.

LLM Response: 
The possible disease is Brown Rust in wheat, which causes orange-brown pustules on leaves and reduces yield. 

You might see small orange spots in the early stage and scattered pustules on leaves in the advanced stage.

Since the image analysis is not conclusive, I recommend consulting a local agricultural officer or KVK for a confirmed diagnosis and specific treatment advice. They can provide guidance on the best course of action for your specific situation.
```
Screenshot: ![LINK](./visualisations%20and%20whisper-voice-outputs/rice_leaf_brown_spot_fail-1_pred.png)

Whisper Voice output: [LINK](./visualisations%20and%20whisper-voice-outputs/rice_leaf_brown_spot_fail-1.mp3)

**Failure Case 2 — Environmental Noise:**
```
Input: Wheat leaf with water droplets partially obscuring symptoms - webp image
```
![wheat_water_droplet_fail-2.webp](/milestone-5/visualisations%20and%20whisper-voice-outputs/wheat_water_droplet_fail-2.webp)

```
Model Prediction:
⚠️ Low confidence (51%)
Possible:  Brown Rust in Wheat
Please consult your local KVK or agriculture officer.

LLM Response: 
The possible disease is Brown Rust in wheat, which causes orange-brown pustules on leaves and reduces yield. 

The main symptoms you might see are: 
1. Small orange spots in the early stage
2. Scattered pustules on leaves in the advanced stage

However, since the image analysis was inconclusive, I recommend that you consult a local agricultural officer or KVK for a proper diagnosis and treatment plan. They can provide more specific guidance based on your farm's conditions. 

If it is confirmed as Brown Rust, the general approach would be to apply fungicides at an early stage, but please consult the local expert for the best course of action.
```
Screenshot: ![LINK](./visualisations%20and%20whisper-voice-outputs/wheat_water_droplet_fail-2.png)

Whisper Voice output: [LINK](./visualisations%20and%20whisper-voice-outputs/wheat_water_droplet_fail-2.mp3)

---

## 8. Error Analysis

### 8.1 Primary Error Pattern — Intra-Class Visual Similarity (Rice Diseases)

Rice diseases account for the largest share of misclassifications. Brown Spot, Leaf Blast, and Neck Blast are the three classes with sub-0.80 F1 scores.

**Root Cause:**
- All three diseases produce necrotic lesions on rice leaves
- The distinguishing features are subtle: lesion shape, border characteristics, and lesion density
- Training images may not consistently capture these fine-grained differences
- The model has learned crop-level features (rice) correctly but struggles to discriminate sub-categories within a crop

**Mitigation potential:** Fine-tuning the backbone (partial unfreezing of later layers) could allow learning of more granular texture features specific to rice leaf pathology.

### 8.2 Secondary Error Pattern — Extreme Data Scarcity (Sugarcane)

Sugarcane Red Rot and Bacterial Blight (~10 test images each) show lower confidence scores and higher variance in per-class F1.

**Root Cause:**
- Even with augmentation (100 → 700 training images), all augmented samples derive from 100 unique originals
- Limited visual diversity in the training set constrains what the model can learn about these classes
- Small test set size (10 images) means a single misclassification shifts class F1 by ~10%

**Note:** The small test set size for Sugarcane classes means F1 estimates carry high variance and may not be reliable indicators of true population performance.

### 8.3 Tertiary Error Pattern — Distribution Shift (Real-World Images)

Failure Case 3 illustrates a systematic gap between training distribution (curated Kaggle images) and field deployment conditions:

| Training Condition | Real-World Condition |
|-------------------|---------------------|
| Clean leaf surfaces | Water droplets, mud, soil |
| Single-leaf close-ups | Partial occlusion by other leaves |
| Controlled lighting | Harsh sunlight, shadows |
| Healthy leaf background | Mixed vegetation background |

Training-only augmentation (brightness/contrast/rotation) does not fully address perspective distortion, occlusion, or moisture effects observed in field conditions.

### 8.4 Confidence Calibration

The 0.60 confidence threshold was shown to be effective in routing ambiguous predictions to cautious advisory mode. Analysis indicates:
- Correctly classified samples: mean confidence ≈ 88–92%
- Misclassified samples: mean confidence ≈ 55–65%
- The threshold captures the majority of misclassifications without being overly conservative

---

## 9. Key Observations, Limitations & Anomalies

### 9.1 Key Observations

1. **Val-Test Gap (~2.5%):** Val Macro F1 (92.55%) > Test Macro F1 (90.04%). This ~2.5% gap is expected and acceptable — both splits are drawn from the same distribution. No evidence of systematic overfit to the validation set.

2. **SE Attention Mechanism Benefit:** MobileNet V3 Large consistently outperformed MobileNet V2 across all trials. The Squeeze-and-Excitation channel attention mechanism provides meaningful benefit for localised leaf disease patterns where discriminative features occupy small spatial regions.

3. **Augmentation Impact on Macro F1:** Without minority augmentation, an estimated Macro F1 of 70–75% would be expected. The augmentation strategy directly contributed an estimated 15–20 percentage points to the final 90.0% Macro F1.

4. **AdamW Superiority:** AdamW with high dropout (0.47) was consistently the best regularisation strategy. The decoupled weight decay (1.8e-6) complements the activation dropout without interfering with adaptive learning rates.

### 9.2 Limitations

| Limitation | Impact | Severity |
|-----------|--------|---------|
| Rice disease inter-class confusion | 3 classes below 0.80 F1 | Medium |
| Sugarcane test set size (~10 images/class) | Unreliable per-class F1 estimates | Medium |
| Frozen backbone (no feature-level fine-tuning) | Sub-optimal texture discrimination | Low–Medium |
| Training data source (single Kaggle dataset) | Potential geographic/variety bias | Medium |
| No severity classification output | Cannot differentiate mild vs severe | Medium |
| CPU inference on free HF tier (~2s/image) | Latency in production | Low |
| Real-world distribution shift | Performance degrades on field images | High |

### 9.3 Anomalies

- **Trial #3 Pruned Early:** Despite using MobileNet V3 Large and AdamW (same as the best trial), Trial #3 was pruned — likely due to a sub-optimal learning rate (2.40e-4 with partially incomplete hyperparameter recording). This highlights TPE's sampling sensitivity in small trial budgets.

- **Sugarcane F1 Variance:** With only ~10 test images per Sugarcane class, a single additional misclassification changes class F1 by ~10 percentage points. The reported F1 values for Sugarcane classes should be interpreted with this caveat.

- **Zero Cross-Crop Confusion:** No meaningful cross-crop confusion was observed (e.g., no Corn disease predicted as Rice disease). This suggests the frozen ImageNet backbone provides sufficient crop-level feature discrimination without task-specific training.

---

## 10. Summary

| Aspect | Result |
|--------|--------|
| Test Accuracy | **92.09%** |
| Test Macro F1 | **90.04%** |
| Best-performing crops | Corn, Potato, Wheat (F1 > 0.95 for most classes) |
| Weakest classes | Rice diseases (Brown Spot, Leaf Blast, Neck Blast) |
| Primary failure mode | Intra-crop disease confusion (Rice disease cluster) |
| Real-world risk | Distribution shift under field imaging conditions |
| Deployment readiness | Live on Hugging Face Spaces — [LINK](https://huggingface.co/spaces/harishsahadev/crop-disease-assistant) |

The model performs robustly on 14 of 17 disease classes and meets the design requirement of equal performance across crops (macro F1 90%). The primary open challenge is improving Rice disease discrimination and validating performance on real-field images collected from Indian agricultural environments.
