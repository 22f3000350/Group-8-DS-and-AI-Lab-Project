MILESTONE 2 REPORT
  
Multimodal AI Assistant for Smart Agriculture
Dataset Documentation, EDA & Preprocessing

Dataset:  Top Agriculture Crop Disease  |  Kaggle (kamal01)
Crops:  Corn  |  Potato  |  Rice  |  Wheat  |  Sugarcane
Total Images:  13,324  |  17 Classes  |  5 Crops

2025

1. Dataset Documentation
This section provides a complete, structured description of the dataset selected for the Crop Disease Detection project, including its source, licensing, composition, and relevant metadata.

1.1  Dataset Overview
| Field | Details |
| --- | --- |
| Dataset Name | Top Agriculture Crop Disease |
| Source | Kaggle — kaggle.com/datasets/kamal01/top-agriculture-crop-disease |
| License | CC BY 4.0 (Rice subset); PlantVillage terms (Corn, Potato); Kaggle dataset terms apply |
| Task Type | Multi-class Image Classification |
| Total Images | 13,324 |
| Total Classes | 17 disease/health categories |
| Total Crops | 5 (Corn, Potato, Rice, Wheat, Sugarcane) |
| Image Format | JPEG / PNG |
| Image Channels | RGB (3-channel) — confirmed by EDA |


1.2  Source Breakdown by Crop
The dataset aggregates images from multiple well-established plant disease datasets, ensuring diversity in imaging conditions and geographic origin:

Corn (4 classes): Sourced from PlantVillage, the most widely used benchmark dataset for plant disease classification. Images are controlled lab photographs on uniform backgrounds.
Potato (3 classes): Also sourced from PlantVillage, following the same controlled imaging conditions as Corn.
Rice (4 classes): Sourced from Dhan-Shomadhan — a Bangladeshi local rice disease dataset with CC BY 4.0 license — supplemented by the Rice Leafs dataset from Kaggle. Images include both leaf and background variations.
Wheat (3 classes): Sourced from the Wheat Disease Detection dataset on Kaggle, containing real-field photographs with natural backgrounds.
Sugarcane (3 classes): Sourced from the Sugarcane Disease Dataset on Kaggle. This subset is the smallest, with only 100 images per class.

1.3  Class-wise Distribution
The table below details every class, its image count, crop type, and data source. Classes highlighted in yellow belong to the Sugarcane crop, which exhibits significant under-representation.

| # | Class Label | Images | Crop | Source |
| --- | --- | --- | --- | --- |
| 1 | Corn___Common_Rust | 1,192 | Corn | PlantVillage |
| 2 | Corn___Gray_Leaf_Spot | 513 | Corn | PlantVillage |
| 3 | Corn___Healthy | 1,162 | Corn | PlantVillage |
| 4 | Corn___Northern_Leaf_Blight | 985 | Corn | PlantVillage |
| 5 | Potato___Early_Blight | 1,000 | Potato | PlantVillage |
| 6 | Potato___Healthy | 152 | Potato | PlantVillage |
| 7 | Potato___Late_Blight | 1,000 | Potato | PlantVillage |
| 8 | Rice___Brown_Spot | 613 | Rice | Dhan-Shomadhan + Rice Leafs (Kaggle) |
| 9 | Rice___Healthy | 1,488 | Rice | Dhan-Shomadhan + Rice Leafs (Kaggle) |
| 10 | Rice___Leaf_Blast | 977 | Rice | Dhan-Shomadhan + Rice Leafs (Kaggle) |
| 11 | Rice___Neck_Blast | 1,000 | Rice | Dhan-Shomadhan + Rice Leafs (Kaggle) |
| 12 | Wheat___Brown_Rust | 902 | Wheat | Wheat Disease Detection (Kaggle) |
| 13 | Wheat___Healthy | 1,116 | Wheat | Wheat Disease Detection (Kaggle) |
| 14 | Wheat___Yellow_Rust | 924 | Wheat | Wheat Disease Detection (Kaggle) |
| 15 | Sugarcane__Red_Rot | 100 | Sugarcane | Sugarcane Disease Dataset (Kaggle) |
| 16 | Sugarcane__Healthy | 100 | Sugarcane | Sugarcane Disease Dataset (Kaggle) |
| 17 | Sugarcane__Bacterial_Blight | 100 | Sugarcane | Sugarcane Disease Dataset (Kaggle) |


⚠️  Class Imbalance Note:
The most populated class (Rice___Healthy, 1,488 images) has 14.9x more images than the least populated classes (all three Sugarcane classes, 100 images each). This severe imbalance must be addressed during preprocessing before model training.


2. Exploratory Data Analysis (EDA)
EDA was performed to gain a deep understanding of the dataset's structure, quality, and visual characteristics before any preprocessing was applied. The findings directly informed all preprocessing decisions documented in Section 3.

2.1  Class Distribution Analysis
The class distribution was analysed by counting images per class and per crop. The following key findings were identified:

Total of 13,324 images across 17 classes and 5 crop species.
Rice has the most images overall (4,078), contributing 30.6% of the dataset.
Sugarcane is the most under-represented crop with only 300 total images (100 per class), accounting for just 2.3% of the full dataset.
The imbalance ratio between the largest and smallest class is 14.9x (Rice___Healthy: 1,488 vs. any Sugarcane class: 100).
Corn___Gray_Leaf_Spot (513 images) and Potato___Healthy (152 images) are also notably under-represented compared to other classes in their respective crops.

→ Implication: Without addressing class imbalance, a model trained on this raw distribution would be biased towards majority classes (Rice, Corn) and would perform poorly on minority classes (Sugarcane), resulting in misleading overall accuracy metrics.

2.2  Image Size & Resolution Analysis
All images were scanned to extract their width, height, and aspect ratio. The findings revealed high variability across the dataset:

| Metric | Value |
| --- | --- |
| Min Width | 16 px |
| Max Width | 6,000 px |
| Mean Width | 689.8 px |
| Min Height | 1 px |
| Max Height | 4,160 px |
| Mean Height | 657.2 px |


The extreme range — from as small as 16×1 pixels to as large as 6,000×4,160 pixels — confirms that the dataset contains images from fundamentally different sources with incompatible native resolutions. This is expected given that the dataset aggregates from PlantVillage (controlled lab images at fixed resolution), Kaggle field datasets (real-world photographs at varying resolutions), and augmented subsets.

The aspect ratio distribution shows that while many images are approximately square (ratio ≈ 1.0, typical of PlantVillage), a significant proportion of real-field images are in landscape or portrait orientation. This further necessitates resizing to a fixed square dimension for consistent model input.

2.3  Color Channel Analysis
All 13,324 images were verified to be 3-channel RGB images. No grayscale, RGBA, or single-channel images were found in the dataset. This uniformity simplifies the preprocessing pipeline, though a safety RGB conversion step has been retained in the pipeline to handle any edge cases that may arise during data loading in M3.

2.4  Pixel Intensity & Color Analysis
A sample of 500 images (stratified by class) was used to analyse pixel intensity distributions across the RGB channels:

The green channel (G) consistently shows higher mean intensity values across all crop classes, which is expected for leaf imagery where chlorophyll dominates the visual spectrum.
Diseased classes show visibly lower green channel intensity compared to their corresponding healthy class, confirming that color shift is a key discriminative feature between healthy and diseased leaves.
Sugarcane images show a different intensity profile compared to other crops, attributable to different imaging conditions (field vs. lab photography).
The average image per crop (visual fingerprint) confirms these differences: Corn and Potato (PlantVillage) appear brighter and more uniform, while Rice and Wheat field images show more natural, varied backgrounds.

2.5  Data Quality Check
A thorough data quality inspection was performed using three methods:

Corrupt File Detection: All 13,324 images were opened and verified using PIL's verify() method. Result: 0 corrupt files detected. The dataset is fully readable.
Exact Duplicate Detection: MD5 cryptographic hashing was applied to every image file. Result: 0 exact duplicate images detected. No data leakage risk from byte-identical files.
Near-Duplicate Detection: Perceptual hashing (pHash) was applied to a 500-image sample with a similarity threshold of distance ≤ 5. Result: 2 near-duplicate pairs detected. These will be removed during preprocessing.

✅ Overall data quality is high. The dataset is free of corrupt files and exact duplicates.
→ The 2 near-duplicate pairs found in the 500-image sample suggest a small number may exist in the full dataset. Full near-duplicate removal will be applied as a precautionary step during preprocessing.


3. Preprocessing Pipeline
Based on the findings from EDA, a structured preprocessing pipeline was designed. Each step is documented below with a clear justification for why it is necessary. The pipeline ensures the dataset is fully standardised and ready to be fed directly into model architectures during M3 without requiring any further restructuring.

3.1  Preprocessing Steps with Justification

| Step | Technique | Justification |
| --- | --- | --- |
| 1 | Resize to 224×224 | Standard input size for CNNs (ResNet, EfficientNet, VGG, MobileNet). Images range from 16px to 6000px — uniform size is mandatory. 224×224 preserves sufficient detail while being computationally efficient. |
| 2 | RGB Conversion | EDA confirmed 0 non-RGB images; however, the conversion step is retained as a safety measure in the pipeline to handle any edge cases without pipeline failure. |
| 3 | Duplicate Removal | 0 exact duplicates detected via MD5 hashing. 2 near-duplicate pairs found (perceptual hash). Removing these prevents data leakage between train and test splits. |
| 4 | Corrupt File Removal | 0 corrupt files detected. The corrupt-check step is retained to prevent runtime crashes during batch training. |
| 5 | Data Augmentation | Severe class imbalance (14.9x ratio). Sugarcane classes have only 100 images each. Augmentation (flip, rotate ±30°, brightness/contrast shift, zoom) applied to minority classes to balance the dataset without collecting new data. |
| 6 | Normalization | ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] applied. All planned M3 models (ResNet, EfficientNet) are pretrained on ImageNet — using the same normalization ensures the model's pretrained weights are directly applicable. |
| 7 | Stratified Train/Val/Test Split | 80/10/10 split using sklearn's stratified splitting ensures all 17 classes are proportionally represented in every subset, preventing splits where minority classes (Sugarcane, 100 images) appear only in one set. |


3.2  Augmentation Strategy
Since Sugarcane (100 images/class), Potato___Healthy (152 images), and Corn___Gray_Leaf_Spot (513 images) are significantly under-represented, targeted augmentation is applied to bring all classes closer to the dataset mean. The following augmentation transforms are applied only to minority class images during training (never to validation or test sets):

Horizontal & Vertical Flip: Simulates different leaf orientations captured in the field. No semantic meaning is lost for disease detection.
Random Rotation (±30°): Accounts for variability in how leaves are positioned when photographed.
Brightness & Contrast Adjustment (±20%): Simulates different lighting conditions across field and lab images.
Random Zoom (0.9–1.1×): Simulates varying distances between camera and leaf surface.
Gaussian Noise (optional): Adds robustness against image sensor noise, particularly relevant for low-resolution Sugarcane images.

⚠️  Important: Augmentation is applied ONLY to training images, never to validation or test images.
→ Augmenting validation/test sets would artificially improve metrics and misrepresent real-world model performance.

3.3  Train / Validation / Test Split
The dataset is split into three subsets using a stratified approach to ensure proportional class representation across all splits:

| Split | Proportion  |  Purpose |
| --- | --- |
| Training Set (80%) | 10,659 images — Used for model weight updates during training |
| Validation Set (10%) | 1,332 images — Used for hyperparameter tuning and early stopping |
| Test Set (10%) | 1,333 images — Held out; used only for final model evaluation |


Stratified splitting (sklearn.model_selection.train_test_split with stratify=y) is used to guarantee that even the smallest classes (Sugarcane, 100 images each) have approximately 80 training, 10 validation, and 10 test images rather than being concentrated in a single split by chance.

3.4  Final Dataset Structure
After applying all preprocessing steps, the dataset will be saved in the following structure, ready for direct loading in M3:

processed/
    train/
        Corn___Common_Rust/
        Corn___Gray_Leaf_Spot/
        ...  (all 17 classes)
    val/
        ...  (all 17 classes)
    test/
        ...  (all 17 classes)
    eda_master_dataframe.csv   ← image paths, labels, dimensions, hashes

The eda_master_dataframe.csv file — generated during EDA — is preserved alongside the split folders. It contains the file path, class label, crop, image dimensions, channel count, aspect ratio, and MD5 hash for every image. This manifest file serves as a single source of truth for all downstream data loading operations in M3.


4. Summary & M3 Readiness
This section summarises the key findings from EDA and confirms that the dataset is fully prepared for the modelling phase (M3).

4.1  Key EDA Findings
The dataset contains 13,324 images across 17 classes from 5 crop species, aggregated from multiple benchmark and real-world sources. 
A severe class imbalance of 14.9x exists, with Sugarcane classes containing only 100 images each versus 1,488 for the largest class. 
Image dimensions are highly variable (16px to 6,000px), requiring mandatory resizing to a uniform input size. 
All images are in RGB format — no channel conversion is required, though a safety step is retained. 
The dataset has excellent overall quality: 0 corrupt files and 0 exact duplicates, with only 2 near-duplicate pairs detected in a 500-image sample. 
Color analysis confirms that the green channel is the dominant feature for healthy leaves, with measurable channel-level shifts visible in diseased classes. 

4.2  Preprocessing Decisions Summary
Resize: 224×224 — compatible with all major pretrained CNN architectures.
Normalisation: ImageNet mean/std — essential for effective transfer learning.
Augmentation: Applied to minority classes only — addresses 14.9x imbalance without data collection.
Split: 80/10/10 stratified — ensures M3 evaluation is fair and representative.
Data Format: ImageFolder-compatible directory structure — loadable by PyTorch or TensorFlow with zero restructuring.



End of Milestone 2 Report