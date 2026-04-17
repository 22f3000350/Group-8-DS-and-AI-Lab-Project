# Dataset Details

This project uses the **Top Agriculture Crop Disease** dataset for crop disease classification.

## Raw Dataset

- **Source:** Kaggle
- **Dataset Name:** Top Agriculture Crop Disease
- **Dataset Kaggle Link:** [LINK](https://www.kaggle.com/datasets/kamal01/top-agriculture-crop-disease)
- **Task Type:** Multi-class image classification
- **Total Images:** 13,324
- **Total Classes:** 17
- **Crops Covered:** Corn, Potato, Rice, Wheat, Sugarcane

## Dataset Summary

- Images are collected from multiple public sources, including PlantVillage and Kaggle-hosted crop disease datasets.
- The dataset contains both healthy and diseased crop leaf images.
- Images are stored in RGB format and were later standardized during preprocessing for model training.

## Processed Dataset

- **Stored Format:** ImageFolder-compatible directory structure
- **Expected Structure:**

```text
processed/
  train/
  val/
  test/
```

- **Processed Data Drive Location:** [LINK](https://drive.google.com/drive/u/1/folders/1d4DTGLAktB89Ng8YTrcs5f3C6MMR0B2W)
- **Manifest File:** [preprocessing_manifest.csv](https://drive.google.com/file/d/1UjKTHp0TrfJGqnbmYlU7i_M7X_NRGvpi/view?usp=sharing)

---