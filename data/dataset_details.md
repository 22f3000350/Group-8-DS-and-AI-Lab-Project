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

## Model Results and Best Parameter Result

- **Model Results Folder:** [LINK](https://drive.google.com/drive/folders/1hglOl5DpqDkR8bwimz5a1iYUGs0DGZC0?usp=drive_link)
- **Best Parameter Result File:** `best_metrics.json` in the model results folder
- **Best Trial:** Trial 4
- **Best Model:** MobileNet V3 Large with AdamW
- **Best Hyperparameters:** learning rate = `4.28e-4`, dropout = `0.470`, batch size = `16`, weight decay = `1.8e-6`

---