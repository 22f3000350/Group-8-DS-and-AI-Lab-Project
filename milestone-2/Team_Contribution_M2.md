# Group Contribution — Crop Disease Detection (Milestone 2)


## Ayushi Dixit 
### Roll - 22f3000082
- Researched and selected the dataset from Kaggle covering 5 crops, 17 disease classes, and 13,324 images with verified licenses.
- Performed complete EDA including class distribution, image size profiling, RGB analysis, and identified a 14.9x class imbalance.
- Ran data quality checks using MD5 hashing and perceptual hashing — confirmed 0 corrupt files, 0 duplicates, 2 near-duplicates removed.
- Built the full preprocessing pipeline — resizing all images to 224×224, RGB conversion, and JPEG export.
- Applied targeted augmentation (flip, rotation, brightness, contrast, zoom) on minority classes to balance the training set.
- Implemented a stratified 80/10/10 train/val/test split with zero cross-split overlap verified via MD5 hash comparison.
- Documented all EDA findings, preprocessing steps, and justifications in the M2 report and combined Colab notebook.
- Built the combined EDA + Preprocessing Colab notebook (.ipynb) as a single end-to-end reproducible pipeline.
- Prepared the markdown version of the report (.md) for team submission and documentation.

## Manas Rastogi
### Roll - 22f3001477
- Assisted in setting up the Colab environment including Kaggle API integration and dataset download pipeline.
- Helped implement the image scanning and master dataframe building logic for all 17 class folders.
- Contributed to writing the class distribution and crop-level visualization cells in the EDA section.
- Assisted in coding the MD5 hashing and pHash near-duplicate detection logic in the notebook.
- Helped debug and test the augmentation function across multiple minority classes during implementation.
- Assisted in verifying the final train/val/test folder structure and cross-split leakage check cell.

## Sai Naman
### Roll - 22f3000350

- For Milestone 2, I was primarily responsible for preparing the presentation and documenting the dataset preparation process. My contributions include:
- Preparing the PowerPoint presentation summarizing the dataset documentation, EDA findings, and preprocessing pipeline.
- Structuring the slides to clearly present the dataset scope, class distribution, and key observations from the analysis.
- Summarizing the preprocessing steps such as resizing, normalization, augmentation, and stratified data splitting.
- Highlighting the final dataset structure and readiness for the modeling phase in Milestone 3.

# Harish Sahadev M, 
### Roll - 21f1005856
For milestone 2, these are my contributions:
- ⁠Reviewed the dataset preparation requirements for Milestone 2.
- ⁠Conducted early research related to data preparation, preprocessing, and EDA for the project.
- ⁠Explored relevant resources to understand how the dataset will be structured and used in the upcoming modeling phase.

## Saketh Allanki
For Milestone 2, I reviewed the selected dataset and its documentation, including the source, licensing, number of samples, and class distribution. I also helped ensure the dataset was properly organized and formatted for the training pipeline. Additionally, I went through the EDA results and preprocessing steps such as image resizing and normalization to confirm that the data was ready for the modeling phase in the next milestone