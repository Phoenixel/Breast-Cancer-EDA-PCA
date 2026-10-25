# Breast-Cancer-EDA-PCA
Exploratory Data Analysis and Principal Component Analysis of the Breast Cancer Wisconsin dataset using Python.
# Breast Cancer Wisconsin Dataset: EDA and PCA

## Project Overview

This project explores the Breast Cancer Wisconsin Diagnostic Dataset using Python, focusing on data preprocessing, Exploratory Data Analysis (EDA), and Principal Component Analysis (PCA).

The objective is to understand the relationships between cell nucleus measurements, investigate differences between benign and malignant tumors, and evaluate dimensionality reduction using PCA.

## Dataset

- **Source:** [Kaggle – Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Observations:** 569
- **Numerical features:** 30
- **Diagnosis categories:** Benign (B) and Malignant (M)

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- PyCharm

## Methodology

### 1. Data Preprocessing
- Inspected the dataset structure and data types.
- Identified missing values and excluded the entirely empty column.
- Selected 30 numerical features for analysis.

### 2. Exploratory Data Analysis (EDA)
- Examined descriptive statistics and feature distributions.
- Compared numerical measurements between benign and malignant tumors.
- Investigated correlations between features using a correlation matrix and heatmap.

### 3. Principal Component Analysis (PCA)
- Standardized numerical features using StandardScaler.
- Applied PCA to the 30 numerical measurements.
- Examined explained variance and cumulative explained variance.
- Visualized the first two principal components.
- Investigated PC1 component weights.

## Key Findings

- **PC1:** Explained approximately 44.27% of total variance.
- **PC2:** Explained approximately 18.97%.
- **PC1 + PC2:** Explained approximately 63.24%.
- **10 principal components:** Preserved approximately 95.16% of total variance.

The PCA scatter plot showed noticeable but incomplete separation between benign and malignant observations, primarily along PC1.

Nucleus shape measurements, including concavity and concave points, were among the strongest contributors to PC1.

## Limitations

PCA is an unsupervised method. Although the first two components showed patterns associated with diagnosis, this analysis does not establish predictive accuracy or clinical diagnostic performance.

## Future Improvements

- Further investigate PCA loadings.
- Compare predictive models with and without PCA.
- Evaluate performance using appropriate classification metrics.
