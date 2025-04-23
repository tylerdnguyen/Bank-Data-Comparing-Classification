# Bank Marketing Classification Notebook

This repository contains a Jupyter Notebook (`Practical_Application_III.ipynb`) that implements an end-to-end Machine Learning pipeline on the UCI Bank Marketing dataset. The goal is to predict whether a client will subscribe to a term deposit, following the CRISP-DM methodology and comparing multiple classification algorithms.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Notebook Structure](#notebook-structure)
6. [Key Results](#key-results)
7. [Next Steps](#next-steps)
8. [Requirements](#requirements)

---

## Project Overview

The notebook guides you through:

- **Business Understanding:** Define objectives and success metrics.
- **Data Understanding:** Load and explore the dataset, identify missing values, and visualize distributions.
- **Data Preparation:** Clean data, engineer features (flags, ratios, cyclical encodings), and encode categoricals.
- **Modeling:** Train and benchmark four classifiers (KNN, Logistic Regression, Decision Tree, SVM).
- **Hyperparameter Tuning:** Optimize each model using `RandomizedSearchCV`.
- **Evaluation:** Compare models on accuracy and training time, and select the best performer.

---

## Dataset

- Source: UCI Machine Learning Repository — Bank Marketing dataset
- File used: `bank-additional-full.csv`
- Records: 41,188 client contacts spanning May 2008 to November 2010
- Features: Demographics, economic indicators, campaign details (e.g., age, job, balance, campaign count, month, etc.)
- Target: `y` (yes/no for subscription to a term deposit)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tylerdnguyen/Bank-Data-Comparing-Classification.git
   cd Bank-Data-Comparing-Classification
   ```
---

## Usage

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open and run `Practical_Application_III.ipynb` sequentially.
3. Review outputs, visualizations, and adjust parameters as needed.

---

## Notebook Structure

1. **Data Understanding** (`01_Data_Understanding`):
   - Check data shape, missing values, and "unknown" placeholders
   - Summarize numeric and categorical distributions
2. **Feature Engineering** (`02_Feature_Engineering`):
   - Drop post-call features
   - Create binary flags and ratio features
   - Apply cyclical encoding for temporal features
   - One-hot encode categorical variables
3. **Modeling & Baseline** (`03_Modeling`):
   - Establish a majority-class baseline
   - Train initial Logistic Regression
4. **Model Comparison** (`04_Model_Comparison`):
   - Fit KNN, Decision Tree, and SVM alongside Logistic Regression
   - Record training time and accuracy
5. **Hyperparameter Tuning** (`05_Hyperparameter_Tuning`):
   - Define search spaces for each model
   - Run `RandomizedSearchCV`
6. **Evaluation & Results** (`06_Evaluation`):
   - Compare final models on train/test accuracy and fit time
   - Summarize key findings

---

## Key Results

| Model               | Train Accuracy | Test Accuracy | Fit Time (s)  |
| ------------------- | -------------- | ------------- | ------------  |
| Logistic Regression | 0.90           | 0.898          | 47.4         |
| K-Nearest Neighbors | 0.90           | 0.896          | 163          |
| Decision Tree       | 0.90           | 0.897          | 2.66         |
| SVM (RBF)           | 0.90           | 0.898          | 231          |

**Best Performer:** Decision Tree achieved a high test accuracy (0.90) with the lowest training time. SVC has the highest training time.

---

## Next Steps

- **Ensemble Methods:** Evaluate Random Forests and Gradient Boosting
- **Class Imbalance:** Experiment with SMOTE, class weighting, or focal loss
- **Feature Selection:** Use feature importance or recursive elimination
- **Deployment:** Package the best pipeline into a REST API or batch scoring script

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- jupyter

Install them with:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

---
