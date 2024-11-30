# Predicting Chronic Kidney Disease with Machine Learning

## Overview

This repository implements a machine learning pipeline to predict chronic kidney disease (CKD) based on patient health data. The project demonstrates data preprocessing, feature selection, model training, and evaluation techniques using Python and popular libraries like scikit-learn.

The pipeline generates comprehensive outputs, including a PDF report, a text summary of results, and detailed logs, making it a complete solution for showcasing ML workflows in healthcare.

---

## Dataset

### Description

The dataset contains 25 features describing patient health metrics collected over a 2-month period in India. The target variable, `classification`, indicates whether a patient has CKD (`ckd`) or not (`notckd`). The dataset includes 400 rows and is moderately imbalanced.

### Features

- **Example Features**:
  - `bgr` (Blood Glucose Random)
  - `rc` (Red Blood Cell Count)
  - `wc` (White Blood Cell Count)
  - `sg` (Specific Gravity)
  - `al` (Albumin)
  - ... (see full dataset for all features).

- **Target Variable**:
  - `classification`: Binary, values are `ckd` (Chronic Kidney Disease) or `notckd` (No CKD).

### Source
- [Kaggle Chronic Kidney Disease Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease/data)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease)

---

## Pipeline Features

### 1. **Data Cleaning**
- Handles missing values:
  - Categorical features: Filled with the mode (most frequent value).
  - Numerical features: Filled with the mean.
- Encodes categorical variables into numeric values using `LabelEncoder`.

### 2. **Feature Selection**
- Selects the top 10 features based on mutual information scores using `SelectKBest`.

### 3. **Class Imbalance Handling**
- Applies **SMOTE (Synthetic Minority Oversampling Technique)** to balance the classes for better model performance.

### 4. **Model Training**
- Trains three models with hyperparameter tuning:
  - Random Forest
  - Gradient Boosting
  - Neural Network (MLPClassifier).
- Hyperparameters are optimized using `GridSearchCV`.

### 5. **Model Evaluation**
- Evaluates models based on:
  - Classification metrics: Accuracy, Precision, Recall, F1-score.
  - ROC-AUC: Plots and scores Receiver Operating Characteristic curves.
  - Feature Importance: Visualizes important features for tree-based models.

### 6. **Outputs**
- **PDF Report**:
  - Includes classification reports, ROC curves, and feature importance plots.
- **Text Summary**:
  - Summarizes evaluation metrics and the best model’s parameters.
- **Log File**:
  - Captures pipeline steps, warnings, and errors.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher.
- Dataset file (`kidney_disease.csv`) in the project directory.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chronic-kidney-disease-prediction.git
   cd chronic-kidney-disease-prediction

# Usage

## Running the Pipeline
1. Ensure the dataset file (`kidney_disease.csv`) is in the root directory of the project.
2. Execute the pipeline script:
   ```bash
   python CKD_Prediction.py
  

---

# Outputs
After running the pipeline, the following files will be created in the `output` directory:

- **PDF Report (`model_results.pdf`)**:
  - Comprehensive results, including classification metrics, ROC curves, and feature importance plots.

- **Text Summary (`model_results.txt`)**:
  - Plain-text summary of evaluation metrics and the best model’s parameters.

- **Log File (`pipeline_log.txt`)**:
  - Step-by-step log of pipeline actions.


---


# Results and Insights

## Best Model:
- The best-performing model is identified based on ROC-AUC and F1-score.
- Model parameters and performance metrics are summarized in both the PDF and text outputs.

## Key Features:
- Feature importance plots show which health indicators are most predictive of CKD.

## Scalability:
- The pipeline is modular, allowing the addition of new models or preprocessing steps.

---

# Acknowledgments
- Dataset from [Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease/data) and [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease).
- Inspired by scikit-learn's official documentation and practical ML tutorials.



