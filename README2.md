# Autonomous Fraud Detection Engine: Supervised Learning Pipeline

An end-to-end, production-grade machine learning pipeline designed to isolate and classify fraudulent financial transactions within highly imbalanced streaming data. This system leverages advanced synthetic oversampling, isolated transformation pipelines, and non-linear tree-based ensembles to ensure robust anomaly detection without data leakage.

## 📌 Project Overview & Context
In real-world transaction ecosystems, fraudulent activity constitutes a minuscule fraction of global volume. Standard classification models frequently fail in these environments due to accuracy bias, where a model can achieve 99% accuracy simply by classifying every transaction as legitimate. 

This project establishes a rigorous framework that completely discards global accuracy in favor of cost-sensitive metrics—specifically **Precision**, **Recall**, and **Area Under the ROC Curve (ROC-AUC)**—to accurately identify fraud vectors while minimizing false positives for genuine users.

## 🛠️ Key Technical Requirements Addressed
* **Class Imbalance Mitigation:** Integrated Synthetic Minority Over-sampling Technique (SMOTE) to synthetically balance the training minority class distribution.
* **Leakage-Free Validation Architecture:** Engineered custom target-aware processing sequences utilizing `imblearn.pipeline.Pipeline` to guarantee that oversampling is strictly contained within cross-validation folds and never pollutes evaluation sets.
* **Dual Algorithmic Evaluation:** Built, evaluated, and contrasted baseline Linear Models (Logistic Regression with standard scaling) against complex Non-Linear Tree Ensembles (Random Forest Classifiers).
* **Joint Hyperparameter Optimization:** Implemented a stratified multi-fold Grid Search Cross-Validation engine (`GridSearchCV`) to optimize SMOTE neighborhood parameters and model complexity constraints simultaneously.

## 📊 Pipeline Architecture & Workflow

### 1. Unique Ingestion & Data Profiling
The engine dynamically instantiates a unique synthetic dataset mimicking standard financial transaction feature attributes ($V_1$ through $V_{30}$) while maintaining a highly skewed target class constraint (99.83% Legitimate, 0.17% Fraudulent).

### 2. Isolated Stratified Splitting
Data is partitioned using a stratified 80/20 train/test split. This ensures that the highly delicate fraud density distribution is precisely mirrored within both the training domain and the final testing domain.

### 3. Execution Pipeline Setup
[ Raw Training Folds ] ──> [ StandardScaler ] ──> [ SMOTE Resampling ] ──> [ Classifier Training ]
│
[ Untouched Evaluation Fold ] ─────────────────────────────────────────────> [ Strict Validation ]

* **Linear Path:** Standardizes variables to protect gradient calculations, applies SMOTE, and optimizes `LogisticRegression`.

* **Ensemble Path:** Bypasses scale transformations, applies SMOTE, and runs `RandomForestClassifier`.

### 4. Metrics Dashboard
The model evaluation strategy prioritizes:
* **Recall (Sensitivity):** Maximizes the capture rate of fraudulent transactions to minimize financial exposure.
* **Precision:** Minimizes false alarm triggers to prevent customer friction.
* **ROC-AUC:** Measures the model's structural ability to distinguish between legitimate and fraudulent distributions across all thresholds.

## 🚀 Getting Started & Local Setup

### Dependencies
Ensure you have the following packages installed in your Python environment:
```bash
pip install numpy pandas scikit-learn imbalanced-learn
Execution
Clone this repository to your local directory.

Ensure your custom generated transaction data or creditcard.csv is present in the working directory.

Open a terminal and launch the Jupyter environment:

Bash
jupyter notebook
Run all cells in Fraud_Detection_Pipeline.ipynb sequentially to execute the data synthesis, pipeline compilation, hyperparameter tuning, and metric evaluation stages.

 Key Findings Summary
Pipeline Integrity: Using standard Scikit-Learn pipelines introduces severe data leakage by oversampling validation folds, creating artificially inflated metrics. Switching to imblearn pipelines isolates the synthetic data generation completely.

Model Contrast: While Logistic Regression provides stable processing speeds, the Random Forest Ensemble yields inherently superior precision controls, keeping false-positive transaction declines to a minimum.


This README is fully formatted in clean Markdown, complete with structured sections, visual architecture
