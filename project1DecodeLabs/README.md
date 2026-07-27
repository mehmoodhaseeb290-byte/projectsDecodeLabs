# projectsDecod# Advanced EDA & Feature Engineering Pipeline (Project 1)

## 📋 Project Description
This repository contains a production-grade data preprocessing and feature engineering pipeline developed strictly under the **Input-Process-Output (IPO) Architecture** framework[span_0](start_span)[span_0](end_span). Starting with a raw, chaotic 1,000-row x 5-column dataset injected with high-density noise, severe outliers, and multi-tier missingness, this pipeline automates structural data cleansing through pure statistical logic[span_1](start_span)[span_1](end_span).

The goal of this pipeline is to transform highly chaotic, unstructured raw inputs into a mathematically sound, optimized feature store completely ready to serve downstream machine learning models with peak stability[span_2](start_span)[span_2](end_span).

### 🛠️ Key Pipeline Capabilities:
* **Structural Missing Data Management:** Programmatically routes features through a rigid *Missing Data Decision Matrix* (<5% triggers row deletion, 5%–20% triggers Global Median Imputation for skewed distributions, and >20% triggers Multi-Dimensional KNN Imputation)[span_3](start_span)[span_3](end_span).
* **Anomalous Outlier Neutralization:** Minimizes variance distortions and transcription errors using Interquartile Range (IQR) bounds[span_4](start_span)[span_4](end_span). Values are handled via Winsorization (`numpy.clip()`) to keep row counts perfectly intact and secure temporal sequencing[span_5](start_span)[span_5](end_span).
* **Vectorized Feature Engineering:** Derives 3 entirely new interaction and velocity predictive columns from existing attributes via block-allocated vectorized operations[span_6](start_span)[span_6](end_span).
* **Collinearity Eradication:** Evaluates overlapping numeric pairs ($r > 0.80$), programmatically benchmarking them against the target label ($y$) to drop the weaker column, preventing unstable Ordinary Least Squares (OLS) parameters[span_7](start_span)[span_7](end_span).
* **Orthogonal Coordinate Space Translation:** Eliminates false ascending hierarchies by mapping categorical groups into true equidistant coordinate spaces using One-Hot Encoding[span_8](start_span)[span_8](end_span).

---

## 🏗️ Core System Architecture (IPO Blueprint)
The workflow follows an enterprise-grade modular paradigm[span_9](start_span)[span_9](end_span):
1. **Module 1: Input (Securing Fidelity):** Loads the raw data, monitors missingness thresholds, and establishes statistical outlier boundaries[span_10](start_span)[span_10](end_span).
2. **Module 2: Process (The Engine):** Computes vectorized feature variables, translates labels into orthogonal axes, and checks collinear matrices to eliminate multi-collinearity[span_11](start_span)[span_11](end_span).
3. **Module 3: Output (Contracts & Serving):** Validates the final pristine dataset layout and registers zero remaining matrix holes[span_12](start_span)[span_12](end_span).

---

## 🚀 How to Run the Project

### Prerequisites
Ensure your working machine or environment has the core data engineering libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
eLabs
