# Machine Learning for Materials Properties: Cohesive Energy Prediction

This repository contains a machine learning pipeline for predicting material stability developed for the **NanoX81 Lab 3** Kaggle competition.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhitran161/Machine-Learning-for-Materials-Properties/blob/main/NANO281_lab3_Quynh_Tran.ipynb)

## 📌 Project Overview
This project predicts the **cohesive energy per atom ($E_c$)** of materials using the **MatPES r2SCAN dataset**. Cohesive energy is a fundamental indicator of thermodynamic stability, critical for identifying viable candidates for semiconductors and structural applications.

By engineering **90+ compositional and structural features** and optimizing ensemble models, this pipeline achieves a prediction error (MSE) of **0.072 eV/atom**, offering a high-throughput method to screen materials without expensive DFT calculations.

## 🚀 Key Results & Performance
* **Best Model:** Histogram-based Gradient Boosting Regressor (HistGradientBoosting)
* **Validation Performance:** 0.072 eV/atom (Mean Squared Error)
* **Kaggle Test Set Score:** 0.075 eV/atom (Private Leaderboard)
* **Improvement:** Reduced prediction error by **88%** compared to the baseline mean predictor.

## 🔍 Features & Insights
This project utilizes `matminer` and `pymatgen` to extract physical meaning from chemical formulas.
* **Dominant Features:** The model identified **average electronegativity** and **atomic radius** as the strongest drivers of cohesive energy.
* **Insight:** Materials with high electronegativity differences tend to exhibit stronger bonding (ionic character), leading to higher cohesive energies (more negative values). The model successfully captured these non-linear periodic trends that simple linear regression missed.

## 🛠️ Tech Stack
* **Core:** Python 3.x, `pandas`, `numpy`, `scikit-learn`
* **Materials Informatics:** `matminer`, `pymatgen`
* **Tools:** Google Colab, Kaggle API

## 📂 Repository Structure
* `Machine_Learning_for_Materials_Properties (1).ipynb`: The main notebook with data loading, feature engineering, and model tuning.
* `nanox81_train_data.csv`: Training dataset (chemical formulas & target energies).
* `nanox81_test_data.csv`: Test dataset for final evaluation.

## 💻 How to Run This Code
You can run the analysis directly in the browser. No local setup required.

1.  Click the **"Open in Colab"** badge at the top of this README.
2.  In Colab, go to **Runtime > Run all**.
3.  The notebook is configured to automatically pull data from this repository:
    ```python
    # Data loads automatically via raw GitHub links
    train_url = '[https://raw.githubusercontent.com/nhitran161/Machine-Learning-for-Materials-Properties/main/nanox81_train_data.csv](https://raw.githubusercontent.com/nhitran161/Machine-Learning-for-Materials-Properties/main/nanox81_train_data.csv)'
    ```

## 📊 Methodology
1.  **Data Preprocessing:** Converted raw chemical strings into `pymatgen` Composition objects.
2.  **Feature Engineering:** Generated 90+ descriptors including stoichiometric attributes, valence electron counts, and statistical aggregates of elemental properties (mean, deviation, range).
3.  **Model Selection:**
    * **Baseline:** Ridge Regression (Linear with L2 regularization).
    * **Advanced:** Random Forest & Histogram-based Gradient Boosting.
4.  **Optimization:** Implemented `GridSearchCV` for hyperparameter tuning and `StandardScaler` pipelines to normalize feature distributions.

---
*Created by Quynh Tran - 2025*
