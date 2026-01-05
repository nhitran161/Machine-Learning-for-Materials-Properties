# Machine Learning for Materials Properties

This repository contains notebooks for machine learning applications in materials science.  
Click the badge below to open the main notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhitran161/Machine-Learning-for-Materials-Properties/blob/main/NANO281_lab3_Quynh_Tran.ipynb)

## 📌 Project Overview
This project leverages machine learning to predict the **cohesive energy per atom ($E_c$)** of materials using the **MatPES r2SCAN dataset**. Cohesive energy is a critical indicator of thermodynamic stability, essential for accelerating the discovery of novel semiconductors and structural materials.

By engineering 90+ compositional and structural features and optimizing Ridge Regression and Gradient Boosting models, this pipeline achieves a prediction error (MSE) of **0.072 eV/atom**, significantly outperforming baseline averages.

## 🚀 Key Results
* **Best Model:** Gradient Boosting Regressor (tuned)
* **Performance:** 0.072 eV/atom (Mean Squared Error)
* **Improvement:** 88% reduction in error compared to baseline mean predictions.
* **Key Features:** Identified Electronegativity (average & difference) and Atomic Radius as the strongest predictors of material stability.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
* **Materials Informatics:** `matminer`, `pymatgen`
* **Tools:** Google Colab, Kaggle API

## 📂 Repository Structure
* `Machine_Learning_for_Materials_Properties.ipynb`: The main Jupyter Notebook containing data loading, EDA, feature engineering, and model training.
* `nanox81_train_data.csv`: Training dataset containing chemical formulas and target cohesive energies.
* `nanox81_test_data.csv`: Test dataset for final model evaluation.

## 💻 How to Run This Code
You can run this project directly in Google Colab without downloading any files manually.

1.  Open the `Machine_Learning_for_Materials_Properties.ipynb` file in this repository.
2.  Click the "Open in Colab" badge (if available) or copy the URL into [Google Colab](https://colab.research.google.com/).
3.  Run the cells! The notebook is configured to pull the data directly from this GitHub repository:
    ```python
    # Data is loaded automatically via raw GitHub links
    train_url = '[https://raw.githubusercontent.com/nhitran161/Machine-Learning-for-Materials-Properties/main/nanox81_train_data.csv](https://raw.githubusercontent.com/nhitran161/Machine-Learning-for-Materials-Properties/main/nanox81_train_data.csv)'
    ```

## 📊 Methodology
1.  **Data Preprocessing:** Cleaned chemical formula strings and converted them into composition objects using `pymatgen`.
2.  **Feature Engineering:** Utilized `matminer` to generate 90+ descriptors, including:
    * Stoichiometric attributes (number of elements, L2 norm).
    * Elemental statistics (mean/std dev of atomic mass, melting point, electronegativity).
3.  **Model Selection:**
    * **Baseline:** Ridge Regression (Linear with L2 regularization).
    * **Advanced:** Random Forest & Histogram-based Gradient Boosting.
4.  **Optimization:** Used `GridSearchCV` and `Pipeline` scaling (`StandardScaler`) to fine-tune hyperparameters.

## 📈 Visualizations
(You can add screenshots of your correlation heatmaps or parity plots here later!)

---
*Created by Quynh Tran - 2025*
