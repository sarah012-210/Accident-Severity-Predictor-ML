Here is a complete `README.md` formatted specifically for your **[US Accidents Analysis & Severity Prediction](https://www.kaggle.com/code/sarahmsalah/us-accidents/edit)** notebook project on Kaggle.

---

# US Traffic Accident Severity Prediction & Hotspot Analysis

## Overview

This project provides a comprehensive machine learning pipeline and exploratory analysis for nationwide traffic accidents across the United States using the [US Accidents (2016 - 2023)](https://www.kaggle.com/code/sarahmsalah/us-accidents/edit) dataset. The primary goal is to predict traffic accident severity levels (binary classification for severe vs. non-severe accidents) and identify spatial-temporal hotspot patterns using location-density features, weather metrics, and road infrastructure signals.

---

##  Team Members

* **Mariam Ali**
* **Sarah Mohsen**
* **Sarah Sameh**

---

## 🛠️ Key Features & Methodology

1. **Target Leakage Prevention**: Excluded post-accident features such as `Distance(mi)` and `Description` to build realistic, forward-looking predictive models.
2. **Feature Engineering**:
* **Temporal Signals**: Extracted hour, day of week, month, weekend indicators, and peak rush hour identifiers.
* **Spatial Density**: Calculated dynamic spatial grid density (`cell_density`) derived from geographic coordinates ($~\text{5 km}$ grid cells).
* **Environmental & Infrastructure Factors**: Encoded weather conditions, temperature, pressure, visibility, and key road markers (junctions, traffic signals, crossings, stops).


3. **Machine Learning Pipeline**:
* Implemented `HistGradientBoostingClassifier` natively handling missing values and categorical data.
* Hyperparameter optimization via `RandomizedSearchCV` with 3-fold stratified cross-validation.
* Custom decision threshold tuning (F1-score optimization, Recall target tuning) to address significant class imbalance.



---

##  Dataset Overview

* **Source**: US Accidents (2016 – 2023)
* **Sample Size**: ~386k records (~5% stratifiable sample of 7.7M rows)
* **Target Variable**:
* `0`: Non-severe (Severity 1 & 2)
* `1`: Severe (Severity 3 & 4)



---

## Dependencies & Requirements

```bash
python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib

```

---

##  Performance & Evaluation Metrics

* **Tuned CV ROC-AUC**: ~0.7924
* **Evaluated Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall Curves.

---

##  How to Run

1. Clone this repository or open the notebook on [Kaggle](https://www.kaggle.com/code/sarahmsalah/us-accidents/edit).
2. Ensure the dataset `US_Accidents_March23.csv` is loaded into the input directory.
3. Run the notebook sequentially to execute data cleaning, feature extraction, model tuning, and evaluation.
