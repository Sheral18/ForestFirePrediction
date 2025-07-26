

# Algerian Forest Fires: FWI Prediction

## Overview

The **Algerian Forest Fires** project is an end-to-end machine learning application designed to predict the Fire Weather Index (FWI) for two Algerian regions: Bejaia and Sidi-Bel Abbes. FWI is a numerical rating of fire intensity risk based on weather conditions, and accurate prediction can help forest management authorities in early fire detection and prevention.

This project demonstrates the complete ML pipeline from data collection and preprocessing to model development, evaluation, and prediction.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Getting Started](#getting-started)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [Contact](#contact)

---

## Project Motivation

Forest fires can cause significant ecological and economic damage in Algeria. Predicting the Fire Weather Index (FWI) can support timely interventions and resource allocation. This project aims to:

- Build a reliable predictive model for FWI using meteorological and fire-related features.
- Provide actionable insights for fire management authorities and researchers.

---

## Dataset

### Source

- **Regions:** Bejaia and Sidi-Bel Abbes (Algeria)
- **Time Period:** June to September 2012
- **Attributes:**  
    - `day`, `month`, `year`: Date information  
    - `Temperature`: in °C  
    - `RH`: Relative Humidity (%)  
    - `Ws`: Wind Speed (km/h)  
    - `Rain`: Rainfall (mm)  
    - `FFMC`, `DMC`, `DC`, `ISI`, `BUI`, `FWI`: Fire weather index components  
    - `Classes`: Fire occurrence label (`fire` or `not fire`)

### Structure

The dataset contains daily records with meteorological and fire indices as features and FWI as the target variable.

---

## Project Pipeline

1. **Data Collection & Integration:**  
   - Combined data from Bejaia and Sidi-Bel Abbes regions.

2. **Data Preprocessing:**  
   - Handling missing values and outliers  
   - Feature engineering (date parsing, region encoding)  
   - Data normalization/scaling

3. **Exploratory Data Analysis (EDA):**  
   - Feature distributions and correlations  
   - Fire vs. non-fire days comparison

4. **Model Selection:**  
   - Regression models: Linear Regression, Random Forest, Gradient Boosting, etc.

5. **Model Training & Evaluation:**  
   - Train/test split  
   - Model tuning using cross-validation  
   - Evaluation metrics: RMSE, MAE, R² Score

6. **Prediction & Interpretation:**  
   - FWI prediction on unseen data  
   - Feature importance analysis

7. **Deployment (Optional):**  
   - Model serialization (joblib/pickle)  
   - Simple web interface or API for predictions

---

## Getting Started

### Prerequisites

- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

### Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/Sheral18/ForestFirePrediction.git
   cd algerian-forest-fires
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the project:**
   ```
   python3 application.py
   ```

### Files & Directories

- `data/`: Contains dataset CSV files
- `notebooks/`: Jupyter notebooks for EDA and modeling
- `src/`: Source code for data processing, training, and prediction
- `models/`: Saved models
- `main.py`: Main script to run the pipeline

---

## Modeling Approach

- **Feature Engineering:**  
  - Encoded categorical variables  
  - Created derived features (e.g., seasonal indicators)

- **Model Selection:**  
  - Compared baseline and advanced regression models

- **Evaluation:**  
  - Used cross-validation and multiple metrics for robust assessment

---

## Results

- The [best model] achieved:
  - **RMSE:** XX.XX
  - **MAE:** XX.XX
  - **R² Score:** X.XX

- Feature importance analysis showed that Temperature ,RH,Ws Rain,FFMC,DMC,ISI,Classes,Regions were the most influential in predicting FWI.

---

## Future Work

- Incorporate more recent data for generalization.
- Integrate satellite imagery or remote sensing features.
- Develop an interactive dashboard for real-time prediction.
- Explore deep learning models for improved performance.

---

## References

- [UCI Machine Learning Repository: Algerian Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset)
- Canadian Forest Fire Weather Index System

---

## Contact

For questions or suggestions, reach out to sheral.waskar2002@gmail.com.

---




