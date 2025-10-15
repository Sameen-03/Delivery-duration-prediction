# Delivery Time Prediction

##  Project Overview

This project predicts **delivery time** for customer orders using machine learning models trained on historical delivery data. It demonstrates a full ML pipeline from data cleaning to model evaluation.

---

##  Dataset

The dataset (`historical_data.csv`) contains information about past deliveries.

---

##  Key Features

* Cleans and preprocesses raw data.
* Handles missing values and incorrect timestamps.
* Extracts time-based features (hour, minute, weekday).
* Calculates geographical distance between restaurant and delivery location.
* Encodes categorical features using LabelEncoder or OneHotEncoder.
* Performs EDA with visualization and correlation analysis.
* Trains multiple ML models: Linear Regression, Random Forest, XGBoost, LightGBM.
* Compares model performance using **R²**, **MAE**, **RMSE**.
* Saves trained model with `joblib` for deployment.

---

##  Project Workflow

### 1. Import Libraries

Import pandas, numpy, matplotlib, seaborn, sklearn, and other required modules.

### 2. Data Loading

Read the dataset using `pd.read_csv()` and inspect its structure, missing values, and data types.

### 3. Data Cleaning

* Handle missing or invalid data.
* Fix inconsistent entries (e.g., timestamps, coordinates).
* Remove duplicates.

### 4. Feature Engineering

* Derive time-based features from `Time_Orderd` and `Time_Order_picked`.
* Compute distance using latitude-longitude pairs.
* Encode categorical columns.

### 5. Exploratory Data Analysis (EDA)

* Visualize distributions and detect outliers.
* Plot correlations between variables.
* Analyze how traffic, weather, and ratings impact delivery time.

### 6. Model Training

* Split dataset into training and testing sets.
* Train models (Linear Regression, Random Forest, XGBoost, LightGBM).
* Tune hyperparameters for optimal performance.

### 7. Model Evaluation

* Evaluate each model using R², MAE, and RMSE metrics.
* Compare predicted vs actual delivery times with plots.

### 8. Model Saving

Save the best-performing model with `joblib.dump()` for future predictions.

### 9. Future Improvements

* Implement feature selection and model interpretability (SHAP/LIME).
* Add hyperparameter optimization with GridSearchCV or Optuna.
* Build a web dashboard using Streamlit or Flask.
* Deploy the model to a cloud platform.

---

## ⚙️ Requirements

Save the following to `requirements.txt`:

```
pandas>=1.5
numpy>=1.24
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
xgboost>=1.7
lightgbm>=4.0
joblib>=1.2
jupyterlab
notebook
ipython
category_encoders
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Usage

1. Update dataset path in the notebook:

```python
path = "./data/historical_data.csv"
```

2. Run all notebook cells sequentially.
3. Review evaluation metrics and saved model outputs.

---

## Results Summary

The notebook outputs metrics comparing multiple ML models and selects the one with the highest predictive accuracy and lowest error on unseen data.
