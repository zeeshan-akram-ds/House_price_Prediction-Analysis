# Ames Housing Price Prediction

This project focuses on building a robust regression model to predict housing prices using the Ames dataset. It explores end-to-end machine learning processes including data preprocessing, feature engineering, feature selection, model building, evaluation, and deployment using Streamlit.

---

## Project Objectives

- Understand and explore the Ames housing dataset
- Engineer meaningful features using domain knowledge
- Perform smart feature selection to reduce dimensionality
- Train and evaluate various machine learning models
- Use explainability tools (SHAP, LIME) to interpret model behavior
- Build and deploy an interactive Streamlit app for predictions and analysis
---

## Dataset

The dataset used is the **Ames Housing dataset**, which contains 82 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.

- **Target Variable:** `SalePrice` (the property's sale price)
- **Total Features:** 82 (numeric, categorical, ordinal)
- **Rows:** 2,930 (after cleaning)
- **Source:** [Ames Housing Dataset on Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

---

## Data Preprocessing

Steps taken to clean and prepare the data:

- Removed outliers using domain insights and visualizations
- Handled missing values:
  - Categorical: filled with `"None"` or `"NA"` where meaningful
  - Numerical: filled with `0` or group-based medians
- Performed log transformation on `SalePrice` to normalize distribution
- Applied `RobustScaler` on numerical features to reduce the impact of outliers
---

## Exploratory Data Analysis (EDA)

Performed comprehensive EDA to understand data distributions, correlations, and potential feature importance.

### Key Insights:

- **SalePrice** is **right-skewed**, hence log-transformed
- **OverallQual**, **GrLivArea**, and **GarageCars** show strong correlation with **SalePrice**
- Certain neighborhoods have consistently higher or lower prices
- Some features like **Pool Area** and **3Ssn Porch** have very low variance

### Visualizations:

- **Distribution plots** for `SalePrice` (original and log scale)
- **Heatmaps** for correlation analysis
- **Boxplots** to compare `SalePrice` across categorical features (e.g., `OverallQual`, `Neighborhood`)
- **Scatter plots** for numerical predictors like `GrLivArea`, `Total Bsmt SF`, `1st Flr SF`

---

## Feature Engineering

Created domain-informed features with strong correlation to target:

```python
df['Total_SF'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']
df['Total_Bathrooms'] = (df['Full Bath'] + 0.5 * df['Half Bath'] +
                         df['Bsmt Full Bath'] + 0.5 * df['Bsmt Half Bath'])
df['House_Age'] = df['Yr Sold'] - df['Year Built']
df['Remod_Age'] = df['Yr Sold'] - df['Year Remod/Add']
df['Garage_Age'] = df['Yr Sold'] - df['Garage Yr Blt']
df['Total_Porch_SF'] = df[['Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch']].sum(axis=1)
df['Has_Pool'] = df['Pool Area'].apply(lambda x: 1 if x > 0 else 0)
df['Has_2ndFlr'] = df['2nd Flr SF'].apply(lambda x: 1 if x > 0 else 0)
df['Has_Fireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df['Has_FinishedBsmt'] = df['BsmtFin SF 1'].apply(lambda x: 1 if x > 0 else 0)
```
---  

## Feature Selection

Performed systematic reduction of input features to avoid redundancy and overfitting:

-  Dropped high-cardinality features (more than 30 unique values)
-  Applied `VarianceThreshold` with 1% cutoff to remove near-constant features
-  Removed highly correlated features (Pearson correlation > 0.95)
-  Used `Mutual Information Regression` to select the top 35 most informative numeric features
---

## Modeling

Explored multiple regression models to identify the best-performing algorithm:

-  **Linear Regression**  
  Baseline model with R² ≈ 0.91

-  **Ridge, LassoCV, RidgeCV**  
  Regularization improved generalization slightly over baseline

-  **Random Forest Regressor**  
  Non-linear ensemble model → R² ≈ 0.92

-  **XGBoost Regressor**  
  Best performing model → R² ≈ 0.93

-  **SVM (RBF Kernel)**  
  Underfit the data, performed poorly

-  **AdaBoost, ExtraTrees, Gradient Boosting**  
  Results lower than XGBoost

-  **Stacking Regressor**  
  Combined XGBoost, RFC, Gradient Boosting (meta-model: Ridge) → comparable but slightly lower than XGB

-  **Voting Regressor**  
  Combined weak and strong models, but performance varied based on member choice
---

## Final Evaluation (XGBoost)

Evaluated the best model using both log-transformed and original target scales.

### Log Scale Metrics:

- **RMSE:** 0.1126  
- **MAE:** 0.0790  
- **R²:** 0.9314  

### Original Scale Metrics:

- **RMSE:** 23,835.18  
- **MAE:** 14,862.42  
- **R²:** 0.9291  
- **Adjusted R²:** 0.9246  

---

### 10-Fold Cross Validation:

| Metric     | Log Scale | Original Scale |
|------------|-----------|----------------|
| **RMSE**   | 0.1287    | 23,036.51      |
| **MAE**    | 0.0829    | 14,398.45      |
| **R²**     | 0.9002    | 0.9168         |
| **Adj R²** | –         | 0.9158         |
---

## Explainability

Model interpretability was achieved using both global and local tools:

-  **SHAP (SHapley Additive exPlanations)**  
  Used in Jupyter to visualize global feature importance and individual predictions.
  
-  **LIME (Local Interpretable Model-agnostic Explanations)**  
  Integrated into the Streamlit app to explain individual predictions in real-time, enhancing user trust and transparency.
---

## Streamlit App

An interactive web app was developed to demonstrate model predictions and diagnostics:

### Key Features

- **Prediction Tab:**  
  Users can input house features and receive predicted sale price with **LIME-based explanation**.

- **Model Evaluation Tab:**  
  Displays key performance metrics and diagnostic plots:
  - Predicted vs Actual plot
  - Residuals vs Predicted plot

- **ToolBox:**  
  To Understand about input columns.
### Saved Artifacts

The following files are exported and used in the app:

- `xgboost_model.pkl` – Trained XGBoost model  
- `robust_scaler.pkl` – Fitted robust scaler  
- `input_columns.json` – Required input feature list  
- `X_test.csv`, `y_test.csv` – Test data for evaluation and explainability
---

## Conclusion

This project demonstrated a complete end-to-end pipeline for house price prediction using the Ames Housing dataset. The key stages included:

- Thorough **data cleaning** and **EDA**
- Creation of powerful **domain-informed features**
- Robust **feature selection**
- Testing of multiple **machine learning models**, with **XGBoost** yielding the best performance
- Deep **model evaluation** using both log and original scales
- Clear **model interpretability** with SHAP and LIME
- Deployment of a user-friendly **Streamlit app** for real-time predictions

---

## Acknowledgements

- Ames Housing dataset courtesy of [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- Inspired by best practices in **data science workflows** and **Kaggle notebooks**

---
