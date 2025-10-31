# Methodology

This document outlines the **technical workflow** and **implementation details** used in the UK Health Outcomes Prediction System. It summarizes the data preparation, feature engineering, and modeling process used to predict regional life expectancy and health inequality gaps across UK local authorities.

---

## Data Preparation

- **Data Integration**: Combined 10+ publicly available UK government datasets covering health, demographics, economics, and environment.  
- **Standardization**: Unified all sources to the **local authority level (N=331)** using ONS codes.  
- **Cleaning**: Addressed missing values using median or KNN imputation, standardized formats, and removed redundant variables.  
- **Final Dataset**: 89 raw features reduced to 24 engineered features after preprocessing and feature selection.

---

## Feature Engineering

- **Composite Indices**: Constructed 5 domain indices — socioeconomic, healthcare access, environmental quality, obesity prevalence, and education quality.  
- **Interaction Features**: Introduced 4 key interaction terms (e.g., unemployment × deprivation, education × employment).  
- **Correlation Filtering**: Removed 27 features with Pearson’s *r > 0.9* to minimize multicollinearity.  
- **Normalization**: Applied min–max scaling to composite and continuous variables for model compatibility.

---

## Model Training

- **Targets**:  
  - *Primary*: Life Expectancy (years)  
  - *Secondary*: Health Inequality Gap (years between least and most deprived quintiles)  

- **Models Used**: Random Forest, XGBoost, and LightGBM regressors.  
- **Train/Test Split**: 80/20 stratified (264 training, 67 testing samples).  
- **Cross-Validation**: 5-fold CV for robust generalization estimates.  
- **Scaling**: StandardScaler applied post-split.  
- **Ensemble Methods**: Averaged top-performing model predictions to improve stability and reduce variance.  

---

## Evaluation

- **Metrics**: R², MAE, and RMSE used for assessing predictive accuracy.  
- **Cross-Validation**: Ensured model consistency across folds.  
- **Feature Importance**: Extracted using SHAP and tree-based importance metrics for interpretability.

---

## Limitations

- **Sample Size**: 331 local authorities limit complex deep learning methods.  
- **Temporal Scope**: Cross-sectional analysis — time-series forecasting not included.  
- **Causality**: Models capture associations, not causal relationships.  
- **Imputation Bias**: Potential bias where data gaps are systematic.  
- **Data Currency**: Sources span 2019–2023; updates may affect feature validity.

---

## Reproducibility

To replicate the analysis:

```bash
python data_pipeline/run_data_pipeline.py
python main.py
```

Ensure raw datasets are located in `datasets/raw/`.  
All outputs — models, metrics, and visualizations — are generated under `outputs/`.

