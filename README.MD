# UK Health Outcomes Prediction System

A machine learning system that predicts regional health outcomes across UK local authorities using publicly available demographic, economic, and environmental data to identify factors contributing to health inequalities.

## Project Overview

This project addresses a critical UK public health challenge: predicting and understanding health outcome disparities across 331 local authorities in England and Wales. By integrating multiple government datasets and applying advanced machine learning techniques, the system identifies key drivers of health outcomes and provides evidence-based policy recommendations.

**Key Achievement**: XGBoost model achieves **R² = 0.831** for life expectancy prediction with **±0.467 years MAE**, demonstrating strong predictive capability for supporting NHS resource allocation and public health interventions.

## Key Features

- **Predictive Modeling**: ML models for life expectancy (R² = 0.831) and health inequality gap (R² = 0.454)
- **Multi-Source Data Integration**: Combines 10+ government datasets at local authority level
- **Geographic Clustering**: Identifies 4 distinct regional health profiles for targeted interventions
- **What-If Scenario Analysis**: Simulates policy interventions to estimate health outcome impacts
- **Feature Engineering**: Creates composite indices (socioeconomic, healthcare access, environmental quality)
- **Model Interpretability**: SHAP values and feature importance for transparent decision-making

## Model Performance

### Primary Target (Life Expectancy)
| Model | R² Score | MAE (years) | CV R² | RMSE |
|-------|----------|-------------|-------|------|
| **XGBoost** | **0.831** | **0.467** | **0.825** | 0.644 |
| LightGBM | 0.828 | 0.485 | 0.820 | 0.650 |
| Random Forest | 0.813 | 0.464 | 0.811 | 0.676 |

### Secondary Target (Health Inequality Gap)
| Model | R² Score | MAE (years) | CV R² |
|-------|----------|-------------|-------|
| **LightGBM** | **0.454** | **1.319** | **0.285** |
| XGBoost | 0.451 | 1.325 | 0.339 |
| Random Forest | 0.436 | 1.349 | 0.361 |

### Top Predictive Features
1. **Pct_Sick_Disabled** (37.4% importance) - Percentage of population reporting long-term illness
2. **House_Price_2021** (21.0% importance) - Median house prices as economic indicator
3. **Unemployment_Rate** (6.2% importance) - Local unemployment levels
4. **Healthcare_Access_Index** (5.3% importance) - Composite measure of healthcare availability
5. **Pct_No_Qualifications** (5.2% importance) - Educational attainment levels

## Dataset

This project integrates publicly available UK government datasets. While users can manually download and place data files in datasets/raw/, the necessary datasets are already included in the repository for ease of use.

### Sources

1. **Health Outcomes** (Target Variables)
   - [ONS Life Expectancy](https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/healthandlifeexpectancies) - Life expectancy by local authority
   - [Public Health England Profiles](https://fingertips.phe.org.uk/) - Health indicators (obesity, inequality gaps)

2. **Demographic Data**
   - [Census 2021](https://www.ons.gov.uk/census) - Age structure, ethnicity, education levels
   - [Population Estimates](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration) - Mid-year population counts

3. **Economic Indicators**
   - [NOMIS Labour Market](https://www.nomisweb.co.uk/) - Employment rates, unemployment data
   - [Land Registry](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) - House price statistics
   - [Local Authority Spending](https://www.gov.uk/government/collections/local-authority-revenue-expenditure-and-financing) - Public expenditure data

4. **Environmental Data**
   - [DEFRA Air Quality](https://uk-air.defra.gov.uk/data/) - PM2.5 pollution measurements


### Data Coverage
- **Geographic Scope**: 331 local authorities (England & Wales)
- **Features**: 101 raw features → 24 engineered features after preprocessing
- **Temporal**: Latest available data (2019-2023 depending on source)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd ML_FINAL_UK_HEALTH_PROJECT
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Pipeline (Preprocessing)

Place raw datasets in `datasets/raw/` then run:

```bash
cd data_pipeline
python run_data_pipeline.py
```

This executes:
1. `preprocess.py` - Cleans and standardizes individual datasets
2. `merge_dataset.py` - Integrates all sources into master dataset

**Output**: `datasets/master/uk_health_master_dataset.csv`

### Step 2: ML Analysis Pipeline

Run the complete analysis from project root:

```bash
python main.py
```

### Step 3: Review Outputs

**Saved Models** (`outputs/models/`):
- `model_life_expectancy.pkl` - Best life expectancy predictor (XGBoost)
- `model_inequality_gap.pkl` - Best inequality gap predictor (LightGBM)
- `feature_scaler.pkl` - StandardScaler for inference
- `cluster_model.pkl` - K-Means clustering model
- `model_summary.pkl` - Performance metrics and metadata

**Visualizations** (`outputs/visualizations/`):
- `viz1_correlation_heatmap.png` - Feature correlation matrix
- `viz2_outcomes_distribution.png` - Health outcome distributions
- `viz3_predicted_vs_actual.png` - Model prediction quality
- `viz4_feature_importance.png` - Top 15 predictive features
- `viz5_cluster_analysis.png` - Regional health profiles
- `viz6_whatif_scenarios.png` - Policy intervention impacts

## Key Results

### Geographic Insights

**4 Distinct Regional Clusters Identified:**

| Cluster | Profile | Count | Avg Life Expectancy | Avg Inequality Gap |
|---------|---------|-------|---------------------|-------------------|
| **Prosperous - Low Risk** | High education, employment | 124 | 82.53 years | 5.93 years |
| **Transitioning - Mixed** | Moderate outcomes | 126 | 80.79 years | 7.52 years |
| **Stable - Good Outcomes** | Urban, diverse | 31 | 81.19 years | 6.64 years |
| **High Risk - Deprived** | Post-industrial, high deprivation | 50 | 78.91 years | 10.09 years |

### What-If Scenario Impacts

| Intervention | Parameter Change | Avg Impact | Max Benefit | Top Beneficiary |
|--------------|-----------------|------------|-------------|-----------------|
| Education Investment | +10% graduates | +0.010 years | +0.166 years | Folkestone & Hythe |
| Employment Programs | -20% unemployment | +0.161 years | +0.816 years | Stoke-on-Trent |
| Environmental Action | -15% PM2.5 pollution | +0.016 years | +0.454 years | Dartford |

### High-Risk Areas (Lowest Life Expectancy)
1. **Blackpool** - 76.5 years (inequality gap: 11.4 years)
2. **Manchester** - 77.3 years (inequality gap: 9.2 years)
3. **Knowsley** - 77.6 years (inequality gap: 9.0 years)
4. **Liverpool** - 77.6 years (inequality gap: 10.5 years)
5. **Middlesbrough** - 77.6 years (inequality gap: 9.9 years)

### Best Performing Areas (Highest Life Expectancy)
1. **Hart** - 84.9 years (inequality gap: 2.8 years)
2. **South Cambridgeshire** - 84.3 years (inequality gap: 0.7 years)
3. **Rutland** - 84.3 years (inequality gap: 7.3 years)
Reduction** - 83 areas show inequality gap >8.5 years. Targeted interventions for most deprived quintiles critical.

## Configuration

Edit `config.py` to customize:

- **Model Parameters**: Hyperparameters for RF, XGBoost, LightGBM
- **Feature Selection**: Correlation threshold, missing data handling
- **Clustering**: Number of clusters, features for segmentation
- **Scenarios**: Define custom what-if interventions
- **Outputs**: Visualization style, file paths

## License

This project is licensed under the [MIT License](./LICENSE).

## Contact

- LinkedIn: [Mahika Sehrawat](https://www.linkedin.com/in/mahika-sehrawat-412684242/)
- GitHub: [@mayhixza](https://github.com/mayhixza)
- Email: emailwithmoi@gmail.com


---

**Note**: This project is designed for research and policy analysis purposes. Model predictions should be validated by domain experts before informing real-world health policy decisions.