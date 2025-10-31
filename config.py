import numpy as np

MASTER_DATASET = 'datasets/master/uk_health_master_dataset.csv'

OUTPUT_DIR = 'outputs'
MODEL_DIR = f'{OUTPUT_DIR}/models'
VIZ_DIR = f'{OUTPUT_DIR}/visualizations'

MODEL_LE = f'{MODEL_DIR}/model_life_expectancy.pkl'
MODEL_INEQUALITY = f'{MODEL_DIR}/model_inequality_gap.pkl'
SCALER = f'{MODEL_DIR}/feature_scaler.pkl'
FEATURE_NAMES = f'{MODEL_DIR}/feature_names.pkl'
CLUSTER_MODEL = f'{MODEL_DIR}/cluster_model.pkl'
MODEL_SUMMARY = f'{MODEL_DIR}/model_summary.pkl'

VIZ_CORR_HEATMAP = f'{VIZ_DIR}/viz1_correlation_heatmap.png'
VIZ_DISTRIBUTION = f'{VIZ_DIR}/viz2_outcomes_distribution.png'
VIZ_PRED_V_ACTUAL = f'{VIZ_DIR}/viz3_predicted_vs_actual.png'
VIZ_FEATURE_IMPORTANCE = f'{VIZ_DIR}/viz4_feature_importance.png'
VIZ_CLUSTER = f'{VIZ_DIR}/viz5_cluster_analysis.png'
VIZ_WHAT_IF = f'{VIZ_DIR}/viz6_whatif_scenarios.png'

TARGET_PRIMARY = 'Life_Expectancy_Avg'
TARGET_SECONDARY = 'Avg_Inequality_Gap'

EXCLUDE_FEATURES = [
    'LA_Code', 'Area_Name', 'Area_Name_Stored', 'Region_Type', 'Urban_Rural',
    'Life_Expectancy_Male', 'Life_Expectancy_Female',
    'Life_Expectancy_Gap',
    'Male_Inequality_Gap', 'Female_Inequality_Gap',
    'Male_Most_Deprived_LE', 'Male_Least_Deprived_LE',
    'Female_Most_Deprived_LE', 'Female_Least_Deprived_LE',
    'House_Price_2020', 'House_Price_Change_Pct',
]

ETHNICITY_PREFIXES = ['Asian_', 'Black_', 'Mixed_', 'Other_', 'White_']
ETHNICITY_EXCEPTIONS = ['Pct_White_British', 'Pct_Ethnic_Minority']
SPENDING_SUFFIXES = ['_pct', '_ratio']

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_PARAMS = {
    'rf': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'xgb': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'lgb': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }
}

CLUSTERING_FEATURES = [
    'Life_Expectancy_Avg', 'Avg_Inequality_Gap', 'Socioeconomic_Index',
    'Health_Deprivation_Score', 'Healthcare_Access_Index', 'Environmental_Quality',
    'Pct_Level_4_Plus', 'Employment_Rate', 'Unemployment_Rate'
]
N_CLUSTERS = 4

SNS_STYLE = "whitegrid"
FIGSIZE = (12, 6)
TOP_N_FEATURES = 15 # For feature importance plot
