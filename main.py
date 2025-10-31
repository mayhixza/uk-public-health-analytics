import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle

import config
from data.preprocessing import (load_data, run_feature_engineering, split_and_scale_data)
from eda.exploratory_analysis import run_exploratory_data_analysis
from modeling.model_training import train_all_models
from modeling.evaluation import evaluate_all_models, plot_all_evaluation_visuals
from modeling.interpretation import run_model_interpretation
from clustering_analysis import run_all_clustering
from what_if_scenarios import run_all_scenarios
from policy_recommendations import generate_policy_recommendations
from save_artifacts import save_all_artifacts

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    ADVANCED_MODELS = True
except ImportError:
    ADVANCED_MODELS = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def setup_environment():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.VIZ_DIR, exist_ok=True)
    print(f"Output directories: {config.MODEL_DIR} and {config.VIZ_DIR}")

    sns.set_style(config.SNS_STYLE)
    plt.rcParams['figure.figsize'] = config.FIGSIZE

    print("\n" + "=" * 80)
    print("UK HEALTH OUTCOMES PREDICTION SYSTEM - COMPLETE ANALYSIS")
    print("=" * 80)
    if not ADVANCED_MODELS:
        print("⚠️ XGBoost/LightGBM not available. Using Random Forest only.")
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP not available. Will use basic feature importance.")


def print_final_summary(eval_results, clustering_results, scenario_results):
    best_p = eval_results['best_primary']
    best_s = eval_results['best_secondary']
    df_model = clustering_results['df_model']
    n_clusters = clustering_results['n_clusters']

    impacts = scenario_results['impact_means']

    summary_text = f"""
{"=" * 80}
PROJECT COMPLETION SUMMARY
{"=" * 80}

ANALYSIS COMPLETED SUCCESSFULLY

DATA PROCESSING:
   ✓ Analyzed {len(df_model)} UK local authorities
   ✓ Engineered {len(eval_results['feature_cols'])} predictive features
   ✓ Dropped {eval_results['features_dropped_corr']} highly correlated features

MACHINE LEARNING MODELS:

   PRIMARY TARGET (Life Expectancy):
   ✓ Best Model: {best_p['name']}
   ✓ R² Score: {best_p['R² Score']:.3f}
   ✓ MAE: ±{best_p['MAE']:.3f} years
   ✓ CV R² Score: {best_p['CV_R2_Mean']:.3f}

   SECONDARY TARGET (Inequality Gap):
   ✓ Best Model: {best_s['name']}
   ✓ R² Score: {best_s['R² Score']:.3f}
   ✓ MAE: ±{best_s['MAE']:.3f} years

CLUSTERING ANALYSIS:
   ✓ Created {n_clusters} distinct regional clusters
   ✓ Identified area profiles for targeted interventions

WHAT-IF SCENARIOS:
   ✓ Simulated 4 policy interventions
   ✓ Health spending impact: +{impacts.get('S1', 0):.3f} years (10% increase)
   ✓ Education impact: +{impacts.get('S2', 0):.3f} years (10% increase)
   ✓ Employment impact: +{impacts.get('S3', 0):.3f} years (20% reduction in unemployment)
   ✓ Environmental impact: +{impacts.get('S4', 0):.3f} years (15% pollution reduction)

VISUALIZATIONS CREATED:
   ✓ {config.VIZ_CORR_HEATMAP}
   ✓ {config.VIZ_DISTRIBUTION}
   ✓ {config.VIZ_PRED_V_ACTUAL}
   ✓ {config.VIZ_FEATURE_IMPORTANCE}
   ✓ {config.VIZ_CLUSTER}
   ✓ {config.VIZ_WHAT_IF}

{"=" * 80}
ANALYSIS COMPLETE - All outputs saved!
{"=" * 80}
"""
    print(summary_text)


def main():
    setup_environment()
    df = load_data(config.MASTER_DATASET)
    print("\n--- Preprocessing & Feature Engineering ---")
    df, df_model, feature_cols, features_dropped_corr = run_feature_engineering(
        df,
        config.EXCLUDE_FEATURES,
        config.TARGET_PRIMARY,
        config.TARGET_SECONDARY
    )

    print("\n--- Exploratory Data Analysis ---")
    run_exploratory_data_analysis(df, df_model, config.TARGET_PRIMARY, config.TARGET_SECONDARY)

    data_dict = split_and_scale_data(df_model, feature_cols, config.TARGET_PRIMARY, config.TARGET_SECONDARY
    )

    print("\n--- Model Training ---")
    models_dict = train_all_models(data_dict, config.MODEL_PARAMS, ADVANCED_MODELS)

    print("\n--- Model Evaluation ---")
    eval_results = evaluate_all_models(models_dict, data_dict, ADVANCED_MODELS
    )
    eval_results['feature_cols'] = feature_cols
    eval_results['features_dropped_corr'] = len(features_dropped_corr)

    plot_all_evaluation_visuals(eval_results)

    # Interpretation
    print("\n--- Model Interpretation ---")
    run_model_interpretation( eval_results['best_primary'], data_dict, feature_cols, SHAP_AVAILABLE
    )

    print("\n--- Clustering Analysis ---")
    clustering_results = run_all_clustering( df, df_model, config.CLUSTERING_FEATURES, config.N_CLUSTERS
    )
    # Updating df and df_model with cluster info
    df = clustering_results['df']
    df_model = clustering_results['df_model']

    print("\n--- What-If Scenario Analysis ---")
    scenario_results = run_all_scenarios( df_model, data_dict['scaler'], eval_results['best_primary']['model'], feature_cols
    )
    
    df_model = scenario_results['df_model']

    print("\n--- Policy Recommendations ---")
    policy_text = generate_policy_recommendations(
        df_model,
        eval_results,
        scenario_results,
        config.TARGET_PRIMARY,
        config.TARGET_SECONDARY
    )
    print(policy_text)

    print("\n--- Saving Artifacts ---")
    save_all_artifacts(
        eval_results,
        clustering_results,
        data_dict['scaler'],
        feature_cols
    )
    
    print_final_summary(eval_results, clustering_results, scenario_results)


if __name__ == "__main__":
    main()
