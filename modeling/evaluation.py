"""
Functions for evaluating model performance and generating visuals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import config


def get_model_predictions(model, X_test):
    return model.predict(X_test)


def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    predictions = get_model_predictions(model, X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        cv_r2_mean = np.mean(cv_scores)
    except Exception as e:
        print(f"Warning: Could not run CV for {model.__class__.__name__}. Setting CV R² to 0. Error: {e}")
        cv_r2_mean = 0.0

    return {
        "model": model,
        "predictions": predictions,
        "MAE": mae,
        "RMSE": rmse,
        "R² Score": r2,
        "CV_R2_Mean": cv_r2_mean
    }


def evaluate_all_models(models_dict, data_dict, adv_models_available):

    results_primary = []
    results_secondary = []

    # --- Evaluate Primary Models ---
    print("\n--- Evaluating Life Expectancy Models ---")

    # RF (Primary)
    rf_p_results = evaluate_model_performance(
        models_dict['rf_p'],
        data_dict['X_train'], data_dict['y_train_p'],
        data_dict['X_test'], data_dict['y_test_p']
    )
    rf_p_results['name'] = 'Random Forest'
    results_primary.append(rf_p_results)
    print(f"Random Forest (P): R²={rf_p_results['R² Score']:.3f} | CV R²={rf_p_results['CV_R2_Mean']:.3f}")

    if adv_models_available:
        # XGB (Primary)
        xgb_p_results = evaluate_model_performance(
            models_dict['xgb_p'],
            data_dict['X_train_scaled'], data_dict['y_train_p'],
            data_dict['X_test_scaled'], data_dict['y_test_p']
        )
        xgb_p_results['name'] = 'XGBoost'
        results_primary.append(xgb_p_results)
        print(f"XGBoost (P): R²={xgb_p_results['R² Score']:.3f} | CV R²={xgb_p_results['CV_R2_Mean']:.3f}")

        # LGB (Primary)
        lgb_p_results = evaluate_model_performance(
            models_dict['lgb_p'],
            data_dict['X_train_scaled'], data_dict['y_train_p'],
            data_dict['X_test_scaled'], data_dict['y_test_p']
        )
        lgb_p_results['name'] = 'LightGBM'
        results_primary.append(lgb_p_results)
        print(f"LightGBM (P): R²={lgb_p_results['R² Score']:.3f} | CV R²={lgb_p_results['CV_R2_Mean']:.3f}")

    # --- Evaluate Secondary Models ---
    print("\n--- Evaluating Inequality Gap Models ---")

    # RF (Secondary)
    rf_s_results = evaluate_model_performance(
        models_dict['rf_s'],
        data_dict['X_train'], data_dict['y_train_s'],
        data_dict['X_test'], data_dict['y_test_s']
    )
    rf_s_results['name'] = 'Random Forest'
    results_secondary.append(rf_s_results)
    print(f"Random Forest (S): R²={rf_s_results['R² Score']:.3f} | CV R²={rf_s_results['CV_R2_Mean']:.3f}")

    if adv_models_available:
        # XGB (Secondary)
        xgb_s_results = evaluate_model_performance(
            models_dict['xgb_s'],
            data_dict['X_train_scaled'], data_dict['y_train_s'],
            data_dict['X_test_scaled'], data_dict['y_test_s']
        )
        xgb_s_results['name'] = 'XGBoost'
        results_secondary.append(xgb_s_results)
        print(f"XGBoost (S): R²={xgb_s_results['R² Score']:.3f} | CV R²={xgb_s_results['CV_R2_Mean']:.3f}")

        # LGB (Secondary)
        lgb_s_results = evaluate_model_performance(
            models_dict['lgb_s'],
            data_dict['X_train_scaled'], data_dict['y_train_s'],
            data_dict['X_test_scaled'], data_dict['y_test_s']
        )
        lgb_s_results['name'] = 'LightGBM'
        results_secondary.append(lgb_s_results)
        print(f"LightGBM (S): R²={lgb_s_results['R² Score']:.3f} | CV R²={lgb_s_results['CV_R2_Mean']:.3f}")

    # --- Summarize and Find Best ---

    # Primary
    results_df_p = pd.DataFrame(results_primary)[['name', 'MAE', 'RMSE', 'R² Score', 'CV_R2_Mean']]
    best_idx_p = results_df_p['R² Score'].idxmax()
    best_primary_model_results = results_primary[best_idx_p]
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - LIFE EXPECTANCY")
    print("=" * 80)
    print(results_df_p.to_string(index=False))
    print(f"\n🏆 Best Model (Primary): {best_primary_model_results['name']}")

    # Secondary
    results_df_s = pd.DataFrame(results_secondary)[['name', 'MAE', 'RMSE', 'R² Score', 'CV_R2_Mean']]
    best_idx_s = results_df_s['R² Score'].idxmax()
    best_secondary_model_results = results_secondary[best_idx_s]
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - INEQUALITY GAP")
    print("=" * 80)
    print(results_df_s.to_string(index=False))
    print(f"\n🏆 Best Model (Secondary): {best_secondary_model_results['name']}")

    return {
        "primary_results": results_primary,
        "secondary_results": results_secondary,
        "primary_df": results_df_p,
        "secondary_df": results_df_s,
        "best_primary": best_primary_model_results,
        "best_secondary": best_secondary_model_results,
        "y_test_p": data_dict['y_test_p'],
        "y_test_s": data_dict['y_test_s']
    }


def plot_predicted_vs_actual(eval_results):
    """Generates and saves the Predicted vs. Actual scatter plot."""
    print("\n📊 Generating Visualization 3: Predicted vs Actual...")

    best_p = eval_results['best_primary']
    best_s = eval_results['best_secondary']
    y_test_p = eval_results['y_test_p']
    y_test_s = eval_results['y_test_s']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Life Expectancy
    axes[0].scatter(y_test_p, best_p['predictions'], alpha=0.6, s=50)
    axes[0].plot([y_test_p.min(), y_test_p.max()], [y_test_p.min(), y_test_p.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Life Expectancy (years)', fontsize=12)
    axes[0].set_ylabel('Predicted Life Expectancy (years)', fontsize=12)
    axes[0].set_title(f"Life Expectancy: {best_p['name']} (R²={best_p['R² Score']:.3f})",
                      fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Inequality Gap
    axes[1].scatter(y_test_s, best_s['predictions'], alpha=0.6, s=50, color='orange')
    axes[1].plot([y_test_s.min(), y_test_s.max()], [y_test_s.min(), y_test_s.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Inequality Gap (years)', fontsize=12)
    axes[1].set_ylabel('Predicted Inequality Gap (years)', fontsize=12)
    axes[1].set_title(f"Inequality Gap: {best_s['name']} (R²={best_s['R² Score']:.3f})",
                      fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.VIZ_PRED_V_ACTUAL, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {config.VIZ_PRED_V_ACTUAL}")
    plt.close()


def plot_all_evaluation_visuals(eval_results):    #Generates all evaluation visualizations
    plot_predicted_vs_actual(eval_results)