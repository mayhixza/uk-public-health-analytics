"""
Functions for model interpretation (Feature Importance, SHAP).
"""

import pandas as pd
import matplotlib.pyplot as plt
import config

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def plot_feature_importance(model, feature_names, model_name):
    """Generates and saves a feature importance bar plot."""
    print(f"\n📊 Top {config.TOP_N_FEATURES} Most Important Features ({model_name}):")

    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️ Model {model_name} does not support .feature_importances_. Skipping plot.")
        return

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(feature_importance.head(config.TOP_N_FEATURES).to_string(index=False))

    # Generate visualization
    print("\n📊 Generating Visualization 4: Feature Importance...")
    top_features_imp = feature_importance.head(config.TOP_N_FEATURES)

    plt.figure(figsize=(10, 8))
    plt.barh(range(config.TOP_N_FEATURES), top_features_imp['Importance'], color='steelblue')
    plt.yticks(range(config.TOP_N_FEATURES), top_features_imp['Feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {config.TOP_N_FEATURES} Most Important Features - {model_name}',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(config.VIZ_FEATURE_IMPORTANCE, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {config.VIZ_FEATURE_IMPORTANCE}")
    plt.close()


def run_shap_analysis(model, X_data, feature_names):
    """Runs and prints SHAP analysis if available."""
    print("\n🔬 Computing SHAP values...")
    try:
        
        # SHAP expects the data in the format the model was trained on
        explainer = shap.TreeExplainer(model)
        # Use a subset for faster computation (e.g., first 100 test samples)
        shap_values = explainer.shap_values(X_data[:100])

        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': pd.DataFrame(shap_values, columns=feature_names).abs().mean(axis=0)
        }).sort_values('SHAP_Importance', ascending=False)

        print("\n🎯 Top 10 Features by SHAP Values:")
        print(shap_importance.head(10).to_string(index=False))

        # Note: We are not saving the SHAP plot as it's complex to save
        # plt.figure()
        # shap.summary_plot(shap_values, X_data[:100], plot_type="bar")
        # plt.savefig(...)

    except Exception as e:
        # print(f"⚠️ SHAP analysis failed or skipped: {e}")
        print()


def run_model_interpretation(best_model_results, data_dict, feature_cols, shap_available):
    """Runs the full interpretation pipeline."""

    model = best_model_results['model']
    model_name = best_model_results['name']

    # Plot basic feature importance
    plot_feature_importance(model, feature_cols, model_name)

    # Run SHAP if available and the model is compatible
    if shap_available and model_name != 'Random Forest':
        # Use scaled data for XGB/LGB
        X_for_shap = data_dict['X_test_scaled']
        run_shap_analysis(model, X_for_shap, feature_cols)
    elif shap_available and model_name == 'Random Forest':
        # Use original data for RF
        X_for_shap = data_dict['X_test']
        run_shap_analysis(model, X_for_shap, feature_cols)
