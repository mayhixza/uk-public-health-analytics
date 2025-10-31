"""
Functions for saving (pickling) models and other artifacts.
"""

import pickle
import config


def save_pickle(obj, filepath):
    # saves pickle file
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"✓ Saved: {filepath}")
    except Exception as e:
        print(f"⚠️ Could not save {filepath}: {e}")


def save_all_artifacts(eval_results, clustering_results, scaler, feature_cols):
    # saves all models, scalers, and metadata to disk

    best_p = eval_results['best_primary']
    best_s = eval_results['best_secondary']

    # Save primary model (Life Expectancy)
    save_pickle(best_p['model'], config.MODEL_LE)

    # Save secondary model (Inequality Gap)
    save_pickle(best_s['model'], config.MODEL_INEQUALITY)

    save_pickle(scaler, config.SCALER)
    save_pickle(feature_cols, config.FEATURE_NAMES)
    save_pickle(clustering_results['kmeans_model'], config.CLUSTER_MODEL)

    summary_data = {
        'primary_model_name': best_p['name'],
        'primary_r2': best_p['R² Score'],
        'primary_mae': best_p['MAE'],
        'secondary_model_name': best_s['name'],
        'secondary_r2': best_s['R² Score'],
        'secondary_mae': best_s['MAE'],
        'n_features': len(feature_cols),
        'n_areas': len(clustering_results['df_model']),
        'cluster_names': clustering_results['cluster_names']
    }
    save_pickle(summary_data, config.MODEL_SUMMARY)
    print(f"All models and artifacts saved in {config.MODEL_DIR}")
