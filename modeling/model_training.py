"""
Functions for training machine learning models.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import config

# Conditional imports
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    ADVANCED_MODELS = True
except ImportError:
    ADVANCED_MODELS = False


def train_model(model, X_train, y_train):
    """Helper function to fit a model."""
    model.fit(X_train, y_train)
    return model


def train_all_models(data_dict, params, adv_models_available):
    """Trains all specified models for both primary and secondary targets."""

    models_dict = {}

    # --- Primary Target: Life Expectancy ---
    print("\n--- Training Models for Life Expectancy ---")

    # Model 1: Random Forest
    print("Training Random Forest (Primary)...")
    rf_model_p = RandomForestRegressor(**params['rf'])
    models_dict['rf_p'] = train_model(rf_model_p, data_dict['X_train'], data_dict['y_train_p'])

    if adv_models_available:
        # Model 2: XGBoost
        print("Training XGBoost (Primary)...")
        xgb_model_p = XGBRegressor(**params['xgb'])
        models_dict['xgb_p'] = train_model(xgb_model_p, data_dict['X_train_scaled'], data_dict['y_train_p'])

        # Model 3: LightGBM
        print("Training LightGBM (Primary)...")
        lgb_model_p = LGBMRegressor(**params['lgb'])
        models_dict['lgb_p'] = train_model(lgb_model_p, data_dict['X_train_scaled'], data_dict['y_train_p'])

    # --- Secondary Target: Inequality Gap ---
    print("\n--- Training Models for Inequality Gap ---")

    # Model 1: Random Forest
    print("Training Random Forest (Secondary)...")
    rf_model_s = RandomForestRegressor(**params['rf'])
    models_dict['rf_s'] = train_model(rf_model_s, data_dict['X_train'], data_dict['y_train_s'])

    if adv_models_available:
        # Model 2: XGBoost
        print("Training XGBoost (Secondary)...")
        xgb_model_s = XGBRegressor(**params['xgb'])
        models_dict['xgb_s'] = train_model(xgb_model_s, data_dict['X_train_scaled'], data_dict['y_train_s'])

        # Model 3: LightGBM
        print("Training LightGBM (Secondary)...")
        lgb_model_s = LGBMRegressor(**params['lgb'])
        models_dict['lgb_s'] = train_model(lgb_model_s, data_dict['X_train_scaled'], data_dict['y_train_s'])

    print("✓ All models trained.")
    return models_dict
