"""
Data preprocessing and feature engineering functions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


def load_data(filepath):
    """Loads the master dataset."""
    print(f"\n📊 Loading Data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} local authorities with {len(df.columns)} features")
        # Store area names for later reference
        df['Area_Name_Stored'] = df['Area_Name']
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {filepath}")
        print("Please run merge_dataset.py first to create the master dataset.")
        exit()


def create_composite_indices(df):
    """Creates composite indices for health, socioeconomic, and environment."""
    print("📐 Creating Composite Indices...")

    # 1. Health Deprivation Score
    df['Health_Deprivation_Score'] = (
            df['Pct_Sick_Disabled'] * 0.4 +
            df['primary_obesity_all'] * 0.3 +
            (100 - df['Life_Expectancy_Avg']) * 0.3
    )

    # 2. Socioeconomic Index (higher = better)
    df['Socioeconomic_Index'] = (
            df['Pct_Level_4_Plus'] * 0.3 +
            df['Employment_Rate'] * 0.3 +
            (100 - df['Unemployment_Rate'] * 10) * 0.2 +
            (100 - df['Pct_No_Qualifications']) * 0.2
    )

    # 3. Healthcare Access Index
    df['Healthcare_Access_Index'] = (
            df['public_health_spend'] / df['Total_Population'] * 100
    )

    # 4. Environmental Quality Score
    df['Environmental_Quality'] = 100 - (df['Air_Pollution_PM25'] * 10)

    # 5. Age Dependency Score
    df['Age_Dependency'] = df['Pct_Under_15'] + df['Pct_65_Plus']

    print("✓ Created 5 composite indices")
    return df


def create_interaction_features(df):
    """Creates interaction features between key predictors."""
    print("🔗 Creating Interaction Features...")

    # 1. Education × Employment
    df['Education_Employment_Index'] = df['Pct_Level_4_Plus'] * df['Employment_Rate'] / 100

    # 2. Age × Healthcare Spending
    df['Age_Healthcare_Index'] = df['Pct_65_Plus'] * df['Healthcare_Access_Index'] / 100

    # 3. Deprivation × Healthcare
    df['Deprivation_Healthcare'] = df['Health_Deprivation_Score'] * df['Healthcare_Access_Index'] / 100

    # 4. Education × Income Proxy
    df['Education_Income_Index'] = df['Pct_Level_4_Plus'] * np.log1p(df['House_Price_2021']) / 100

    print("✓ Created 4 interaction features")
    return df


def select_features(df, exclude_features, target_primary, target_secondary):
    """Selects final features, removing irrelevant, high-missing, and correlated ones."""
    print("🔍 Selecting Relevant Features...")

    # Add dynamic exclusions
    ethnicity_cols = [col for col in df.columns if
                      any(col.startswith(p) for p in config.ETHNICITY_PREFIXES) and
                      col not in config.ETHNICITY_EXCEPTIONS]
    exclude_features.extend(ethnicity_cols)

    spending_cols = [col for col in df.columns if
                     any(col.endswith(s) for s in config.SPENDING_SUFFIXES)]
    exclude_features.extend(spending_cols)

    # Select initial feature list
    all_features = df.columns.tolist()
    feature_cols = [col for col in all_features if col not in exclude_features
                    and col not in [target_primary, target_secondary]]

    # Remove features with >30% missing values
    missing_pct = df[feature_cols].isnull().sum() / len(df) * 100
    features_to_drop_missing = missing_pct[missing_pct > 30].index.tolist()
    feature_cols = [col for col in feature_cols if col not in features_to_drop_missing]
    print(f"✓ Dropped {len(features_to_drop_missing)} features with >30% missing")

    # Remove highly correlated features (>0.9)
    # Ensure we only use numeric columns for correlation matrix
    numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    df_for_corr = df[numeric_feature_cols].fillna(df[numeric_feature_cols].median())

    corr_matrix = df_for_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    features_to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
    feature_cols = [col for col in feature_cols if col not in features_to_drop_corr]
    print(f"✓ Dropped {len(features_to_drop_corr)} highly correlated features (r>0.9)")

    print(f"✓ Selected {len(feature_cols)} features for modeling")

    # Prepare final dataset
    final_cols = feature_cols + [target_primary, target_secondary, 'Area_Name_Stored']
    df_model = df[final_cols].copy()

    # Fill remaining missing values with median
    # (As requested, this step is retained from the original script)
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns
    df_model[numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].median())

    print(f"✓ Final model dataset: {len(df_model)} rows × {len(feature_cols)} features")

    return df, df_model, feature_cols, features_to_drop_corr


def run_feature_engineering(df, exclude_features, target_primary, target_secondary):
    """Runs the full preprocessing and feature engineering pipeline."""
    df = create_composite_indices(df)
    df = create_interaction_features(df)
    df, df_model, feature_cols, features_dropped_corr = select_features(
        df, exclude_features, target_primary, target_secondary
    )
    return df, df_model, feature_cols, features_dropped_corr


def split_and_scale_data(df_model, feature_cols, target_primary, target_secondary):
    """Splits data into train/test sets and scales features."""
    print("\n📊 Splitting and Scaling Data...")

    X = df_model[feature_cols]
    y_primary = df_model[target_primary]
    y_secondary = df_model[target_secondary]
    area_names = df_model['Area_Name_Stored']

    # Train-test split (split once for all targets)
    X_train, X_test, y_train_p, y_test_p, y_train_s, y_test_s, names_train, names_test = train_test_split(
        X, y_primary, y_secondary, area_names,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return all data in a dictionary
    data_dict = {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_p": y_train_p,
        "y_test_p": y_test_p,
        "y_train_s": y_train_s,
        "y_test_s": y_test_s,
        "names_train": names_train,
        "names_test": names_test,
        "scaler": scaler
    }
    return data_dict
