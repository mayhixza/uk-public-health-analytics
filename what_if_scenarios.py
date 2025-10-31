"""
Functions for running "What-If" policy scenario simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import config


def run_scenario(df_model, scaler, model, feature_cols, scenario_name, changes_dict):
    """
    Runs a single what-if scenario.

    Args:
    - df_model: The dataframe with model data.
    - scaler: The fitted StandardScaler.
    - model: The trained predictive model.
    - feature_cols: List of features used by the model.
    - scenario_name: Name for printing (e.g., "Scenario 1").
    - changes_dict: A dictionary of functions to apply, e.g.,
      {'public_health_spend': lambda x: x * 1.1}
    """
    print(f"\n{'=' * 60}")
    print(f"{scenario_name.upper()}")
    print(f"{'=' * 60}")

    X_scenario = df_model[feature_cols].copy()

    # Apply initial changes
    for feature, change_func in changes_dict.items():
        if feature in X_scenario.columns:
            X_scenario[feature] = change_func(X_scenario[feature])
        else:
            print(f"⚠️ Feature {feature} not in model features. Skipping change.")
            return None

    # Recalculate dependent features
    # This part can be updated based on known dependencies

    # Dependency: Healthcare_Access_Index
    if 'Healthcare_Access_Index' in X_scenario.columns and 'public_health_spend' in changes_dict:
        X_scenario['Healthcare_Access_Index'] = (
                X_scenario['public_health_spend'] / df_model['Total_Population'] * 100
        )

    # Dependency: Socioeconomic_Index
    if 'Socioeconomic_Index' in X_scenario.columns:
        if 'Pct_Level_4_Plus' in changes_dict:
            X_scenario['Socioeconomic_Index'] = (
                    X_scenario['Pct_Level_4_Plus'] * 0.3 +
                    X_scenario['Employment_Rate'] * 0.3 +
                    (100 - X_scenario['Unemployment_Rate'] * 10) * 0.2 +
                    (100 - df_model['Pct_No_Qualifications']) * 0.2
            )
        if 'Unemployment_Rate' in changes_dict:  # Assumes Employment_Rate also changed
            X_scenario['Socioeconomic_Index'] = (
                    df_model['Pct_Level_4_Plus'] * 0.3 +
                    X_scenario['Employment_Rate'] * 0.3 +
                    (100 - X_scenario['Unemployment_Rate'] * 10) * 0.2 +
                    (100 - df_model['Pct_No_Qualifications']) * 0.2
            )

    # Dependency: Deprivation_Healthcare
    if 'Deprivation_Healthcare' in X_scenario.columns and 'Healthcare_Access_Index' in X_scenario.columns:
        X_scenario['Deprivation_Healthcare'] = (
                df_model['Health_Deprivation_Score'] * X_scenario['Healthcare_Access_Index'] / 100
        )

    # Dependency: Education_Employment_Index
    if 'Education_Employment_Index' in X_scenario.columns:
        if 'Pct_Level_4_Plus' in changes_dict:
            X_scenario['Education_Employment_Index'] = (
                    X_scenario['Pct_Level_4_Plus'] * df_model['Employment_Rate'] / 100
            )
        if 'Employment_Rate' in changes_dict:
            X_scenario['Education_Employment_Index'] = (
                    df_model['Pct_Level_4_Plus'] * X_scenario['Employment_Rate'] / 100
            )

    # Dependency: Education_Income_Index
    if 'Education_Income_Index' in X_scenario.columns and 'Pct_Level_4_Plus' in changes_dict:
        X_scenario['Education_Income_Index'] = (
                X_scenario['Pct_Level_4_Plus'] * np.log1p(df_model['House_Price_2021']) / 100
        )

    # Dependency: Environmental_Quality
    if 'Environmental_Quality' in X_scenario.columns and 'Air_Pollution_PM25' in changes_dict:
        X_scenario['Environmental_Quality'] = 100 - (X_scenario['Air_Pollution_PM25'] * 10)

    pred_scenario = model.predict(scaler.transform(X_scenario[feature_cols]))
    impact = pred_scenario - df_model['Predicted_LE']

    print(f"Average impact: +{impact.mean():.3f} years")
    print(f"Max benefit: +{impact.max():.3f} years")

    # Top beneficiaries
    df_model[f'Impact_{scenario_name}'] = impact
    top_benefit = df_model.nlargest(5, f'Impact_{scenario_name}')[
        ['Area_Name_Stored', 'Life_Expectancy_Avg', f'Impact_{scenario_name}']
    ]
    print("\nTop 5 beneficiaries:")
    for _, row in top_benefit.iterrows():
        print(
            f"  {row['Area_Name_Stored']}: +{row[f'Impact_{scenario_name}']:.3f} years (current: {row['Life_Expectancy_Avg']:.2f})")

    return impact


def plot_what_if_summary(impact_means):
    # generates and saves the summary bar chart for all scenarios
    print("\n📊 Generating Visualization 6: What-If Scenarios...")

    scenario_names = list(impact_means.keys())
    scenario_impacts = list(impact_means.values())

    # mapping names for better plotting
    name_map = {
        'S1': 'Health Spending\n+10%',
        'S2': 'Education\n+10%',
        'S3': 'Unemployment\n-20%',
        'S4': 'Air Quality\n+15%'
    }
    plot_names = [name_map.get(n, n) for n in scenario_names]

    if not scenario_impacts:
        print("⚠️ No scenario impacts to plot.")
        return

    plt.figure(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    bars = plt.bar(plot_names, scenario_impacts, color=colors[:len(plot_names)], alpha=0.8, edgecolor='black')

    # adds value labels on bars
    for bar, impact in zip(bars, scenario_impacts):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'+{impact:.3f}\nyears', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Average Life Expectancy Increase (years)', fontsize=12)
    plt.title('Policy Impact Simulation: What-If Scenarios', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(bottom=0)  # Ensure y-axis starts at 0
    plt.tight_layout()
    plt.savefig(config.VIZ_WHAT_IF, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {config.VIZ_WHAT_IF}")
    plt.close()


def run_all_scenarios(df_model, scaler, model, feature_cols):
    """Runs all defined what-if scenarios."""

    # Get baseline predictions
    df_model['Predicted_LE'] = model.predict(scaler.transform(df_model[feature_cols]))

    impacts_dict = {}
    impact_means = {}

    # Scenario 1: Health Spending
    s1_changes = {'public_health_spend': lambda x: x * 1.1}
    impact_s1 = run_scenario(df_model, scaler, model, feature_cols, "S1", s1_changes)
    if impact_s1 is not None:
        impacts_dict['S1'] = impact_s1
        impact_means['S1'] = impact_s1.mean()

    # Scenario 2: Education
    s2_changes = {'Pct_Level_4_Plus': lambda x: x * 1.1}
    impact_s2 = run_scenario(df_model, scaler, model, feature_cols, "S2", s2_changes)
    if impact_s2 is not None:
        impacts_dict['S2'] = impact_s2
        impact_means['S2'] = impact_s2.mean()

    # Scenario 3: Employment
    # This one is more complex as it changes two features
    def change_employment(df):
        reduction = df['Unemployment_Rate'] * 0.2
        df['Unemployment_Rate'] = df['Unemployment_Rate'] * 0.8
        df['Employment_Rate'] = df['Employment_Rate'] + reduction
        return df

    X_scenario3 = df_model[feature_cols].copy()
    X_scenario3 = change_employment(X_scenario3)
    impact_s3 = run_scenario(df_model, scaler, model, feature_cols, "S3",
                             {'Unemployment_Rate': lambda x: X_scenario3['Unemployment_Rate'],
                              'Employment_Rate': lambda x: X_scenario3['Employment_Rate']})
    if impact_s3 is not None:
        impacts_dict['S3'] = impact_s3
        impact_means['S3'] = impact_s3.mean()

    # Scenario 4: Environment
    s4_changes = {'Air_Pollution_PM25': lambda x: x * 0.85}
    impact_s4 = run_scenario(df_model, scaler, model, feature_cols, "S4", s4_changes)
    if impact_s4 is not None:
        impacts_dict['S4'] = impact_s4
        impact_means['S4'] = impact_s4.mean()

    plot_what_if_summary(impact_means)

    return {
        "df_model": df_model,
        "impacts_dict": impacts_dict,
        "impact_means": impact_means
    }
