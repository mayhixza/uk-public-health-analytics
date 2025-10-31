"""
EDA functions and visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config

def print_summary_stats(df, target_primary, target_secondary):
    print("\n📈 Health Outcome Statistics:")
    print(f"Life Expectancy: {df[target_primary].mean():.2f} ± {df[target_primary].std():.2f} years")
    print(f"Range: {df[target_primary].min():.2f} - {df[target_primary].max():.2f} years")
    print(f"\nHealth Inequality Gap: {df[target_secondary].mean():.2f} ± {df[target_secondary].std():.2f} years")
    print(f"Range: {df[target_secondary].min():.2f} - {df[target_secondary].max():.2f} years")

def print_key_correlations(df_model, target_primary):
    print("\n🔗 Top 10 Positive Correlations with Life Expectancy:")
    numeric_cols_only = df_model.select_dtypes(include=[np.number]).columns.tolist()
    correlations = df_model[numeric_cols_only].corr()[target_primary].sort_values(ascending=False)
    print(correlations.head(11)[1:])

    print("\n⚠️ Top 10 Negative Correlations (Risk Factors):")
    print(correlations.tail(10))
    return correlations

def plot_correlation_heatmap(df_model, correlations, target_primary):
    print("\n📊 Generating Visualization 1: Correlation Heatmap...")
    top_features = correlations.abs().sort_values(ascending=False).head(16).index.tolist()
    # Ensure target is not in the list of features, just for the heatmap matrix
    top_features = [f for f in top_features if f != target_primary][:config.TOP_N_FEATURES]

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_model[top_features + [target_primary]].corr(),
                annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1)
    plt.title(f'Correlation Heatmap: Top {config.TOP_N_FEATURES} Features vs Life Expectancy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.VIZ_CORR_HEATMAP, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {config.VIZ_CORR_HEATMAP}")
    plt.close()

def plot_outcomes_distribution(df, target_primary, target_secondary):
    # Generating and saving histograms for the target variables
    print("\n📊 Generating Visualization 2: Health Outcomes Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Life Expectancy
    axes[0].hist(df[target_primary], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df[target_primary].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].set_xlabel('Life Expectancy (years)', fontsize=12)
    axes[0].set_ylabel('Number of Areas', fontsize=12)
    axes[0].set_title('Distribution of Life Expectancy Across UK', fontsize=13, fontweight='bold')
    axes[0].legend()

    # Plot 2: Inequality Gap
    axes[1].hist(df[target_secondary], bins=30, color='salmon', edgecolor='black', alpha=0.7)
    axes[1].axvline(df[target_secondary].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].set_xlabel('Inequality Gap (years)', fontsize=12)
    axes[1].set_ylabel('Number of Areas', fontsize=12)
    axes[1].set_title('Distribution of Health Inequality Gap', fontsize=13, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(config.VIZ_DISTRIBUTION, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {config.VIZ_DISTRIBUTION}")
    plt.close()

def run_exploratory_data_analysis(df, df_model, target_primary, target_secondary):
    #running full eda pipeline
    print_summary_stats(df, target_primary, target_secondary)
    correlations = print_key_correlations(df_model, target_primary)
    plot_correlation_heatmap(df_model, correlations, target_primary)
    plot_outcomes_distribution(df, target_primary, target_secondary)
