"""
Functions for clustering analysis (K-Means).
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import config


def run_kmeans_clustering(df, clustering_features, n_clusters):
    """Performs K-Means clustering on the provided features."""
    print("\n🗺️ Performing K-Means Clustering...")

    # filtering to only features that exist in df
    features_available = [f for f in clustering_features if f in df.columns]

    # prepare clustering data from original df (has all composite features)
    X_cluster = df[features_available].copy()
    X_cluster = X_cluster.fillna(X_cluster.median())
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

    print(f"✓ Created {n_clusters} clusters using {len(features_available)} features")
    return df, kmeans


def profile_clusters(df, n_clusters):
    """Prints a summary profile for each generated cluster."""
    print("\n📊 Cluster Profiles:")
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        print(f"\n{'=' * 60}")
        print(f"CLUSTER {i + 1} ({len(cluster_data)} areas)")
        print(f"{'=' * 60}")
        print(f"Avg Life Expectancy: {cluster_data['Life_Expectancy_Avg'].mean():.2f} years")
        print(f"Avg Inequality Gap: {cluster_data['Avg_Inequality_Gap'].mean():.2f} years")

        if 'Socioeconomic_Index' in cluster_data.columns:
            print(f"Avg Socioeconomic Index: {cluster_data['Socioeconomic_Index'].mean():.1f}")
        if 'Health_Deprivation_Score' in cluster_data.columns:
            print(f"Avg Health Deprivation: {cluster_data['Health_Deprivation_Score'].mean():.1f}")

        # Sample areas - shows worst performers in this cluster
        sample_areas = cluster_data.nsmallest(5, 'Life_Expectancy_Avg')['Area_Name_Stored'].tolist()
        print(f"Sample areas: {', '.join(sample_areas[:5])}")


def name_clusters(df, n_clusters):
    """Assigns descriptive names to clusters based on their profiles."""
    cluster_names_map = {}
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        avg_le = cluster_data['Life_Expectancy_Avg'].mean()
        avg_ineq = cluster_data['Avg_Inequality_Gap'].mean()
        avg_socio = cluster_data['Socioeconomic_Index'].mean() if 'Socioeconomic_Index' in cluster_data.columns else 50

        if avg_le < 79 and avg_ineq > 9:
            cluster_names_map[i] = "High Risk - Deprived"
        elif avg_le > 82 and avg_socio > 60:
            cluster_names_map[i] = "Prosperous - Low Risk"
        elif avg_le > 80 and avg_ineq < 7:
            cluster_names_map[i] = "Stable - Good Outcomes"
        else:
            cluster_names_map[i] = "Transitioning - Mixed"

    df['Cluster_Name'] = df['Cluster'].map(cluster_names_map)

    print("\n" + "=" * 80)
    print("CLUSTER SUMMARY")
    print("=" * 80)
    print(df['Cluster_Name'].value_counts())

    return df, list(cluster_names_map.values())


def plot_cluster_analysis(df, n_clusters, cluster_names):
    # generates and saves the 4-panel cluster analysis visualization
    print("\n📊 Generating Visualization 5: Cluster Analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Life Expectancy by Cluster
    cluster_le = df.groupby('Cluster_Name')['Life_Expectancy_Avg'].mean().sort_values()
    axes[0, 0].barh(cluster_le.index, cluster_le.values, color='steelblue')
    axes[0, 0].set_xlabel('Average Life Expectancy (years)', fontsize=11)
    axes[0, 0].set_title('Life Expectancy by Cluster', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # Plot 2: Inequality Gap by Cluster
    cluster_ineq = df.groupby('Cluster_Name')['Avg_Inequality_Gap'].mean().sort_values()
    axes[0, 1].barh(cluster_ineq.index, cluster_ineq.values, color='coral')
    axes[0, 1].set_xlabel('Average Inequality Gap (years)', fontsize=11)
    axes[0, 1].set_title('Health Inequality by Cluster', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Plot 3: Scatter - Life Expectancy vs Inequality
    unique_clusters = df['Cluster'].unique()
    cluster_name_list = df.drop_duplicates('Cluster').set_index('Cluster')['Cluster_Name']

    for i in unique_clusters:
        cluster_data = df[df['Cluster'] == i]
        axes[1, 0].scatter(cluster_data['Life_Expectancy_Avg'],
                           cluster_data['Avg_Inequality_Gap'],
                           label=cluster_name_list[i], alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Life Expectancy (years)', fontsize=11)
    axes[1, 0].set_ylabel('Inequality Gap (years)', fontsize=11)
    axes[1, 0].set_title('Life Expectancy vs Inequality by Cluster', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cluster Size
    cluster_sizes = df['Cluster_Name'].value_counts()
    axes[1, 1].pie(cluster_sizes.values, labels=cluster_sizes.index, autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Set3.colors[:len(cluster_sizes)])
    axes[1, 1].set_title('Distribution of Areas by Cluster', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(config.VIZ_CLUSTER, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {config.VIZ_CLUSTER}")
    plt.close()


def run_all_clustering(df, df_model, clustering_features, n_clusters):
    df, kmeans_model = run_kmeans_clustering(df, clustering_features, n_clusters)
    profile_clusters(df, n_clusters)
    df, cluster_names = name_clusters(df, n_clusters)

    # adds cluster info to df_model as well
    df_model['Cluster'] = df.loc[df_model.index, 'Cluster'].values
    df_model['Cluster_Name'] = df.loc[df_model.index, 'Cluster_Name'].values

    plot_cluster_analysis(df, n_clusters, cluster_names)

    return {
        "df": df,
        "df_model": df_model,
        "kmeans_model": kmeans_model,
        "cluster_names": cluster_names,
        "n_clusters": n_clusters
    }
