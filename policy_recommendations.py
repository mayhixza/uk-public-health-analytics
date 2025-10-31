"""
File for generating policy insights and recommendations.
"""

import pandas as pd


def categorize_risk(df_model, target_primary):
    # Categorizes areas into High, Medium, and Low Risk based on Life Expectancy
    le_threshold_low = df_model[target_primary].quantile(0.25)
    le_threshold_high = df_model[target_primary].quantile(0.75)

    df_model['Risk_Category'] = pd.cut(
        df_model[target_primary],
        bins=[0, le_threshold_low, le_threshold_high, 100],
        labels=['High Risk', 'Medium Risk', 'Low Risk']
    )
    return df_model


def print_risk_analysis(df_model, target_primary, target_secondary):
    #Prints risk distribution and top/bottom 10 areas
    print(f"\n⚠️ Risk Distribution:")
    print(df_model['Risk_Category'].value_counts())

    print(f"\n📍 Top 10 Highest Risk Areas (Lowest Life Expectancy):")
    high_risk = df_model.nsmallest(10, target_primary)[
        ['Area_Name_Stored', target_primary, target_secondary, 'Cluster_Name', 'Risk_Category']
    ]
    print(high_risk.to_string(index=False))

    print(f"\n🌟 Top 10 Best Performing Areas (Highest Life Expectancy):")
    best_areas = df_model.nlargest(10, target_primary)[
        ['Area_Name_Stored', target_primary, target_secondary, 'Cluster_Name', 'Risk_Category']
    ]
    print(best_areas.to_string(index=False))
    return len(df_model[df_model['Risk_Category'] == 'High Risk']), \
        len(df_model[df_model['Risk_Category'] == 'Low Risk'])


def generate_recommendation_text(df_model, eval_results, scenario_results, target_primary, target_secondary):
    # Generates the final text block of policy recommendations.

    impact_means = scenario_results['impact_means']
    best_s = eval_results['best_secondary']

    ineq_threshold = df_model[target_secondary].quantile(0.75)
    high_ineq_count = len(df_model[df_model[target_secondary] > ineq_threshold])
    high_risk_count = len(df_model[df_model['Risk_Category'] == 'High Risk'])
    low_risk_count = len(df_model[df_model['Risk_Category'] == 'Low Risk'])

    recommendations = f"""
{"=" * 80}
💡 EVIDENCE-BASED POLICY RECOMMENDATIONS
{"=" * 80}

Based on the comprehensive ML analysis of UK health outcomes:

1. **EDUCATION AS PRIMARY LEVER** (Strongest predictor)
   - Higher education (Level 4+) shows one of the strongest positive correlations with life expectancy.
   - What-If Analysis: 10% increase in graduates → +{impact_means.get('S2', 0):.3f} years average gain.
   - Priority: Adult education programs, apprenticeships, skills training.
   - Target: Areas in "High Risk - Deprived" cluster.

2. **EMPLOYMENT & ECONOMIC SECURITY**
   - Employment rate directly impacts health outcomes.
   - What-If Analysis: 20% unemployment reduction → +{impact_means.get('S3', 0):.3f} years gain.
   - Priority: Job creation, employment support programs.
   - Target: Post-industrial and deprived urban areas.

3. **PREVENTIVE HEALTHCARE INVESTMENT**
   - Public health spending shows measurable impact.
   - What-If Analysis: 10% spending increase → +{impact_means.get('S1', 0):.3f} years gain.
   - Priority: Preventive care, early intervention programs.
   - Target: High inequality areas (gap >{ineq_threshold:.1f} years).

4. **ENVIRONMENTAL QUALITY**
   - Air pollution (PM2.5) negatively impacts life expectancy.
   - What-If Analysis: 15% pollution reduction → +{impact_means.get('S4', 0):.3f} years gain.
   - Priority: Clean air zones, green space development.
   - Target: Urban industrial clusters.

5. **ADDRESS HEALTH INEQUALITY**
   - Inequality gap ranges from {df_model[target_secondary].min():.1f} to {df_model[target_secondary].max():.1f} years across UK.
   - Model predicts inequality with R² = {best_s['R² Score']:.3f}.
   - Priority: Targeted interventions in high-gap areas.
   - Target: {high_ineq_count} areas with inequality >{ineq_threshold:.1f} years.

6. **GEOGRAPHIC TARGETING**
   - Focus resources on identified {high_risk_count} high-risk areas.
   - Learn from {low_risk_count} low-risk "Prosperous" cluster areas.
   - Cluster-specific strategies more effective than one-size-fits-all.
"""
    return recommendations


def generate_policy_recommendations(df_model, eval_results, scenario_results, target_primary, target_secondary):
    df_model = categorize_risk(df_model, target_primary)
    print_risk_analysis(df_model, target_primary, target_secondary)
    policy_text = generate_recommendation_text(
        df_model, eval_results, scenario_results, target_primary, target_secondary
    )
    return policy_text
