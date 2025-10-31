""""
Dataset merger file
Merges all preprocessed datasets into a master dataset for analysis and modeling.

""" 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# Input paths (processed datasets)
CENSUS_DEMOGRAPHICS = "datasets/processed/census_demographics_processed.csv"
HEALTH_OUTCOMES = "datasets/processed/master_health_data.csv"
OBESITY_DATA = "datasets/processed/obesity_admissions_processed.csv"
AGE_DISTRIBUTION = "datasets/processed/ts007aageby5yearagebandslowertierlocalauthorities.csv"
AIR_POLLUTION = "datasets/processed/air polution.csv"
HOUSING_PRICES = "datasets/processed/England-annual-price-change-by-local-authority-2021-03.csv"
SPENDING_DATA = "datasets/processed/local_auth_spending.csv"

# Output path (master dataset)
FINAL_OUTPUT = "datasets/master/uk_health_master_dataset.csv"

print("LOADING CENSUS DEMOGRAPHICS (BASE DATASET)")
print()

master = pd.read_csv(CENSUS_DEMOGRAPHICS)
print(f"✓ Loaded: {master.shape}")
print(f"Columns: {master.columns.tolist()[:5]}...")

print("PROCESSING HEALTH OUTCOMES (TARGET VARIABLES)")
print()

health = pd.read_csv(HEALTH_OUTCOMES)
print(f"Raw health data: {health.shape}")

# Renaming to match
health.rename(columns={'Area_Code': 'LA_Code'}, inplace=True)

# Processing Health Data

# key columns
health_cols = ['LA_Code', 'Area_Name', 'Life_Expectancy_Male', 'Life_Expectancy_Female', 
               'Life_Expectancy_Avg', 'Life_Expectancy_Gap', 
               'Male_Inequality_Gap', 'Female_Inequality_Gap', 'Avg_Inequality_Gap',
               'Male_Most_Deprived_LE', 'Male_Least_Deprived_LE',
               'Female_Most_Deprived_LE', 'Female_Least_Deprived_LE',
               'Region_Type', 'Urban_Rural']

health_final = health[health_cols].copy()

print("\nMissing values in health data:")
print(health_final.isnull().sum()[health_final.isnull().sum() > 0])

# Imputing missing life expectancy with median
numeric_cols = ['Life_Expectancy_Male', 'Life_Expectancy_Female', 'Life_Expectancy_Avg',
                'Male_Inequality_Gap', 'Female_Inequality_Gap', 'Avg_Inequality_Gap']

for col in numeric_cols:
    if col in health_final.columns:
        if health_final[col].isnull().sum() > 0:
            median_val = health_final[col].median()
            health_final[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} with median: {median_val:.2f}")

print(f"\n✓ Health outcomes processed: {health_final.shape}")

master = master.merge(health_final, on='LA_Code', how='left')
print(f"✓ Merged with master: {master.shape}")

# Process Age Data

print("PROCESSING AGE DISTRIBUTION")
print()

age = pd.read_csv(AGE_DISTRIBUTION)
print(f"Raw age data: {age.shape}")

age.rename(columns={'Lower tier local authority code': 'LA_Code'}, inplace=True)

# Remove commas from age columns and convert to numeric
age_cols = [col for col in age.columns if 'Aged' in col]
for col in age_cols:
    if age[col].dtype == 'object':
        age[col] = age[col].str.replace(',', '').astype(float)

age['Total_Population_Age'] = age[age_cols].sum(axis=1)

# Age dependency ratios
age['Pct_Under_15'] = ((age['Aged 4 years and under'] + age['Aged 5 to 9 years'] + 
                        age['Aged 10 to 14 years']) / age['Total_Population_Age'] * 100).round(2)

age['Pct_65_Plus'] = ((age['Aged 65 to 69 years'] + age['Aged 70 to 74 years'] + 
                       age['Aged 75 to 79 years'] + age['Aged 80 to 84 years'] + 
                       age['Aged 85 years and over']) / age['Total_Population_Age'] * 100).round(2)

age['Pct_Working_Age'] = (100 - age['Pct_Under_15'] - age['Pct_65_Plus']).round(2)

# Old age dependency ratio
age['Old_Age_Dependency_Ratio'] = (age['Pct_65_Plus'] / age['Pct_Working_Age'] * 100).round(2)

# Median age approximation (weighted average of age bands)
age_band_midpoints = {
    'Aged 4 years and under': 2,
    'Aged 5 to 9 years': 7,
    'Aged 10 to 14 years': 12,
    'Aged 15 to 19 years': 17,
    'Aged 20 to 24 years': 22,
    'Aged 25 to 29 years': 27,
    'Aged 30 to 34 years': 32,
    'Aged 35 to 39 years': 37,
    'Aged 40 to 44 years': 42,
    'Aged 45 to 49 years': 47,
    'Aged 50 to 54 years': 52,
    'Aged 55 to 59 years': 57,
    'Aged 60 to 64 years': 62,
    'Aged 65 to 69 years': 67,
    'Aged 70 to 74 years': 72,
    'Aged 75 to 79 years': 77,
    'Aged 80 to 84 years': 82,
    'Aged 85 years and over': 87
}

age['Estimated_Median_Age'] = 0
for col, midpoint in age_band_midpoints.items():
    age['Estimated_Median_Age'] += age[col] * midpoint

age['Estimated_Median_Age'] = (age['Estimated_Median_Age'] / age['Total_Population_Age']).round(1)

# final age features
age_final = age[['LA_Code', 'Pct_Under_15', 'Pct_65_Plus', 'Pct_Working_Age', 
                 'Old_Age_Dependency_Ratio', 'Estimated_Median_Age']].copy()

print(f"✓ Age distribution processed: {age_final.shape}")
print(f"\nAge statistics:")
print(age_final[['Pct_65_Plus', 'Old_Age_Dependency_Ratio', 'Estimated_Median_Age']].describe())

master = master.merge(age_final, on='LA_Code', how='left')
print(f"✓ Merged with master: {master.shape}")

# 4. PROCESS OBESITY DATA

print("\n" + "="*70)
print("PROCESSING OBESITY DATA")
print("="*70)

obesity = pd.read_csv(OBESITY_DATA)
print(f"Raw obesity data: {obesity.shape}")

obesity.rename(columns={'Area Code': 'LA_Code'}, inplace=True)
\
print("\nMissing values in obesity data:")
print(obesity.isnull().sum()[obesity.isnull().sum() > 0])

for col in obesity.columns:
    if col != 'LA_Code' and obesity[col].isnull().sum() > 0:
        median_val = obesity[col].median()
        obesity[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} with median: {median_val:.2f}")

print(f"\n✓ Obesity data processed: {obesity.shape}")

master = master.merge(obesity, on='LA_Code', how='left')
print(f"✓ Merged with master: {master.shape}")

# Filling remaining obesity NaNs with median (for LAs not in obesity dataset)
obesity_cols = ['primary_obesity_all', 'total_obesity_burden', 'bariatric_surgery_rate', 'obesity_gender_gap']
for col in obesity_cols:
    if col in master.columns and master[col].isnull().sum() > 0:
        master[col].fillna(master[col].median(), inplace=True)


# 5. PROCESS LOCAL AUTHORITY SPENDING

print("\n" + "="*70)
print("PROCESSING LOCAL AUTHORITY SPENDING")
print("="*70)

spending = pd.read_csv(SPENDING_DATA)
print(f"✓ Loaded: {spending.shape}")

# Rename ons_code to LA_Code
spending.rename(columns={'ons_code': 'LA_Code'}, inplace=True)

# Select key spending features (drop one-hot encoded authority types to avoid redundancy)
spending_cols = ['LA_Code', 'education_spend', 'children_social_care', 'adult_social_care', 
                 'public_health_spend', 'housing_spend', 'environmental_spend', 
                 'cultural_spend', 'transport_spend', 'planning_spend', 
                 'central_services_spend', 'council_tax_revenue', 
                 'total_revenue_expenditure', 'total_service_expenditure',
                 'education_spend_pct', 'children_social_care_pct', 'adult_social_care_pct',
                 'public_health_spend_pct', 'housing_spend_pct', 'environmental_spend_pct',
                 'cultural_spend_pct', 'transport_spend_pct', 'planning_spend_pct',
                 'central_services_spend_pct', 'total_health_social_spend',
                 'health_spend_ratio', 'preventive_vs_reactive_ratio']

spending_final = spending[spending_cols].copy()

# Handle missing values
print("\nMissing values in spending data:")
missing_spending = spending_final.isnull().sum()
if missing_spending.sum() > 0:
    print(missing_spending[missing_spending > 0])
    
    for col in spending_final.columns:
        if col != 'LA_Code' and spending_final[col].isnull().sum() > 0:
            spending_final[col].fillna(spending_final[col].median(), inplace=True)
            print(f"  Filled {col} with median")
else:
    print("✓ No missing values in spending data")

print(f"\n✓ Spending data processed: {spending_final.shape}")
print(f"\nKey spending statistics:")
print(spending_final[['public_health_spend', 'adult_social_care', 'health_spend_ratio']].describe())

# Merge with master
master = master.merge(spending_final, on='LA_Code', how='left')
print(f"✓ Merged with master: {master.shape}")

# Fill remaining spending NaNs with median (for LAs not in spending dataset)
spending_numeric_cols = [col for col in spending_cols if col != 'LA_Code']
for col in spending_numeric_cols:
    if col in master.columns and master[col].isnull().sum() > 0:
        master[col].fillna(master[col].median(), inplace=True)
        print(f"  Imputed {col} with median")

# ============================================================================
# 6. PROCESS AIR POLLUTION
# ============================================================================

print("PROCESSING AIR POLLUTION DATA")
print()

pollution = pd.read_csv(AIR_POLLUTION)
print(f"Raw pollution data: {pollution.shape}")

pollution.rename(columns={'Area Code': 'LA_Code', 'Value': 'Air_Pollution_PM25'}, inplace=True)

pollution_final = pollution[['LA_Code', 'Air_Pollution_PM25']].copy()

# Handling missing values
if pollution_final['Air_Pollution_PM25'].isnull().sum() > 0:
    pollution_final['Air_Pollution_PM25'].fillna(pollution_final['Air_Pollution_PM25'].median(), inplace=True)

print(f"✓ Air pollution processed: {pollution_final.shape}")
print(f"\nAir pollution statistics:")
print(pollution_final['Air_Pollution_PM25'].describe())

master = master.merge(pollution_final, on='LA_Code', how='left')
print(f"✓ Merged with master: {master.shape}")

# Filling remaining pollution NaNs with median
if master['Air_Pollution_PM25'].isnull().sum() > 0:
    master['Air_Pollution_PM25'].fillna(master['Air_Pollution_PM25'].median(), inplace=True)

# 7. PROCESS HOUSING PRICES

print("PROCESSING HOUSING PRICES")
print()

try:
    housing = pd.read_csv(HOUSING_PRICES, encoding='latin1')
    print(f"✓ Loaded: {housing.shape}")
    print(f"Columns: {housing.columns.tolist()}")
    
    housing['March 2021'] = housing['March 2021'].str.replace('£', '').str.replace(',', '').astype(float)
    housing['March 2020'] = housing['March 2020'].str.replace('£', '').str.replace(',', '').astype(float)
    housing['Difference'] = housing['Difference'].str.replace('%', '').astype(float)
    
    housing.rename(columns={
        'Local authorities': 'Area_Name',
        'March 2021': 'House_Price_2021',
        'March 2020': 'House_Price_2020',
        'Difference': 'House_Price_Change_Pct'
    }, inplace=True)
    
    # Merging on Area_Name instead of LA_Code
    housing_final = housing[['Area_Name', 'House_Price_2021', 'House_Price_2020', 'House_Price_Change_Pct']].copy()
    
    # Handling missing values
    for col in ['House_Price_2021', 'House_Price_2020', 'House_Price_Change_Pct']:
        if housing_final[col].isnull().sum() > 0:
            housing_final[col].fillna(housing_final[col].median(), inplace=True)
    
    print(f"\n✓ Housing prices processed: {housing_final.shape}")
    print(f"\nHousing statistics:")
    print(housing_final[['House_Price_2021', 'House_Price_Change_Pct']].describe())
    
    master = master.merge(housing_final, on='Area_Name', how='left')
    print(f"✓ Merged with master: {master.shape}")
    
    # Filling remaining housing NaNs with median
    housing_cols = ['House_Price_2021', 'House_Price_2020', 'House_Price_Change_Pct']
    for col in housing_cols:
        if col in master.columns and master[col].isnull().sum() > 0:
            master[col].fillna(master[col].median(), inplace=True)
            print(f"  Imputed {col} missing values with median")
    
except Exception as e:
    print(f"⚠️  Could not load housing prices: {e}")
    print("Skipping housing data")

# FINAL DATA QUALITY CHECKS

print("\n" + "="*70)
print("FINAL DATASET SUMMARY")
print("="*70)

print(f"Final shape: {master.shape}")
print(f"Total features: {len(master.columns) - 1}")

# Missing values summary
print("\n" + "="*70)
print("MISSING VALUES SUMMARY")
print("="*70)

missing = master.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
    print(f"\nTotal missing values: {missing.sum()}")
    
    # Impute remaining missing values with median
    for col in master.columns:
        if master[col].dtype in ['float64', 'int64'] and master[col].isnull().sum() > 0:
            master[col].fillna(master[col].median(), inplace=True)
            print(f"  Imputed {col} with median")
else:
    print("✓ No missing values!")

# Data summary
print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)
print(master.info())

# SAVING MASTER DATASET

import os
os.makedirs('datasets/master', exist_ok=True)

master.to_csv(FINAL_OUTPUT, index=False)
print(f"\n✓ MASTER DATASET SAVED: {FINAL_OUTPUT}")
print(f"   Shape: {master.shape}")

print("\n" + "="*70)
print("✓ COMPLETE! Master dataset ready for ML modeling")
print("="*70)
print(f"\nDataset location: {FINAL_OUTPUT}")
print()