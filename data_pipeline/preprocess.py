"""
Dataset preprocessing module

Processes:
1. Census Demographics (Ethnicity, Economic Activity, Education)
2. Life Expectancy Data
3. Obesity Hospital Admissions
4. Local Authority Finance Data

"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Census Demographics Dataset Processing

class CensusPreprocessor:
    # Census Demographics: Ethnicity, Economic Activity, Education

    @staticmethod
    def process_ethnicity_data(filepath):
        df = pd.read_csv(filepath)
        df = df[df['Ethnic group (20 categories) Code'] != -8].copy()

        df['Ethnicity_Clean'] = df['Ethnic group (20 categories)'].str.replace(
            'Asian, Asian British or Asian Welsh: ', 'Asian_'
        ).str.replace(
            'Black, Black British, Black Welsh, Caribbean or African: ', 'Black_'
        ).str.replace(
            'Mixed or Multiple ethnic groups: ', 'Mixed_'
        ).str.replace(
            'White: ', 'White_'
        ).str.replace(
            'Other ethnic group: ', 'Other_'
        ).str.replace(' ', '_').str.replace(',', '').str.replace(':', '')

        ethnic_wide = df.pivot(
            index='Lower Tier Local Authorities Code',
            columns='Ethnicity_Clean',
            values='Observation'
        ).reset_index()

        ethnic_wide['Total_Population'] = ethnic_wide.iloc[:, 1:].sum(axis=1)

        ethnicity_cols = [col for col in ethnic_wide.columns if col not in
                         ['Lower Tier Local Authorities Code', 'Total_Population']]

        for col in ethnicity_cols:
            ethnic_wide[f'{col}_Pct'] = (ethnic_wide[col] / ethnic_wide['Total_Population'] * 100).round(2)

        def calculate_diversity_index(row):
            ethnicity_counts = pd.to_numeric(row[ethnicity_cols], errors='coerce').astype(float)
            ethnicity_counts = ethnicity_counts[~np.isnan(ethnicity_counts)]

            if ethnicity_counts.sum() == 0 or len(ethnicity_counts) == 0:
                return 0.0

            proportions = ethnicity_counts / ethnicity_counts.sum()
            return entropy(proportions)

        ethnic_wide['Ethnic_Diversity_Index'] = ethnic_wide.apply(calculate_diversity_index, axis=1).round(3)

        if 'White_English_Welsh_Scottish_Northern_Irish_or_British' in ethnic_wide.columns:
            ethnic_wide['Pct_White_British'] = (
                ethnic_wide['White_English_Welsh_Scottish_Northern_Irish_or_British'] /
                ethnic_wide['Total_Population'] * 100
            ).round(2)
            ethnic_wide['Pct_Ethnic_Minority'] = (100 - ethnic_wide['Pct_White_British']).round(2)

        keep_cols = ['Lower Tier Local Authorities Code', 'Total_Population',
                     'Ethnic_Diversity_Index', 'Pct_White_British', 'Pct_Ethnic_Minority']
        keep_cols += [col for col in ethnic_wide.columns if col.endswith('_Pct')]

        ethnic_final = ethnic_wide[keep_cols].copy()
        ethnic_final.rename(columns={'Lower Tier Local Authorities Code': 'LA_Code'}, inplace=True)

        print(f"Ethnicity data processed: {len(ethnic_final)} local authorities, {len(ethnic_final.columns)-1} features")
        return ethnic_final

    @staticmethod
    def process_economic_activity_data(filepath):
        # Process TS066 economic activity census data.

        df = pd.read_csv(filepath)
        df = df[df['Economic activity status (20 categories) Code'] != -8].copy()

        econ_totals = df.groupby('Lower Tier Local Authorities Code')['Observation'].sum().reset_index()
        econ_totals.columns = ['LA_Code', 'Total_Working_Age_Population']

        df['Category'] = 'Other'
        employed_codes = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        unemployed_codes = [7, 14]
        inactive_codes = [15, 16, 17, 18, 19]

        df.loc[df['Economic activity status (20 categories) Code'].isin(employed_codes), 'Category'] = 'Employed'
        df.loc[df['Economic activity status (20 categories) Code'].isin(unemployed_codes), 'Category'] = 'Unemployed'
        df.loc[df['Economic activity status (20 categories) Code'].isin(inactive_codes), 'Category'] = 'Inactive'

        econ_summary = df.groupby(['Lower Tier Local Authorities Code', 'Category'])['Observation'].sum().reset_index()
        econ_wide = econ_summary.pivot(
            index='Lower Tier Local Authorities Code',
            columns='Category',
            values='Observation'
        ).reset_index()

        econ_wide.columns.name = None
        econ_wide.rename(columns={'Lower Tier Local Authorities Code': 'LA_Code'}, inplace=True)
        econ_wide = econ_wide.merge(econ_totals, on='LA_Code', how='left')

        econ_wide['Employment_Rate'] = (econ_wide['Employed'] / econ_wide['Total_Working_Age_Population'] * 100).round(2)
        econ_wide['Unemployment_Rate'] = (econ_wide['Unemployed'] / econ_wide['Total_Working_Age_Population'] * 100).round(2)
        econ_wide['Economic_Inactivity_Rate'] = (econ_wide['Inactive'] / econ_wide['Total_Working_Age_Population'] * 100).round(2)

        retired = df[df['Economic activity status (20 categories) Code'] == 15].groupby('Lower Tier Local Authorities Code')['Observation'].sum()
        sick_disabled = df[df['Economic activity status (20 categories) Code'] == 18].groupby('Lower Tier Local Authorities Code')['Observation'].sum()

        econ_wide = econ_wide.merge(retired.rename('Retired_Count').reset_index().rename(columns={'Lower Tier Local Authorities Code': 'LA_Code'}),
                                      on='LA_Code', how='left')
        econ_wide = econ_wide.merge(sick_disabled.rename('Sick_Disabled_Count').reset_index().rename(columns={'Lower Tier Local Authorities Code': 'LA_Code'}),
                                      on='LA_Code', how='left')

        econ_wide['Pct_Sick_Disabled'] = (econ_wide['Sick_Disabled_Count'] / econ_wide['Total_Working_Age_Population'] * 100).round(2)

        final_cols = ['LA_Code', 'Total_Working_Age_Population', 'Employment_Rate',
                      'Unemployment_Rate', 'Economic_Inactivity_Rate', 'Pct_Sick_Disabled']

        econ_final = econ_wide[final_cols].copy()

        print(f"Economic activity data processed: {len(econ_final)} local authorities, {len(econ_final.columns)-1} features")
        return econ_final

    @staticmethod
    def process_education_data(filepath):
        # Processing Education Data

        df = pd.read_csv(filepath)
        df = df[df['Highest level of qualification (8 categories) Code'] != -8].copy()

        edu_totals = df.groupby('Lower Tier Local Authorities Code')['Observation'].sum().reset_index()
        edu_totals.columns = ['LA_Code', 'Total_Population_16plus']

        df['Education_Level'] = 'Other'
        df.loc[df['Highest level of qualification (8 categories) Code'] == 0, 'Education_Level'] = 'No_Qualifications'
        df.loc[df['Highest level of qualification (8 categories) Code'] == 1, 'Education_Level'] = 'Level_1'
        df.loc[df['Highest level of qualification (8 categories) Code'] == 2, 'Education_Level'] = 'Level_2'
        df.loc[df['Highest level of qualification (8 categories) Code'] == 3, 'Education_Level'] = 'Apprenticeship'
        df.loc[df['Highest level of qualification (8 categories) Code'] == 4, 'Education_Level'] = 'Level_3'
        df.loc[df['Highest level of qualification (8 categories) Code'] == 5, 'Education_Level'] = 'Level_4_Plus'
        df.loc[df['Highest level of qualification (8 categories) Code'] == 6, 'Education_Level'] = 'Other'

        edu_wide = df.groupby(['Lower Tier Local Authorities Code', 'Education_Level'])['Observation'].sum().reset_index()
        edu_wide = edu_wide.pivot(
            index='Lower Tier Local Authorities Code',
            columns='Education_Level',
            values='Observation'
        ).reset_index()

        edu_wide.columns.name = None
        edu_wide.rename(columns={'Lower Tier Local Authorities Code': 'LA_Code'}, inplace=True)
        edu_wide = edu_wide.merge(edu_totals, on='LA_Code', how='left')

        edu_cols = ['No_Qualifications', 'Level_1', 'Level_2', 'Apprenticeship', 'Level_3', 'Level_4_Plus']

        for col in edu_cols:
            if col in edu_wide.columns:
                edu_wide[f'Pct_{col}'] = (edu_wide[col] / edu_wide['Total_Population_16plus'] * 100).round(2)

        edu_wide['Pct_High_Skills'] = edu_wide['Pct_Level_4_Plus']

        if 'Pct_No_Qualifications' in edu_wide.columns and 'Pct_Level_1' in edu_wide.columns:
            edu_wide['Pct_Low_Skills'] = (edu_wide['Pct_No_Qualifications'] + edu_wide['Pct_Level_1']).round(2)

        final_cols = ['LA_Code', 'Total_Population_16plus', 'Pct_No_Qualifications',
                      'Pct_Level_2', 'Pct_Level_3', 'Pct_Level_4_Plus',
                      'Pct_High_Skills', 'Pct_Low_Skills']

        final_cols = [col for col in final_cols if col in edu_wide.columns]
        edu_final = edu_wide[final_cols].copy()

        print(f"Education data processed: {len(edu_final)} local authorities, {len(edu_final.columns)-1} features")
        return edu_final

    @staticmethod
    def process_census_demographics(ethnicity_path, economic_path, education_path, output_path=None):
        # Pipeline to process all census demographic data."""
        print("="*70)
        print("CENSUS DEMOGRAPHICS PROCESSING PIPELINE")
        print("="*70)

        ethnicity_df = CensusPreprocessor.process_ethnicity_data(ethnicity_path)
        economic_df = CensusPreprocessor.process_economic_activity_data(economic_path)
        education_df = CensusPreprocessor.process_education_data(education_path)

        print("\nMerging Census Demographics...")
        master = ethnicity_df.copy()
        master = master.merge(economic_df, on='LA_Code', how='left')
        master = master.merge(education_df, on='LA_Code', how='left')

        print(f"Census demographics merged: {len(master)} local authorities, {len(master.columns)-1} features")

        if output_path:
            master.to_csv(output_path, index=False)
            print(f"\nSaved to: {output_path}")

        print("\n" + "="*70)
        print("CENSUS PROCESSING COMPLETE!")
        print("="*70)

        return master


#Life Extpectancy Data Processing

class LifeExpectancyPreprocessor:
    # Process UK life expectancy and health inequality data

    @staticmethod
    def preprocess_health_data(input_file, output_dir='datasets/processed'):
        print()
        print("UK HEALTH DATA PREPROCESSING")

        print("\n[1/7] Loading data...")
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

        print("\n[2/7] Exploring data structure...")
        print(f"   Unique local authorities: {df['Area Name'].nunique()}")
        print(f"   Time periods: {df['Time period'].unique()}")
        print(f"   Sex categories: {df['Sex'].unique()}")

        print("\n[3/7] Creating aggregate dataset...")
        overall_df = df[df['Category Type'].isna()].copy()

        male_df = overall_df[overall_df['Sex'] == 'Male'][['Area Code', 'Area Name', 'Value']].copy()
        male_df.rename(columns={'Value': 'Life_Expectancy_Male'}, inplace=True)

        female_df = overall_df[overall_df['Sex'] == 'Female'][['Area Code', 'Area Name', 'Value']].copy()
        female_df.rename(columns={'Value': 'Life_Expectancy_Female'}, inplace=True)

        aggregate_df = pd.merge(male_df, female_df, on=['Area Code', 'Area Name'], how='outer')

        aggregate_df['Life_Expectancy_Gap'] = (
            aggregate_df['Life_Expectancy_Female'] - aggregate_df['Life_Expectancy_Male']
        )

        aggregate_df['Life_Expectancy_Avg'] = (
            aggregate_df['Life_Expectancy_Female'] + aggregate_df['Life_Expectancy_Male']
        ) / 2

        aggregate_df = aggregate_df[aggregate_df['Area Code'] != 'E92000001']
        print(f"   Created aggregate dataset: {len(aggregate_df)} local authorities")

        print("\n[4/7] Creating deprivation inequality dataset...")
        deprivation_df = df[df['Category Type'].notna()].copy()

        inequality_metrics = []
        for area_code in deprivation_df['Area Code'].unique():
            if area_code == 'E92000001':
                continue

            area_data = deprivation_df[deprivation_df['Area Code'] == area_code]
            area_name = area_data['Area Name'].iloc[0]

            male_data = area_data[area_data['Sex'] == 'Male']
            male_most_deprived = male_data[male_data['Category'].str.contains('Most deprived', na=False)]['Value'].values
            male_least_deprived = male_data[male_data['Category'].str.contains('Least deprived', na=False)]['Value'].values

            female_data = area_data[area_data['Sex'] == 'Female']
            female_most_deprived = female_data[female_data['Category'].str.contains('Most deprived', na=False)]['Value'].values
            female_least_deprived = female_data[female_data['Category'].str.contains('Least deprived', na=False)]['Value'].values

            male_gap = male_least_deprived[0] - male_most_deprived[0] if len(male_most_deprived) > 0 and len(male_least_deprived) > 0 else np.nan
            female_gap = female_least_deprived[0] - female_most_deprived[0] if len(female_most_deprived) > 0 and len(female_least_deprived) > 0 else np.nan

            inequality_metrics.append({
                'Area_Code': area_code,
                'Area_Name': area_name,
                'Male_Inequality_Gap': male_gap,
                'Female_Inequality_Gap': female_gap,
                'Avg_Inequality_Gap': np.nanmean([male_gap, female_gap]),
                'Male_Most_Deprived_LE': male_most_deprived[0] if len(male_most_deprived) > 0 else np.nan,
                'Male_Least_Deprived_LE': male_least_deprived[0] if len(male_least_deprived) > 0 else np.nan,
                'Female_Most_Deprived_LE': female_most_deprived[0] if len(female_most_deprived) > 0 else np.nan,
                'Female_Least_Deprived_LE': female_least_deprived[0] if len(female_least_deprived) > 0 else np.nan
            })

        inequality_df = pd.DataFrame(inequality_metrics)
        print(f"   Created inequality dataset: {len(inequality_df)} local authorities")

        print("\n[5/7] Creating master dataset...")
        master_df = pd.merge(aggregate_df, inequality_df, left_on='Area Code', right_on='Area_Code', how='left', suffixes=('', '_drop'))

        cols_to_drop = [col for col in master_df.columns if col.endswith('_drop') or col == 'Area_Code']
        master_df.drop(columns=cols_to_drop, inplace=True)

        if 'Area Code' in master_df.columns:
            master_df.rename(columns={'Area Code': 'Area_Code', 'Area Name': 'Area_Name'}, inplace=True)

        print("\n[6/7] Adding geographic indicators...")
        master_df['Region_Type'] = master_df['Area_Code'].str[:3]
        urban_codes = ['E08', 'E09']
        master_df['Urban_Rural'] = master_df['Region_Type'].apply(
            lambda x: 'Urban' if x in urban_codes else 'Mixed/Rural'
        )

        print("\n[7/7] Performing quality checks and saving...")
        import os
        os.makedirs(output_dir, exist_ok=True)

        master_df.to_csv(f'{output_dir}/master_health_data.csv', index=False)
        aggregate_df.to_csv(f'{output_dir}/aggregate_life_expectancy.csv', index=False)
        inequality_df.to_csv(f'{output_dir}/deprivation_inequality.csv', index=False)

        print(f"\n   ✓ Files saved to '{output_dir}/' directory")
        print("\n" + "=" * 60)
        print("LIFE EXPECTANCY PREPROCESSING COMPLETE!")
        print("=" * 60)

        return master_df, aggregate_df, inequality_df


# Obesity Data Processing

class ObesityAdmissionsPreprocessor:
    # Process NHS obesity hospital admissions data

    def __init__(self, input_path, output_dir='datasets/processed'):
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.processed_df = None

    def load_data(self):
        print("Loading obesity admissions data...")
        self.df = pd.read_csv(self.input_path)
        print(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        return self

    def filter_local_authorities(self):
        print("\nFiltering to local authority level...")
        la_prefixes = ['E06', 'E08', 'E09', 'E10']
        self.df = self.df[self.df['OrgCode'].str.startswith(tuple(la_prefixes))].copy()
        print(f"Filtered to {len(self.df)} rows for {self.df['OrgCode'].nunique()} local authorities")
        return self

    def pivot_by_measure_and_sex(self):
        print("\nPivoting by measure and sex...")
        self.df['feature_name'] = (
            self.df['Measure'].str.replace('FAE_', '').str.replace('FCE_', '').str.lower() +
            '_' +
            self.df['Sex'].str.replace('All persons', 'all').str.replace(' ', '_').str.lower()
        )

        pivot_df = self.df.pivot_table(
            index='OrgCode',
            columns='feature_name',
            values='Stand_Admissions_Rate',
            aggfunc='first'
        ).reset_index()

        pivot_df.columns.name = None
        self.processed_df = pivot_df
        print(f"Created {len(self.processed_df)} rows with {len(self.processed_df.columns) - 1} obesity features")
        return self

    def add_derived_metrics(self):
        print("\nCalculating derived metrics...")

        if 'primary_obesity_female' in self.processed_df.columns and 'primary_obesity_male' in self.processed_df.columns:
            self.processed_df['obesity_gender_gap'] = (
                self.processed_df['primary_obesity_female'] - self.processed_df['primary_obesity_male']
            )

        if 'primary_obesity_bariatric_all' in self.processed_df.columns:
            self.processed_df['bariatric_surgery_rate'] = self.processed_df['primary_obesity_bariatric_all']

        if 'primarysecondary_obesity_all' in self.processed_df.columns:
            self.processed_df['total_obesity_burden'] = self.processed_df['primarysecondary_obesity_all']

        if 'primary_obesity_bariatric_all' in self.processed_df.columns and 'primary_obesity_all' in self.processed_df.columns:
            self.processed_df['severe_obesity_ratio'] = (
                self.processed_df['primary_obesity_bariatric_all'] /
                (self.processed_df['primary_obesity_all'] + 1)
            )

        return self

    def handle_missing_values(self):
        print("\nHandling missing values...")
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            self.processed_df[col].fillna(0, inplace=True)

        return self

    def add_metadata(self):
        print("\nAdding metadata...")
        self.processed_df['data_year'] = '2019/20'
        self.processed_df['data_source'] = 'NHS Digital'

        def get_area_type(code):
            if code.startswith('E06'):
                return 'Unitary Authority'
            elif code.startswith('E08'):
                return 'Metropolitan District'
            elif code.startswith('E09'):
                return 'London Borough'
            elif code.startswith('E10'):
                return 'County'
            else:
                return 'Unknown'

        self.processed_df['area_type'] = self.processed_df['OrgCode'].apply(get_area_type)
        return self

    def rename_orgcode(self):
        self.processed_df.rename(columns={'OrgCode': 'Area Code'}, inplace=True)
        return self

    def save_processed_data(self, filename='obesity_admissions_processed.csv'):
        output_path = self.output_dir / filename

        metadata_cols = ['Area Code', 'data_year', 'data_source', 'area_type']
        feature_cols = [c for c in self.processed_df.columns if c not in metadata_cols]
        column_order = metadata_cols + sorted(feature_cols)
        self.processed_df = self.processed_df[[c for c in column_order if c in self.processed_df.columns]]

        self.processed_df.to_csv(output_path, index=False)
        print(f"\n✓ Processed data saved to: {output_path}")
        return self

    def run_full_pipeline(self):
        print("=" * 70)
        print("OBESITY ADMISSIONS PREPROCESSING PIPELINE")
        print("=" * 70)

        (self.load_data()
         .filter_local_authorities()
         .pivot_by_measure_and_sex()
         .add_derived_metrics()
         .handle_missing_values()
         .add_metadata()
         .rename_orgcode()
         .save_processed_data())

        print("OBESITY PREPROCESSING COMPLETE!")

        return self.processed_df


# Local Authority Finance Data Processing

class LocalAuthorityFinancePreprocessor:
    # Process UK local authority financial data

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_clean = None
        self.spending_categories = {
            'Education services': 'education_spend',
            'Children Social Care': 'children_social_care',
            'Adult Social Care': 'adult_social_care',
            'Public Health': 'public_health_spend',
            'Housing services (GFRA only)': 'housing_spend',
            'Environmental and regulatory services': 'environmental_spend',
            'Cultural and related services': 'cultural_spend',
            'Highways and transport services': 'transport_spend',
            'Planning and development services': 'planning_spend',
            'Central services': 'central_services_spend'
        }
        self.revenue_indicators = {
            'COUNCIL TAX REQUIREMENT (total of lines 805 to 885)': 'council_tax_revenue',
            'NET REVENUE EXPENDITURE': 'total_revenue_expenditure',
            'TOTAL SERVICE EXPENDITURE (total of lines 190 to 698)': 'total_service_expenditure'
        }
        self.metadata_cols = ['ONS Code', 'Local authority', 'Class']

    def load_data(self):
        print("Loading finance data...")
        try:
            self.df = pd.read_csv(self.filepath, encoding='utf-8', thousands=',')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.filepath, encoding='latin-1', thousands=',')
        print(f"✓ Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self

    def clean_column_names(self):
        self.df.columns = self.df.columns.str.strip()
        return self

    def extract_key_features(self):
        print("\nExtracting key features...")
        cols_to_keep = self.metadata_cols.copy()

        for orig_col in self.spending_categories.keys():
            if orig_col in self.df.columns:
                cols_to_keep.append(orig_col)

        for orig_col in self.revenue_indicators.keys():
            if orig_col in self.df.columns:
                cols_to_keep.append(orig_col)

        self.df_clean = self.df[cols_to_keep].copy()
        print(f"✓ Extracted {len(cols_to_keep)} key columns")
        return self

    def rename_columns(self):
        rename_dict = {}
        rename_dict.update(self.spending_categories)
        rename_dict.update(self.revenue_indicators)
        rename_dict.update({
            'ONS Code': 'ons_code',
            'Local authority': 'local_authority',
            'Class': 'authority_type'
        })
        self.df_clean.rename(columns=rename_dict, inplace=True)
        return self

    def convert_to_numeric(self):
        print("\nConverting to numeric values...")
        numeric_cols = [col for col in self.df_clean.columns
                       if col not in ['ons_code', 'local_authority', 'authority_type']]

        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(self.df_clean[col]):
                continue

            try:
                self.df_clean[col] = self.df_clean[col].astype(str).str.replace(',', '')
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce')
            except Exception:
                pass

        print("✓ All values converted successfully!")
        return self

    def handle_missing_values(self):
        spending_cols = list(self.spending_categories.values())

        for col in spending_cols:
            if col in self.df_clean.columns:
                self.df_clean[col].fillna(0, inplace=True)

        revenue_cols = list(self.revenue_indicators.values())
        for col in revenue_cols:
            if col in self.df_clean.columns:
                self.df_clean[col] = self.df_clean.groupby('authority_type')[col].transform(
                    lambda x: x.fillna(x.median())
                )

        return self

    def create_derived_features(self):
        print("\nCreating derived features...")

        if 'total_service_expenditure' in self.df_clean.columns:
            total_spend = self.df_clean['total_service_expenditure']
            spending_cols = list(self.spending_categories.values())

            for col in spending_cols:
                if col in self.df_clean.columns and total_spend.sum() > 0:
                    self.df_clean[f'{col}_pct'] = (self.df_clean[col] / total_spend * 100).round(2)

        health_cols = ['public_health_spend', 'adult_social_care', 'children_social_care']
        health_cols_present = [c for c in health_cols if c in self.df_clean.columns]

        if health_cols_present and 'total_service_expenditure' in self.df_clean.columns:
            self.df_clean['total_health_social_spend'] = self.df_clean[health_cols_present].sum(axis=1)
            self.df_clean['health_spend_ratio'] = (
                self.df_clean['total_health_social_spend'] /
                self.df_clean['total_service_expenditure'] * 100
            ).round(2)

        if 'public_health_spend' in self.df_clean.columns and 'adult_social_care' in self.df_clean.columns:
            self.df_clean['preventive_vs_reactive_ratio'] = (
                self.df_clean['public_health_spend'] /
                (self.df_clean['adult_social_care'] + 1)
            ).round(4)

        return self

    def remove_outliers(self, method='iqr', threshold=3):
        print(f"\nDetecting outliers using {method} method...")
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.df_clean[col].quantile(0.25)
                Q3 = self.df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                self.df_clean[col] = self.df_clean[col].clip(lower=lower_bound, upper=upper_bound)

        return self

    def add_authority_dummies(self):
        print("\nCreating authority type dummies...")
        if 'authority_type' in self.df_clean.columns:
            dummies = pd.get_dummies(self.df_clean['authority_type'], prefix='authority_type')
            self.df_clean = pd.concat([self.df_clean, dummies], axis=1)
        return self

    def validate_data(self):
        print("\n✓ Validating cleaned data...")
        nan_count = self.df_clean.isnull().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Warning: {nan_count} NaN values remain")
        else:
            print("✓ No NaN values")

        print(f"\nFinal dataset: {len(self.df_clean)} authorities")
        print(f"Features: {len(self.df_clean.columns)} columns")
        return self

    def save_cleaned_data(self, output_path='datasets/processed/local_auth_spending.csv'):
        print(f"\nSaving to {output_path}...")
        self.df_clean.to_csv(output_path, index=False)
        print("✓ Saved successfully!")
        return self

    def run_full_pipeline(self, output_path='datasets/processed/local_auth_spending.csv'):
        print("=" * 70)
        print("FINANCE DATA PREPROCESSING PIPELINE")
        print("=" * 70)

        (self
         .load_data()
         .clean_column_names()
         .extract_key_features()
         .rename_columns()
         .convert_to_numeric()
         .handle_missing_values()
         .create_derived_features()
         .remove_outliers()
         .add_authority_dummies()
         .validate_data()
         .save_cleaned_data(output_path))

        print("FINANCE PREPROCESSING COMPLETE!")

        return self.df_clean


def copy_additional_datasets(source_dir='datasets/raw', dest_dir='datasets/processed'):
    """
    Copying additional datasets that don't require preprocessing to the processed folder.

    Files copied:
    - air polution.csv
    - England-annual-price-change-by-local-authority-2021-03.csv
    - ts007aageby5yearagebandslowertierlocalauthorities.csv
    """
    import shutil
    import os

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # List of files to copy
    files_to_copy = [
        'air polution.csv',
        'England-annual-price-change-by-local-authority-2021-03.csv',
        'ts007aageby5yearagebandslowertierlocalauthorities.csv'
    ]

    copied_count = 0
    for filename in files_to_copy:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                print(f"✓ Copied: {filename}")
                copied_count += 1
            else:
                print(f"⚠️  Not found: {filename}")
        except Exception as e:
            print(f"✗ Error copying {filename}: {str(e)}")

    print(f"\n✓ Copied {copied_count}/{len(files_to_copy)} additional datasets")
    print("="*70 + "\n")


# Main Execution & Usage Examples

def main():
    print("\n" + "="*80)
    print("UK HEALTH OUTCOMES PREDICTION - DATA PREPROCESSING SUITE")
    print("="*80)
    print("\nProcessing all datasets...")
    print("="*80 + "\n")

    # Copy additional datasets first
    copy_additional_datasets(source_dir='datasets/raw', dest_dir='datasets/processed')

    census_data = CensusPreprocessor.process_census_demographics(
        ethnicity_path='datasets/raw/TS021-2021-1_ethnic.csv',
        economic_path='datasets/raw/TS066-2021-1_ecoActivity.csv',
        education_path='datasets/raw/TS067-2021-1_eduLevels.csv',
        output_path='datasets/processed/census_demographics_processed.csv'
    )
    print(f"\nCensus data shape: {census_data.shape}")

    master_health, aggregate, inequality = LifeExpectancyPreprocessor.preprocess_health_data(
        input_file='datasets/raw/life_expectancy_data.csv',
        output_dir='datasets/processed'
    )
    print(f"\nLife expectancy data shape: {master_health.shape}")

    obesity_preprocessor = ObesityAdmissionsPreprocessor(
        input_path='datasets/raw/obesity.csv',
        output_dir='datasets/processed'
    )
    obesity_data = obesity_preprocessor.run_full_pipeline()
    print(f"\nObesity data shape: {obesity_data.shape}")

    finance_preprocessor = LocalAuthorityFinancePreprocessor('datasets/raw/LA_spending.csv')
    finance_data = finance_preprocessor.run_full_pipeline(
        output_path='datasets/processed/local_auth_spending.csv'
    )
    print(f"\nFinance data shape: {finance_data.shape}")

    print("\n" + "="*80)
    print("✓ ALL PREPROCESSING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()