# utils/data_loader.py
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_and_process_data(file_path="delagua_stove_data_cleaned.csv"):
    """Load and process the DelAgua data"""
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Identify usage month columns
        usage_cols = [col for col in df.columns if 'usage_month' in col]
        
        if usage_cols and len(usage_cols) >= 1:
            # Convert usage columns to numeric
            for col in usage_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate average monthly usage
            df['avg_monthly_usage_kg'] = df[usage_cols].mean(axis=1, skipna=True)
            
            # Check if we have baseline data
            if 'baseline_fuel_kg_person_week' in df.columns and 'household_size' in df.columns:
                df['baseline_fuel_kg_person_week'] = pd.to_numeric(df['baseline_fuel_kg_person_week'], errors='coerce')
                df['household_size'] = pd.to_numeric(df['household_size'], errors='coerce')
                
                # Calculate expected monthly fuel use
                df['expected_weekly_kg'] = df['baseline_fuel_kg_person_week'] * df['household_size']
                df['expected_monthly_kg'] = df['expected_weekly_kg'] * 4.33
                
                # Calculate reduction percentage
                df['avg_reduction'] = ((df['expected_monthly_kg'] - df['avg_monthly_usage_kg']) / 
                                      df['expected_monthly_kg'].replace(0, np.nan)) * 100
            else:
                max_usage = df[usage_cols].max(axis=1, skipna=True)
                df['avg_reduction'] = ((max_usage - df['avg_monthly_usage_kg']) / 
                                      max_usage.replace(0, np.nan)) * 100
            
            # Handle NaN values and extreme values
            df['avg_reduction'] = df['avg_reduction'].fillna(0)
            df['avg_reduction'] = df['avg_reduction'].replace([np.inf, -np.inf], 0)
            df['avg_reduction'] = df['avg_reduction'].clip(-100, 100)
            
        else:
            df['avg_reduction'] = 0
        
        # Create essential columns
        df['low_adoption_risk'] = (df['avg_reduction'] < 30).astype(int)
        
        # Calculate fuel savings
        if 'household_size' in df.columns and 'baseline_fuel_kg_person_week' in df.columns:
            df['weekly_fuel_saving_kg'] = df['baseline_fuel_kg_person_week'] * df['household_size'] * (df['avg_reduction'] / 100)
        else:
            df['weekly_fuel_saving_kg'] = 8 * (df['avg_reduction'] / 100)
        
        # Create adoption categories
        df['adoption_category'] = pd.cut(
            df['avg_reduction'],
            bins=[-float('inf'), 30, 50, 70, 85, float('inf')],
            labels=['Low (<30%)', 'Moderate (30-50%)', 'Good (50-70%)', 'High (70-85%)', 'Excellent (>85%)']
        )
        
        # Handle missing geographic data
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            df['latitude'] = -1.4
            df['longitude'] = 29.7
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()
