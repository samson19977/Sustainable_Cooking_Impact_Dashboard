# utils/data_loader.py - Data loading and cleaning utilities
# =====================================================

import pandas as pd
import numpy as np
import streamlit as st

def load_and_clean_data():
    """Load and clean dataset with proper district name cleaning."""
    try:
        # Load your actual CSV
        df = pd.read_csv(".streamlit/delagua_stove_data_cleaned.csv")
        
        # Clean district names
        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.strip().str.title()
            
            # Fix spelling variations
            district_corrections = {
                'Burера': 'Burera',
                'Gakenki': 'Gakenke',
                'Musanza': 'Musanze',
                'Nyabihi': 'Nyabihu',
                'Rulino': 'Rulindo'
            }
            
            df['district'] = df['district'].replace(district_corrections)
        
        # Handle dates if column exists
        if 'distribution_date' in df.columns:
            def safe_date_parse(date_str):
                try:
                    return pd.to_datetime(date_str, format='%d/%m/%Y')
                except:
                    try:
                        return pd.to_datetime(date_str, format='%Y-%m-%d')
                    except:
                        return pd.to_datetime(date_str, errors='coerce')
            
            df['distribution_date'] = df['distribution_date'].apply(safe_date_parse)
            df['distribution_year'] = df['distribution_date'].dt.year
            df['distribution_month'] = df['distribution_date'].dt.month
            
            df['distribution_year'] = df['distribution_year'].fillna(2023).astype(int)
            df['distribution_month'] = df['distribution_month'].fillna(1).astype(int)
        else:
            df['distribution_year'] = 2023
            df['distribution_month'] = 1
        
        # Ensure required columns exist
        required_columns = {
            'avg_reduction': 0,
            'distance_to_market_km': 0,
            'elevation_m': 1500,
            'household_size': 1,
            'latitude': -1.5,
            'longitude': 29.7,
            'baseline_fuel_kg_person_week': 0
        }
        
        for col, default in required_columns.items():
            if col not in df.columns:
                df[col] = default
        
        # Clean numeric columns
        df['avg_reduction'] = pd.to_numeric(df['avg_reduction'], errors='coerce')
        df['avg_reduction'] = df['avg_reduction'].clip(-100, 100).fillna(0)
        
        # Create performance metrics
        df['low_adoption_risk'] = (df['avg_reduction'] < 30).astype(int)
        
        # Filter to Northern Province districts
        northern_districts = ['Gakenke', 'Musanze', 'Burera', 'Rulindo', 'Nyabihu']
        df = df[df['district'].isin(northern_districts)].copy()
        
        # Create performance categories
        reduction_min = df['avg_reduction'].min()
        reduction_max = df['avg_reduction'].max()
        
        if reduction_max - reduction_min > 0:
            bins = np.linspace(reduction_min, reduction_max, 6)
            labels = ['Very Low', 'Low', 'Moderate', 'Good', 'Excellent']
            df['performance_category'] = pd.cut(df['avg_reduction'], bins=bins, labels=labels)
        else:
            df['performance_category'] = 'Moderate'
        
        # Calculate fuel savings
        df['weekly_fuel_saving_kg'] = (
            df['baseline_fuel_kg_person_week'] * 
            df['household_size'] * 
            (df['avg_reduction'] / 100)
        )
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        # Create sample data as fallback
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    n = 7976
    
    districts = ['Burera', 'Gakenke', 'Musanze', 'Nyabihu', 'Rulindo']
    
    data = {
        'household_id': [f'HH{i:05d}' for i in range(n)],
        'district': np.random.choice(districts, n, p=[0.2, 0.2, 0.3, 0.15, 0.15]),
        'avg_reduction': np.random.uniform(-100, 90, n),
        'distance_to_market_km': np.random.exponential(8, n).clip(0.5, 30),
        'elevation_m': np.random.uniform(1300, 2900, n),
        'household_size': np.random.choice([1,2,3,4,5,6,7,8], n, p=[0.05,0.1,0.2,0.25,0.2,0.1,0.05,0.05]),
        'latitude': np.random.uniform(-1.5, -1.3, n),
        'longitude': np.random.uniform(29.6, 29.9, n),
        'baseline_fuel_kg_person_week': np.random.uniform(5, 12, n),
        'distribution_year': np.random.choice([2023, 2024], n),
        'distribution_month': np.random.randint(1, 13, n)
    }
    
    df = pd.DataFrame(data)
    df['low_adoption_risk'] = (df['avg_reduction'] < 30).astype(int)
    df['weekly_fuel_saving_kg'] = df['baseline_fuel_kg_person_week'] * df['household_size'] * (df['avg_reduction'] / 100)
    
    bins = np.linspace(df['avg_reduction'].min(), df['avg_reduction'].max(), 6)
    labels = ['Very Low', 'Low', 'Moderate', 'Good', 'Excellent']
    df['performance_category'] = pd.cut(df['avg_reduction'], bins=bins, labels=labels)
    
    return df
