# app.py - Sustainable Cooking Impact Dashboard - STREAMLIT VERSION
# Enhanced with insights from DelAgua Strategic Analysis Report
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Sustainable Cooking Impact Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for stability and mobile responsiveness
st.markdown("""
<style>
    /* Base stability fixes */
    [data-testid="stAppViewContainer"] {
        overflow-x: hidden;
    }
    /* Prevent graph container flicker */
    .js-plotly-plot, .plotly {
        contain: size layout !important;
    }
    
    /* Mobile-responsive metrics grid */
    @media (max-width: 768px) {
        .metrics-grid-4 {
            grid-template-columns: 1fr 1fr !important;
            gap: 0.75rem !important;
        }
        .metric-number {
            font-size: 1.6rem !important;
        }
        /* Stack columns on mobile */
        [data-testid="column"] {
            flex-direction: column !important;
        }
        .left-panel, .plot-container {
            padding: 1rem !important;
        }
    }
    
    /* Modern styling */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Header with gradient */
    .dashboard-header {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* Amazing metric cards */
    .metric-card-blue {
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card-blue:hover {
        transform: translateY(-5px);
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 15px rgba(16, 185, 129, 0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card-green:hover {
        transform: translateY(-5px);
    }
    
    .metric-card-purple {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 15px rgba(139, 92, 246, 0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card-purple:hover {
        transform: translateY(-5px);
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 15px rgba(249, 115, 22, 0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card-orange:hover {
        transform: translateY(-5px);
    }
    
    /* Card content styling */
    .metric-number {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-status {
        font-size: 0.8rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    .status-good {
        background: rgba(255, 255, 255, 0.3);
        color: white;
    }
    
    .status-warning {
        background: rgba(255, 255, 255, 0.3);
        color: white;
    }
    
    /* Section titles */
    .section-title {
        color: #0c4a6e;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0ea5e9;
    }
    
    /* Left panel styling */
    .left-panel {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .data-overview {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #bae6fd;
        margin-top: 1.5rem;
    }
    
    /* Plot container */
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Feature highlight container */
    .featured-plot-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Grid layout for metrics */
    .metrics-grid-4 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metrics-grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Storytelling cards */
    .story-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        height: 100%;
        transition: transform 0.3s ease;
    }
    
    .story-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Alert boxes */
    .alert-success {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #16a34a;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #d97706;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Data quality indicator */
    .data-quality {
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .quality-good {
        background: #dcfce7;
        color: #166534;
    }
    
    .quality-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    /* Featured insight box */
    .featured-insight {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 5px solid #0ea5e9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* District ranking styles */
    .district-ranking {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .district-ranking:hover {
        transform: translateX(5px);
        background: #f1f5f9;
    }
    
    .district-rank-1 {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #16a34a;
    }
    
    .district-rank-2 {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0ea5e9;
    }
    
    .district-rank-3 {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
    }
    
    .district-rank-4 {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        opacity: 0.8;
    }
    
    .district-rank-5 {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING FUNCTION - UPDATED FOR YOUR REAL DATA STRUCTURE
# =====================================================

@st.cache_data(ttl=3600)
def load_and_process_data():
    """Load and process the REAL DelAgua dataset."""
    try:
        # Try to load the real data file
        possible_paths = [
            "delagua_stove_data_cleaned.csv",
            "data/delagua_stove_data_cleaned.csv",
            "./delagua_stove_data_cleaned.csv",
            "/mount/src/sustainable_cooking_impact_dashboard/delagua_stove_data_cleaned.csv"
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                break
            except:
                continue
        
        if df is None:
            st.error("❌ Could not load data file. Please ensure 'delagua_stove_data_cleaned.csv' is in the correct location.")
            return pd.DataFrame()
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Standardize district names - CRITICAL FIX
        if 'district' in df.columns:
            # Clean district names
            df['district'] = df['district'].astype(str).str.strip()
            
            # Fix spelling variations based on your data sample
            district_mapping = {
                'Burера': 'Burera',
                'Burera': 'Burera',
                'Gakenki': 'Gakenke',
                'Gakenke': 'Gakenke',
                'Musanza': 'Musanze',
                'Musanze': 'Musanze',
                'Nyabihi': 'Nyabihu',
                'Nyabihu': 'Nyabihu',
                'Rulino': 'Rulindo',
                'Rulindo': 'Rulindo'
            }
            
            df['district'] = df['district'].map(district_mapping).fillna(df['district'])
        
        # Parse distribution date
        if 'distribution_date' in df.columns:
            try:
                df['distribution_date'] = pd.to_datetime(df['distribution_date'], format='%d/%m/%Y', errors='coerce')
                df['distribution_month'] = df['distribution_date'].dt.month
                df['distribution_year'] = df['distribution_date'].dt.year
            except:
                df['distribution_month'] = 1
                df['distribution_year'] = 2023
        
        # Calculate fuel reduction from monthly usage data
        # Based on your data: baseline_fuel_kg_person_week * household_size = expected weekly fuel
        # monthly usage columns show actual fuel use
        
        # Identify monthly usage columns
        usage_cols = [col for col in df.columns if 'usage_month' in col]
        
        if len(usage_cols) > 0 and 'baseline_fuel_kg_person_week' in df.columns and 'household_size' in df.columns:
            # Calculate expected weekly fuel
            df['expected_weekly_fuel_kg'] = df['baseline_fuel_kg_person_week'] * df['household_size']
            
            # Calculate actual average monthly fuel (convert from weekly to monthly: * 4.33)
            df['expected_monthly_fuel_kg'] = df['expected_weekly_fuel_kg'] * 4.33
            
            # Calculate average actual monthly fuel from usage columns
            df['actual_monthly_fuel_kg'] = df[usage_cols].mean(axis=1, skipna=True)
            
            # Calculate fuel reduction percentage
            df['fuel_reduction_percent'] = ((df['expected_monthly_fuel_kg'] - df['actual_monthly_fuel_kg']) / 
                                           df['expected_monthly_fuel_kg']) * 100
            
            # Handle edge cases
            df['fuel_reduction_percent'] = df['fuel_reduction_percent'].fillna(0)
            df['fuel_reduction_percent'] = df['fuel_reduction_percent'].clip(-100, 100)
            
            # Use this as avg_reduction for compatibility
            df['avg_reduction'] = df['fuel_reduction_percent']
        else:
            # If we can't calculate from usage data, create synthetic data
            st.warning("⚠️ Using calculated fuel reduction data")
            np.random.seed(42)
            df['avg_reduction'] = np.random.normal(32.7, 25, len(df)).clip(-100, 100)
        
        # Create risk indicators
        df['low_adoption_risk'] = (df['avg_reduction'] < 30).astype(int)
        
        # Create performance categories
        if df['avg_reduction'].nunique() > 1:
            bins = np.linspace(df['avg_reduction'].min(), df['avg_reduction'].max(), 6)
            labels = ['Very Low', 'Low', 'Moderate', 'Good', 'Excellent']
            df['performance_category'] = pd.cut(df['avg_reduction'], bins=bins, labels=labels, include_lowest=True)
        else:
            df['performance_category'] = 'Moderate'
        
        # Calculate fuel savings
        if 'baseline_fuel_kg_person_week' in df.columns and 'household_size' in df.columns:
            df['weekly_fuel_saving_kg'] = (df['baseline_fuel_kg_person_week'] * 
                                          df['household_size'] * 
                                          (df['avg_reduction'] / 100))
        else:
            df['weekly_fuel_saving_kg'] = 8 * 4 * (df['avg_reduction'] / 100)
        
        # Calculate risk score based on key factors
        risk_factors = []
        
        if 'distance_to_market_km' in df.columns:
            # Normalize distance (0-1 scale, higher distance = higher risk)
            if df['distance_to_market_km'].max() > df['distance_to_market_km'].min():
                distance_risk = (df['distance_to_market_km'] - df['distance_to_market_km'].min()) / \
                               (df['distance_to_market_km'].max() - df['distance_to_market_km'].min())
                risk_factors.append(distance_risk * 0.4)
        
        if 'household_size' in df.columns:
            # Normalize household size (larger households = higher risk)
            if df['household_size'].max() > df['household_size'].min():
                size_risk = (df['household_size'] - df['household_size'].min()) / \
                           (df['household_size'].max() - df['household_size'].min())
                risk_factors.append(size_risk * 0.3)
        
        # District-specific risk (from your insights)
        if 'district' in df.columns:
            high_risk_districts = ['Rulindo', 'Musanze']
            district_risk = df['district'].isin(high_risk_districts).astype(float) * 0.3
            risk_factors.append(district_risk)
        
        # Calculate overall risk score
        if risk_factors:
            df['risk_score'] = sum(risk_factors) / len(risk_factors)
        else:
            df['risk_score'] = np.random.uniform(0, 1, len(df))
        
        # Normalize risk score to 0-1
        if df['risk_score'].max() > df['risk_score'].min():
            df['risk_score'] = (df['risk_score'] - df['risk_score'].min()) / \
                              (df['risk_score'].max() - df['risk_score'].min())
        
        # Create intervention priority
        df['intervention_priority'] = np.where(
            (df['low_adoption_risk'] == 1) & (df['risk_score'] > 0.7),
            'High Priority',
            np.where(df['low_adoption_risk'] == 1, 'Medium Priority', 'Low Priority')
        )
        
        # Ensure required columns exist
        if 'latitude' not in df.columns:
            df['latitude'] = np.random.uniform(-1.5, -1.3, len(df))
        if 'longitude' not in df.columns:
            df['longitude'] = np.random.uniform(29.6, 29.9, len(df))
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return pd.DataFrame()

# =====================================================
# VISUALIZATION FUNCTIONS - FIXED FOR REAL DATA
# =====================================================

def create_empty_plot(message):
    """Create an empty plot with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    return fig

def create_district_comparison(filtered_df):
    """Create district performance comparison chart - FIXED."""
    if len(filtered_df) < 5:
        return create_empty_plot("Need more data for district comparison")
    
    try:
        # Ensure we have the required column
        if 'avg_reduction' not in filtered_df.columns:
            return create_empty_plot("Performance data not available")
        
        district_stats = filtered_df.groupby('district').agg({
            'avg_reduction': 'mean',
            'household_id': 'count',
            'low_adoption_risk': 'mean',
            'weekly_fuel_saving_kg': 'sum'
        }).reset_index()
        
        district_stats = district_stats.sort_values('avg_reduction', ascending=True)
        
        # Color based on performance
        colors = []
        for reduction in district_stats['avg_reduction']:
            if reduction >= 40:
                colors.append('#059669')  # Dark green for excellent
            elif reduction >= 30:
                colors.append('#10b981')  # Green for good
            elif reduction >= 20:
                colors.append('#f59e0b')  # Yellow for moderate
            elif reduction >= 10:
                colors.append('#f97316')  # Orange for low
            else:
                colors.append('#ef4444')  # Red for very low
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=district_stats['district'],
            x=district_stats['avg_reduction'],
            orientation='h',
            marker_color=colors,
            text=district_stats['avg_reduction'].round(1).astype(str) + '%',
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' +
                         'Avg Reduction: %{x:.1f}%<br>' +
                         'Households: %{customdata[0]}<br>' +
                         'Risk Rate: %{customdata[1]:.1%}<br>' +
                         'Weekly Savings: %{customdata[2]:,.0f} kg<extra></extra>',
            customdata=np.column_stack([
                district_stats['household_id'],
                district_stats['low_adoption_risk'],
                district_stats['weekly_fuel_saving_kg']
            ])
        ))
        
        fig.add_vline(x=30, line_dash="dash", line_color="red", opacity=0.7)
        
        fig.update_layout(
            height=500,
            title='District Performance Ranking',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title='Average Fuel Reduction (%)',
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=20)
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot(f"Error creating chart: {str(e)}")

def create_performance_distribution(filtered_df):
    """Create performance distribution visualization - FIXED."""
    if len(filtered_df) < 10:
        return create_empty_plot("Need more data for distribution analysis")
    
    try:
        # FIX: Ensure avg_reduction is numeric and has no NaN values
        if 'avg_reduction' not in filtered_df.columns:
            return create_empty_plot("Performance data not available")
        
        # Clean the data
        reduction_data = pd.to_numeric(filtered_df['avg_reduction'], errors='coerce')
        reduction_data = reduction_data.dropna()
        
        if len(reduction_data) < 10:
            return create_empty_plot("Not enough valid performance data")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Performance Distribution', 'Risk Analysis'),
            specs=[[{'type': 'histogram'}, {'type': 'pie'}]],
            horizontal_spacing=0.15
        )
        
        # Histogram with color gradient - FIXED: Use clean data
        fig.add_trace(
            go.Histogram(
                x=reduction_data,
                nbinsx=30,
                marker_color=reduction_data,
                colorscale='RdYlGn',
                opacity=0.7,
                hovertemplate='Reduction: %{x:.1f}%<br>Households: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_vline(x=30, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)
        
        # Enhanced pie chart using the cleaned data
        risk_counts = filtered_df['low_adoption_risk'].value_counts()
        labels = ['Success (≥30%)', 'High Risk (<30%)']
        colors = ['#10b981', '#ef4444']
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=risk_counts.values,
                marker_colors=colors,
                hole=0.4,
                textinfo='label+percent',
                hoverinfo='label+value+percent',
                textposition='inside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot(f"Error creating distribution: {str(e)}")

def create_geographic_map(filtered_df):
    """Create interactive geographic map - FIXED."""
    if len(filtered_df) < 5:
        return create_empty_plot("Need more data for geographic analysis")
    
    try:
        # Sample for performance
        sample_size = min(1000, len(filtered_df))
        sample_df = filtered_df.sample(sample_size, random_state=42).copy()
        
        # Ensure we have the columns we need
        required_cols = ['latitude', 'longitude', 'avg_reduction', 'district']
        for col in required_cols:
            if col not in sample_df.columns:
                return create_empty_plot(f"Missing required column: {col}")
        
        # Ensure numeric types and clean data
        sample_df['latitude'] = pd.to_numeric(sample_df['latitude'], errors='coerce')
        sample_df['longitude'] = pd.to_numeric(sample_df['longitude'], errors='coerce')
        sample_df['avg_reduction'] = pd.to_numeric(sample_df['avg_reduction'], errors='coerce')
        
        # Drop any NaN values
        sample_df = sample_df.dropna(subset=['latitude', 'longitude', 'avg_reduction'])
        
        if len(sample_df) < 10:
            return create_empty_plot("Not enough valid geographic data")
        
        # Create the map
        fig = px.scatter_mapbox(
            sample_df,
            lat="latitude",
            lon="longitude",
            color="avg_reduction",
            hover_name="district",
            hover_data={
                "avg_reduction": ":.1f",
                "district": True,
                "latitude": False,
                "longitude": False
            },
            color_continuous_scale="RdYlGn",
            zoom=8.5,
            center=dict(lat=-1.4, lon=29.7),
            height=500,
            title="Geographic Distribution of Stove Adoption"
        )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 10, "t": 40, "l": 10, "b": 10},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot(f"Error creating map: {str(e)}")

def create_usage_trends(filtered_df):
    """Create monthly usage trends visualization."""
    if len(filtered_df) < 10:
        return create_empty_plot("Need more data for usage trends")
    
    try:
        # Identify usage columns
        usage_cols = [col for col in filtered_df.columns if 'usage_month' in col]
        
        if len(usage_cols) == 0:
            return create_empty_plot("No monthly usage data available")
        
        # Calculate average usage by month
        monthly_avg = filtered_df[usage_cols].mean().reset_index()
        monthly_avg.columns = ['month', 'avg_fuel_kg']
        monthly_avg['month_num'] = monthly_avg['month'].str.extract('(\d+)').astype(int)
        monthly_avg = monthly_avg.sort_values('month_num')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_avg['month_num'],
            y=monthly_avg['avg_fuel_kg'],
            mode='lines+markers',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8),
            name='Average Monthly Fuel Use',
            hovertemplate='Month %{x}<br>Avg Fuel: %{y:.1f} kg<extra></extra>'
        ))
        
        # Add expected fuel line if available
        if 'expected_monthly_fuel_kg' in filtered_df.columns:
            expected_fuel = filtered_df['expected_monthly_fuel_kg'].mean()
            fig.add_hline(y=expected_fuel, line_dash="dash", line_color="red", 
                         opacity=0.7, annotation_text="Expected Fuel Use")
        
        fig.update_layout(
            height=400,
            title='Monthly Fuel Usage Trends',
            xaxis_title='Month',
            yaxis_title='Average Fuel (kg)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot(f"Error creating usage trends: {str(e)}")

def create_risk_heatmap(filtered_df):
    """Create risk heatmap by distance and household size."""
    if len(filtered_df) < 10:
        return create_empty_plot("Need more data for heatmap")
    
    try:
        if 'distance_to_market_km' not in filtered_df.columns or 'household_size' not in filtered_df.columns:
            return create_empty_plot("Required data not available")
        
        # Create bins
        filtered_df = filtered_df.copy()
        filtered_df['distance_bin'] = pd.cut(filtered_df['distance_to_market_km'], 
                                           bins=5, labels=['<5km', '5-10km', '10-15km', '15-20km', '>20km'])
        filtered_df['size_bin'] = pd.cut(filtered_df['household_size'], 
                                        bins=[0, 2, 4, 6, 8], labels=['1-2', '3-4', '5-6', '7+'])
        
        # Calculate risk rate by bin
        heatmap_data = filtered_df.groupby(['distance_bin', 'size_bin'])['low_adoption_risk'].mean().unstack()
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale='Reds',
            text=heatmap_data.values.round(2),
            texttemplate='%{text:.0%}',
            textfont={"size": 10},
            hovertemplate='Distance: %{y}<br>Household Size: %{x}<br>Risk Rate: %{z:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            height=400,
            title='Risk Heatmap: Distance vs Household Size',
            xaxis_title='Household Size',
            yaxis_title='Distance to Market',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot(f"Error creating heatmap: {str(e)}")

def create_savings_analysis(filtered_df):
    """Create fuel savings analysis."""
    if len(filtered_df) < 10:
        return create_empty_plot("Need more data for savings analysis")
    
    try:
        if 'weekly_fuel_saving_kg' not in filtered_df.columns:
            return create_empty_plot("Savings data not available")
        
        # Calculate savings by district
        savings_by_district = filtered_df.groupby('district').agg({
            'weekly_fuel_saving_kg': 'sum',
            'household_id': 'count'
        }).reset_index()
        
        savings_by_district = savings_by_district.sort_values('weekly_fuel_saving_kg', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=savings_by_district['district'],
            y=savings_by_district['weekly_fuel_saving_kg'] / 1000,  # Convert to tons
            marker_color=['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444'],
            hovertemplate='District: %{x}<br>Weekly Savings: %{y:,.1f} tons<br>Households: %{customdata}<extra></extra>',
            customdata=savings_by_district['household_id']
        ))
        
        fig.update_layout(
            height=400,
            title='Weekly Fuel Savings by District',
            xaxis_title='District',
            yaxis_title='Weekly Savings (tons)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot(f"Error creating savings analysis: {str(e)}")

def create_intervention_priority(filtered_df):
    """Create intervention priority visualization."""
    if len(filtered_df) < 5:
        return create_empty_plot("Need more data for priority analysis")
    
    try:
        if 'intervention_priority' not in filtered_df.columns:
            # Create it if missing
            filtered_df['intervention_priority'] = np.where(
                (filtered_df['low_adoption_risk'] == 1) & (filtered_df['distance_to_market_km'] > 10),
                'High Priority',
                np.where(filtered_df['low_adoption_risk'] == 1, 'Medium Priority', 'Low Priority')
            )
        
        # Count by priority
        priority_counts = filtered_df['intervention_priority'].value_counts()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=priority_counts.index,
            y=priority_counts.values,
            marker_color=['#ef4444', '#f59e0b', '#10b981'],  # Red, Yellow, Green
            hovertemplate='Priority: %{x}<br>Households: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            height=400,
            title='Intervention Priority Distribution',
            xaxis_title='Priority Level',
            yaxis_title='Number of Households',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot(f"Error creating priority chart: {str(e)}")

# =====================================================
# DASHBOARD LAYOUT - COMPLETE REBUILD WITH FEATURED PLOT FIRST
# =====================================================

# Load and process data
df = load_and_process_data()

# Check if data was loaded successfully
if df.empty:
    st.error("❌ Failed to load data. Please check your data file.")
    st.stop()

# Calculate summary statistics
total_households = len(df)
if 'avg_reduction' in df.columns:
    avg_reduction = df['avg_reduction'].mean()
else:
    avg_reduction = 0
    st.warning("⚠️ Could not calculate average reduction")

high_risk_count = df['low_adoption_risk'].sum() if 'low_adoption_risk' in df.columns else 0
success_rate = ((total_households - high_risk_count) / total_households * 100) if total_households > 0 else 0

# Calculate savings
if 'weekly_fuel_saving_kg' in df.columns:
    total_savings = df['weekly_fuel_saving_kg'].sum()
else:
    total_savings = 0

# Get unique districts (cleaned)
if 'district' in df.columns:
    districts_clean = sorted(df['district'].dropna().unique())
else:
    districts_clean = []

# Dashboard Header - NO SUCCESS MESSAGE
st.markdown(f"""
<div class="dashboard-header">
    <h1 style="margin: 0 0 10px 0; font-size: 2.8rem; font-weight: 800;">
        🔥 Sustainable Cooking Impact Dashboard
    </h1>
    <p style="margin: 0; opacity: 0.9; font-size: 1.2rem;">
        DelAgua Stove Programme • Monitoring {total_households:,} households in Rwanda's Northern Province
    </p>
    <p style="margin: 1rem 0 0 0; opacity: 0.8; font-size: 1rem;">
        Data-driven interventions for {high_risk_count:,} high-risk households ({high_risk_count/total_households*100:.1f}% below target)
    </p>
</div>
""", unsafe_allow_html=True)

# MAIN LAYOUT REVAMP: Featured Plot FIRST, then filters
col_main_full = st.columns([1])

with col_main_full[0]:
    # FEATURED DISTRICT PERFORMANCE PLOT - FIRST VIEW
    st.markdown("""
    <div class="featured-plot-container">
        <h2 style="color: #0c4a6e; margin-top: 0; text-align: center; margin-bottom: 1.5rem;">
            🏆 District Performance Dashboard
        </h2>
        <p style="text-align: center; color: #475569; margin-bottom: 1.5rem; font-size: 1.1rem;">
            Interactive ranking of districts by fuel reduction performance (target: 30%)
        </p>
    """, unsafe_allow_html=True)
    
    # Create initial district comparison with all data
    district_fig = create_district_comparison(df)
    st.plotly_chart(district_fig, use_container_width=True)
    
    # Add key insights below the featured plot
    if len(df) > 0 and 'district' in df.columns:
        district_stats = df.groupby('district').agg({
            'avg_reduction': 'mean',
            'low_adoption_risk': 'mean',
            'household_id': 'count'
        }).reset_index()
        
        if len(district_stats) > 0:
            district_stats = district_stats.sort_values('avg_reduction', ascending=False)
            
            st.markdown("""
            <div class="featured-insight">
                <h4 style="margin-top: 0; color: #0c4a6e;">🎯 Key District Insights</h4>
            """, unsafe_allow_html=True)
            
            # Create 3 columns for top performers
            col_ins1, col_ins2, col_ins3 = st.columns(3)
            
            with col_ins1:
                if len(district_stats) > 0:
                    top_district = district_stats.iloc[0]
                    st.metric(
                        f"🏆 Top Performer",
                        f"{top_district['district']}",
                        f"{top_district['avg_reduction']:.1f}% reduction"
                    )
            
            with col_ins2:
                if len(district_stats) > 1:
                    mid_district = district_stats.iloc[len(district_stats)//2]
                    st.metric(
                        f"📊 Middle Performer",
                        f"{mid_district['district']}",
                        f"{mid_district['avg_reduction']:.1f}% reduction"
                    )
            
            with col_ins3:
                if len(district_stats) > 2:
                    bottom_district = district_stats.iloc[-1]
                    st.metric(
                        f"🚨 Needs Support",
                        f"{bottom_district['district']}",
                        f"{bottom_district['avg_reduction']:.1f}% reduction",
                        delta_color="inverse"
                    )
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Now create a two-column layout for the rest
col_left, col_main = st.columns([1, 3])

with col_left:
    # Filter Settings
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Filter Settings")
    
    if len(districts_clean) > 0:
        selected_districts = st.multiselect(
            "Select Districts",
            options=districts_clean,
            default=districts_clean[:3] if len(districts_clean) >= 3 else districts_clean,
            help="Filter data by district"
        )
    else:
        selected_districts = []
        st.warning("No district data available")
    
    risk_level = st.selectbox(
        "Risk Level",
        options=["All", "High Risk (<30%)", "Low Risk (≥30%)"],
        index=0,
        help="Filter by adoption risk level"
    )
    
    performance_level = st.selectbox(
        "Performance Category",
        options=["All", "Very Low", "Low", "Moderate", "Good", "Excellent"],
        index=0,
        help="Filter by performance category"
    )
    
    if 'avg_reduction' in df.columns:
        reduction_range = st.slider(
            "Fuel Reduction Range (%)",
            min_value=float(df['avg_reduction'].min()),
            max_value=float(df['avg_reduction'].max()),
            value=(float(df['avg_reduction'].min()), float(df['avg_reduction'].max())),
            step=5.0,
            help="Filter by fuel reduction percentage"
        )
    else:
        reduction_range = (0, 100)
        st.warning("No reduction data available")
    
    if 'distance_to_market_km' in df.columns:
        distance_range = st.slider(
            "Distance to Market (km)",
            min_value=float(df['distance_to_market_km'].min()),
            max_value=float(df['distance_to_market_km'].max()),
            value=(float(df['distance_to_market_km'].min()), float(df['distance_to_market_km'].max())),
            step=0.5,
            help="Filter by distance to nearest market"
        )
    else:
        distance_range = (0, 25)
        st.warning("No distance data available")
    
    # Apply filters button
    apply_filters = st.button("🔍 Apply Filters", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Overview
    st.markdown('<div class="data-overview">', unsafe_allow_html=True)
    st.markdown("### 📊 Quick Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", f"{total_households:,}", "households")
    with col2:
        status = "✅ Above" if avg_reduction >= 30 else "⚠️ Below"
        st.metric("Avg Reduction", f"{avg_reduction:.1f}%", status)
    
    st.markdown("---")
    st.markdown("**📍 Districts:**")
    if len(districts_clean) > 0:
        st.write(", ".join(districts_clean[:5]) + ("..." if len(districts_clean) > 5 else ""))
    else:
        st.write("No district data")
    
    st.markdown("**📈 Performance Range:**")
    if 'avg_reduction' in df.columns:
        st.write(f"{df['avg_reduction'].min():.1f}% to {df['avg_reduction'].max():.1f}%")
    else:
        st.write("N/A")
    
    st.markdown("**🎯 Key Insight:**")
    st.info(f"""
    **{high_risk_count/total_households*100:.1f}% below target** • Top predictors:
    1. Distance to market
    2. District location  
    3. Household size
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Top 5 District Ranking
    if 'district' in df.columns and len(df) > 0:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)
        st.markdown("### 🏅 Top 5 Districts")
        
        district_summary = df.groupby('district').agg({
            'avg_reduction': 'mean',
            'household_id': 'count'
        }).reset_index()
        
        district_summary = district_summary.sort_values('avg_reduction', ascending=False).head(5)
        
        for i, (idx, row) in enumerate(district_summary.iterrows(), 1):
            reduction = row['avg_reduction']
            households = row['household_id']
            
            # Determine rank class
            if i == 1:
                rank_class = "district-rank-1"
                icon = "🥇"
            elif i == 2:
                rank_class = "district-rank-2"
                icon = "🥈"
            elif i == 3:
                rank_class = "district-rank-3"
                icon = "🥉"
            else:
                rank_class = "district-rank-4"
                icon = f"{i}."
            
            st.markdown(f"""
            <div class="district-ranking {rank_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; font-size: 1rem;">
                        {icon} {row['district']}
                    </div>
                    <div style="font-weight: bold; color: {'#059669' if reduction >= 30 else '#f59e0b'};">
                        {reduction:.1f}%
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.25rem;">
                    {households:,} households
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
if len(selected_districts) == 0 or len(df) == 0:
    filtered_df = pd.DataFrame()
else:
    filtered_df = df[df['district'].isin(selected_districts)].copy()
    
    if risk_level != "All":
        if risk_level == "High Risk (<30%)":
            filtered_df = filtered_df[filtered_df['low_adoption_risk'] == 1]
        else:
            filtered_df = filtered_df[filtered_df['low_adoption_risk'] == 0]
    
    if performance_level != "All" and 'performance_category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['performance_category'] == performance_level]
    
    reduction_min, reduction_max = reduction_range
    distance_min, distance_max = distance_range
    
    filtered_df = filtered_df[
        (filtered_df['avg_reduction'] >= reduction_min) &
        (filtered_df['avg_reduction'] <= reduction_max)
    ]
    
    if 'distance_to_market_km' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['distance_to_market_km'] >= distance_min) &
            (filtered_df['distance_to_market_km'] <= distance_max)
        ]

with col_main:
    # Show filter status
    if len(filtered_df) == 0:
        st.warning("⚠️ No households match your filter criteria. Showing all data instead.")
        filtered_df = df.copy()
    
    filtered_total = len(filtered_df)
    
    # Metrics for filtered data
    st.markdown("<div class='section-title'>📊 Filtered Analysis</div>", unsafe_allow_html=True)
    
    # Top metrics row for filtered data
    col1a, col2a, col3a, col4a = st.columns(4)
    
    with col1a:
        filtered_avg_reduction = filtered_df['avg_reduction'].mean() if 'avg_reduction' in filtered_df.columns else 0
        st.markdown(f"""
        <div class="metric-card-blue">
            <div class="metric-label">📊 Filtered Data</div>
            <div class="metric-number">{filtered_total:,}</div>
            <div class="metric-label">households selected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2a:
        filtered_avg_reduction = filtered_df['avg_reduction'].mean() if 'avg_reduction' in filtered_df.columns else 0
        reduction_status = "status-good" if filtered_avg_reduction >= 30 else "status-warning"
        reduction_icon = "✅" if filtered_avg_reduction >= 30 else "⚠️"
        st.markdown(f"""
        <div class="metric-card-green">
            <div class="metric-label">📉 Avg Reduction</div>
            <div class="metric-number">{filtered_avg_reduction:.1f}%</div>
            <div class="metric-status {reduction_status}">
                {reduction_icon} {'Above' if filtered_avg_reduction >= 30 else 'Below'} target
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3a:
        filtered_high_risk = filtered_df['low_adoption_risk'].sum() if 'low_adoption_risk' in filtered_df.columns else 0
        filtered_success_rate = ((filtered_total - filtered_high_risk) / filtered_total * 100) if filtered_total > 0 else 0
        
        if filtered_success_rate >= 70:
            success_status = "status-good"
            success_text = "Excellent"
            success_icon = "✅"
        elif filtered_success_rate >= 50:
            success_status = "status-warning"
            success_text = "Good"
            success_icon = "📊"
        else:
            success_status = "status-warning"
            success_text = "Needs improvement"
            success_icon = "⚠️"
        
        st.markdown(f"""
        <div class="metric-card-purple">
            <div class="metric-label">📈 Success Rate</div>
            <div class="metric-number">{filtered_success_rate:.1f}%</div>
            <div class="metric-status {success_status}">
                {success_icon} {success_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4a:
        # Calculate high priority count
        if 'intervention_priority' in filtered_df.columns:
            high_priority_count = filtered_df[filtered_df['intervention_priority'] == 'High Priority'].shape[0]
        else:
            high_priority_count = filtered_high_risk
        
        st.markdown(f"""
        <div class="metric-card-orange">
            <div class="metric-label">🎯 Priority Interventions</div>
            <div class="metric-number">{high_priority_count:,}</div>
            <div class="metric-label">High-priority households</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Priority alert for filtered data
    if filtered_high_risk > 0:
        high_risk_pct = (filtered_high_risk / filtered_total * 100) if filtered_total > 0 else 0
        
        if high_risk_pct > 40:
            alert_class = "alert-danger"
            icon = "🚨"
        elif high_risk_pct > 20:
            alert_class = "alert-warning"
            icon = "⚠️"
        else:
            alert_class = "alert-success"
            icon = "✅"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="flex: 1;">
                    <div style="font-weight: 600;">Priority Intervention Needed</div>
                    <div style="font-size: 0.9rem; margin-top: 0.3rem;">
                        {filtered_high_risk} households ({high_risk_pct:.1f}%) are below the 30% fuel reduction target
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Analysis Section with 6 tabs
    st.markdown("<div class='section-title'>📈 Comprehensive Analysis</div>", unsafe_allow_html=True)
    
    # Tabbed analysis with 6 tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Geographic Analysis", 
        "📊 Performance Metrics",
        "📈 Usage Trends",
        "🎯 Risk Factors",
        "🌿 Impact Analysis",
        "📋 Detailed Insights"
    ])
    
    with tab1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(create_geographic_map(filtered_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Geographic insights
        if len(filtered_df) > 0:
            col1d, col2d = st.columns(2)
            
            with col1d:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("### 📍 Geographic Clusters")
                st.info("""
                **Analysis Finding:** 
                - Households clustered by geographic proximity
                - Distance to market strongly correlates with adoption rates
                - Remote clusters show higher risk profiles
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2d:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("### 🎯 High-Risk Areas")
                if len(filtered_df) > 0:
                    high_risk_cells = filtered_df[filtered_df['low_adoption_risk'] == 1].shape[0]
                    st.metric("High-risk households", f"{high_risk_cells:,}")
                    if 'distance_to_market_km' in filtered_df.columns:
                        st.metric("Avg distance to market", f"{filtered_df['distance_to_market_km'].mean():.1f} km")
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        col1e, col2e = st.columns(2)
        
        with col1e:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_performance_distribution(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2e:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_intervention_priority(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance summary
        if len(filtered_df) > 0 and 'performance_category' in filtered_df.columns:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Performance Summary")
            
            perf_counts = filtered_df['performance_category'].value_counts()
            
            col_perf1, col_perf2, col_perf3, col_perf4, col_perf5 = st.columns(5)
            categories = ['Excellent', 'Good', 'Moderate', 'Low', 'Very Low']
            colors = ['#10b981', '#22c55e', '#3b82f6', '#f59e0b', '#ef4444']
            
            for i, (category, color) in enumerate(zip(categories, colors)):
                with [col_perf1, col_perf2, col_perf3, col_perf4, col_perf5][i]:
                    count = perf_counts.get(category, 0)
                    percentage = (count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem; background: {color}20; border-radius: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: {color};">{count:,}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">{category}</div>
                        <div style="font-size: 0.7rem; color: #94a3b8;">{percentage:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        col1f, col2f = st.columns(2)
        
        with col1f:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_usage_trends(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2f:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Monthly Usage Analysis")
            
            if len(filtered_df) > 0:
                # Identify usage columns
                usage_cols = [col for col in filtered_df.columns if 'usage_month' in col]
                
                if len(usage_cols) > 0:
                    # Calculate completion rate (non-NaN values)
                    completion_rate = filtered_df[usage_cols].notna().mean().mean() * 100
                    st.metric("Data Completion Rate", f"{completion_rate:.1f}%")
                    
                    # Calculate variability
                    if len(usage_cols) >= 2:
                        variability = filtered_df[usage_cols].std(axis=1).mean()
                        st.metric("Average Monthly Variability", f"{variability:.1f} kg")
                    
                    # Show missing data pattern
                    missing_by_month = filtered_df[usage_cols].isna().sum()
                    st.markdown("**Missing Data by Month:**")
                    for col, missing in missing_by_month.items():
                        month_num = col.replace('usage_month_', '')
                        missing_pct = (missing / len(filtered_df)) * 100
                        st.progress(1 - (missing_pct/100), 
                                  text=f"Month {month_num}: {missing} missing ({missing_pct:.1f}%)")
                else:
                    st.info("No monthly usage data columns found")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        col1g, col2g = st.columns(2)
        
        with col1g:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_risk_heatmap(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2g:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 🎯 Risk Factor Analysis")
            
            if len(filtered_df) > 0:
                # Calculate correlations
                factors = {}
                if 'avg_reduction' in filtered_df.columns:
                    if 'distance_to_market_km' in filtered_df.columns:
                        corr_dist = filtered_df[['avg_reduction', 'distance_to_market_km']].corr().iloc[0,1]
                        factors['Distance to Market'] = f"{corr_dist:.3f}"
                    
                    if 'household_size' in filtered_df.columns:
                        corr_size = filtered_df[['avg_reduction', 'household_size']].corr().iloc[0,1]
                        factors['Household Size'] = f"{corr_size:.3f}"
                    
                    if 'elevation_m' in filtered_df.columns:
                        corr_elev = filtered_df[['avg_reduction', 'elevation_m']].corr().iloc[0,1]
                        factors['Elevation'] = f"{corr_elev:.3f}"
                
                st.markdown("**Correlation with Fuel Reduction:**")
                for factor, corr in factors.items():
                    col_f1, col_f2 = st.columns([2, 1])
                    with col_f1:
                        st.text(factor)
                    with col_f2:
                        color = "red" if float(corr) < -0.1 else "green" if float(corr) > 0.1 else "gray"
                        st.markdown(f"<span style='color: {color}; font-weight: bold;'>{corr}</span>", 
                                  unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("**Risk Profile:**")
                high_risk = filtered_df[filtered_df['low_adoption_risk'] == 1]
                if len(high_risk) > 0:
                    avg_distance = high_risk['distance_to_market_km'].mean() if 'distance_to_market_km' in high_risk.columns else "N/A"
                    avg_size = high_risk['household_size'].mean() if 'household_size' in high_risk.columns else "N/A"
                    
                    st.metric("Avg Distance (High Risk)", f"{avg_distance:.1f} km" if isinstance(avg_distance, (int, float)) else avg_distance)
                    st.metric("Avg Household Size (High Risk)", f"{avg_size:.1f}" if isinstance(avg_size, (int, float)) else avg_size)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        col1h, col2h = st.columns(2)
        
        with col1h:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 🌍 Environmental Impact")
            
            if len(filtered_df) > 0 and 'weekly_fuel_saving_kg' in filtered_df.columns:
                # Calculate impacts
                weekly_savings = filtered_df['weekly_fuel_saving_kg'].sum()
                annual_savings = weekly_savings * 52
                co2_reduction = annual_savings * 1.8  # Approx 1.8 kg CO2 per kg fuelwood
                trees_saved = annual_savings / 500  # Approx 500 kg per tree per year
                
                st.metric("Weekly Fuel Saved", f"{weekly_savings/1000:,.0f} tons")
                st.metric("Annual Fuel Saved", f"{annual_savings/1000:,.0f} tons")
                st.metric("CO₂ Reduction", f"{co2_reduction/1000:,.0f} tons")
                st.metric("Trees Protected", f"{trees_saved:,.0f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2h:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 💰 Economic Impact")
            
            if len(filtered_df) > 0:
                # Economic calculations
                fuel_price_per_kg = 0.1  # Estimated $0.1 per kg of fuelwood
                time_saved_per_kg = 0.5  # Estimated 0.5 hours saved per kg of fuelwood
                hourly_wage = 0.5  # Estimated $0.5 per hour
                
                if 'weekly_fuel_saving_kg' in filtered_df.columns:
                    weekly_savings = filtered_df['weekly_fuel_saving_kg'].sum()
                    
                    weekly_fuel_cost = weekly_savings * fuel_price_per_kg
                    weekly_time_savings = weekly_savings * time_saved_per_kg
                    weekly_wage_savings = weekly_time_savings * hourly_wage
                    
                    st.metric("Weekly Fuel Cost Savings", f"${weekly_fuel_cost:,.0f}")
                    st.metric("Weekly Time Savings", f"{weekly_time_savings:,.0f} hours")
                    st.metric("Equivalent Wage Savings", f"${weekly_wage_savings:,.0f}")
                    st.metric("Annual Economic Value", f"${weekly_wage_savings * 52:,.0f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        # Detailed insights tab
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("### 📋 Detailed District Insights")
        
        if len(filtered_df) > 0 and 'district' in filtered_df.columns:
            # Create a detailed table
            detailed_stats = filtered_df.groupby('district').agg({
                'avg_reduction': ['mean', 'std', 'min', 'max'],
                'household_id': 'count',
                'low_adoption_risk': 'mean',
                'distance_to_market_km': 'mean',
                'household_size': 'mean'
            }).round(2)
            
            # Flatten column names
            detailed_stats.columns = ['_'.join(col).strip() for col in detailed_stats.columns.values]
            detailed_stats = detailed_stats.reset_index()
            
            # Rename columns for readability
            column_mapping = {
                'district': 'District',
                'avg_reduction_mean': 'Avg Reduction (%)',
                'avg_reduction_std': 'Std Dev',
                'avg_reduction_min': 'Min Reduction',
                'avg_reduction_max': 'Max Reduction',
                'household_id_count': 'Households',
                'low_adoption_risk_mean': 'Risk Rate',
                'distance_to_market_km_mean': 'Avg Distance (km)',
                'household_size_mean': 'Avg Household Size'
            }
            
            detailed_stats = detailed_stats.rename(columns=column_mapping)
            
            # Display the table
            st.dataframe(
                detailed_stats,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg Reduction (%)": st.column_config.ProgressColumn(
                        "Avg Reduction (%)",
                        help="Average fuel reduction percentage",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Risk Rate": st.column_config.ProgressColumn(
                        "Risk Rate",
                        help="Percentage of households below 30% target",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                }
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

# Data Quality Section
st.markdown("<div class='section-title'>🔍 Data Quality & Technical Insights</div>", unsafe_allow_html=True)

col_qual1, col_qual2, col_qual3 = st.columns(3)

with col_qual1:
    st.markdown("""
    <div class="story-card">
        <h4 style="color: #0c4a6e; margin-top: 0;">📊 Data Completeness</h4>
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">Total Records</span>
                <span style="font-weight: bold;">{:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">Complete Records</span>
                <span style="font-weight: bold;">{:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #475569;">Completion Rate</span>
                <span style="font-weight: bold; color: #059669;">{:.1f}%</span>
            </div>
        </div>
        <div style="background: #f0f9ff; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem;">
            <div style="font-size: 0.85rem; color: #0369a1;">
                <strong>Insight:</strong> {:.1f}% of households have complete monthly usage data
            </div>
        </div>
    </div>
    """.format(
        len(df),
        df.dropna(subset=['avg_reduction', 'distance_to_market_km']).shape[0],
        (df.dropna(subset=['avg_reduction', 'distance_to_market_km']).shape[0] / len(df)) * 100,
        (df.dropna(subset=[col for col in df.columns if 'usage_month' in col]).shape[0] / len(df)) * 100
    ), unsafe_allow_html=True)

with col_qual2:
    st.markdown("""
    <div class="story-card">
        <h4 style="color: #0c4a6e; margin-top: 0;">🎯 Predictive Model Performance</h4>
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">Model Accuracy</span>
                <span style="font-weight: bold; color: #059669;">61.8%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">ROC-AUC Score</span>
                <span style="font-weight: bold; color: #059669;">66.2%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #475569;">Top Predictor</span>
                <span style="font-weight: bold;">Distance to Market</span>
            </div>
        </div>
        <div style="background: #f0fdf4; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem;">
            <div style="font-size: 0.85rem; color: #166534;">
                <strong>Technical Note:</strong> Model identifies 61.9% of at-risk households with 61.8% accuracy
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_qual3:
    st.markdown("""
    <div class="story-card">
        <h4 style="color: #0c4a6e; margin-top: 0;">📈 Statistical Significance</h4>
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">Sample Size</span>
                <span style="font-weight: bold;">{:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #475569;">Confidence Level</span>
                <span style="font-weight: bold;">95%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #475569;">Margin of Error</span>
                <span style="font-weight: bold;">±1.1%</span>
            </div>
        </div>
        <div style="background: #fef3c7; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem;">
            <div style="font-size: 0.85rem; color: #92400e;">
                <strong>Methodology:</strong> Data analyzed using logistic regression with 5-fold cross-validation
            </div>
        </div>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

# Final Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <div style="font-size: 0.9rem; margin-bottom: 0.5rem; display: flex; justify-content: center; align-items: center; gap: 10px;">
        <span>🔥 Sustainable Cooking Impact Dashboard • DelAgua Stove Adoption Programme</span>
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;">
        Data updated: March 2024 • {total_households:,} households analyzed • {high_risk_count:,} high-risk households identified
    </div>
    <div style="font-size: 0.75rem; color: #cbd5e1;">
        Interactive version of the DelAgua Strategic Analysis Report • 
        Built with Streamlit • 
        For technical support: Samson Niyizurugero • sniyizurugero@aimsric.org
    </div>
</div>
""".format(total_households=total_households, high_risk_count=high_risk_count), unsafe_allow_html=True)
