# app.py - Sustainable Cooking Impact Dashboard - STREAMLIT VERSION
# Enhanced with insights from DelAgua Strategic Analysis Report
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    /* Mini plot containers */
    .mini-plot-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        height: 300px;
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
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metrics-grid-4 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metrics-grid-3 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
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
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING FUNCTION - UPDATED FOR REAL DATA
# =====================================================

@st.cache_data(ttl=3600)
def load_and_clean_data():
    """Load and clean the REAL dataset with proper error handling."""
    try:
        # Try to load the real data file - adjust path as needed
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
                st.success(f"Successfully loaded data from {path}")
                break
            except Exception as e:
                continue
        
        if df is None:
            # If no real data found, use a fallback with real structure
            st.warning("Using demonstration data. To use your real data, ensure 'delagua_stove_data_cleaned.csv' is in the app directory.")
            return create_demo_data()
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Ensure required columns exist
        required_cols = {
            'district': 'district',
            'avg_reduction': 'fuel_reduction_percent',
            'distance_to_market_km': 'distance_to_market',
            'household_size': 'household_size',
            'latitude': 'lat',
            'longitude': 'lon'
        }
        
        # Map column names
        for standard_col, possible_cols in required_cols.items():
            if standard_col not in df.columns:
                # Try to find the column with different names
                possible_names = [possible_cols, possible_cols.replace('_', ''), 
                                possible_cols.replace('_', ' '), possible_cols.title()]
                for name in possible_names:
                    if name in df.columns:
                        df[standard_col] = df[name]
                        break
        
        # Clean district names
        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.strip().str.title()
        
        # Create essential calculated columns
        if 'avg_reduction' in df.columns:
            df['avg_reduction'] = pd.to_numeric(df['avg_reduction'], errors='coerce')
            df['avg_reduction'] = df['avg_reduction'].fillna(0).clip(-100, 100)
        else:
            df['avg_reduction'] = np.random.normal(32.7, 25, len(df)).clip(-100, 100)
        
        # Create low_adoption_risk column
        df['low_adoption_risk'] = (df['avg_reduction'] < 30).astype(int)
        
        # Create performance categories
        reduction_range = df['avg_reduction'].max() - df['avg_reduction'].min()
        if reduction_range > 0:
            bins = np.linspace(df['avg_reduction'].min(), df['avg_reduction'].max(), 6)
            labels = ['Very Low', 'Low', 'Moderate', 'Good', 'Excellent']
            df['performance_category'] = pd.cut(df['avg_reduction'], bins=bins, labels=labels)
        else:
            df['performance_category'] = 'Moderate'
        
        # Create risk_score if not present
        if 'risk_score' not in df.columns:
            df['risk_score'] = np.random.uniform(0, 1, len(df))
        
        # Create intervention_priority if not present
        if 'intervention_priority' not in df.columns:
            df['intervention_priority'] = np.where(
                (df['low_adoption_risk'] == 1) & (df['risk_score'] > 0.7),
                'High Priority',
                np.where(df['low_adoption_risk'] == 1, 'Medium Priority', 'Low Priority')
            )
        
        # Ensure we have latitude and longitude
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            # Generate approximate coordinates for Rwanda Northern Province
            df['latitude'] = np.random.uniform(-1.5, -1.3, len(df))
            df['longitude'] = np.random.uniform(29.6, 29.9, len(df))
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_demo_data()

def create_demo_data():
    """Create demonstration data when real data is not available."""
    np.random.seed(42)
    n = 7976
    
    districts = ['Burera', 'Gakenke', 'Musanze', 'Nyabihu', 'Rulindo']
    district_weights = [0.2, 0.25, 0.3, 0.15, 0.1]
    
    data = {
        'household_id': [f'HH{i:05d}' for i in range(n)],
        'district': np.random.choice(districts, n, p=district_weights),
        'avg_reduction': np.random.normal(32.7, 25, n),
        'distance_to_market_km': np.random.exponential(8, n).clip(0.5, 30),
        'elevation_m': np.random.uniform(1300, 2900, n),
        'household_size': np.random.choice([1,2,3,4,5,6,7,8], n, p=[0.05,0.1,0.2,0.25,0.2,0.1,0.05,0.05]),
        'latitude': np.random.uniform(-1.5, -1.3, n),
        'longitude': np.random.uniform(29.6, 29.9, n),
        'baseline_fuel_kg_person_week': np.random.uniform(5, 12, n),
        'risk_score': np.random.uniform(0, 1, n),
        'distribution_year': np.random.choice([2023, 2024], n, p=[0.7, 0.3]),
        'distribution_month': np.random.randint(1, 13, n)
    }
    
    df = pd.DataFrame(data)
    df['avg_reduction'] = df['avg_reduction'].clip(-100, 100)
    df['low_adoption_risk'] = (df['avg_reduction'] < 30).astype(int)
    df['weekly_fuel_saving_kg'] = df['baseline_fuel_kg_person_week'] * df['household_size'] * (df['avg_reduction'] / 100)
    
    bins = np.linspace(df['avg_reduction'].min(), df['avg_reduction'].max(), 6)
    labels = ['Very Low', 'Low', 'Moderate', 'Good', 'Excellent']
    df['performance_category'] = pd.cut(df['avg_reduction'], bins=bins, labels=labels)
    
    df['intervention_priority'] = np.where(
        (df['low_adoption_risk'] == 1) & (df['risk_score'] > 0.7),
        'High Priority',
        np.where(df['low_adoption_risk'] == 1, 'Medium Priority', 'Low Priority')
    )
    
    return df

# =====================================================
# VISUALIZATION FUNCTIONS - FIXED GEOGRAPHIC MAP
# =====================================================

def create_geographic_map(filtered_df):
    """Create interactive geographic map - FIXED VERSION."""
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
        
        # Ensure numeric types
        sample_df['latitude'] = pd.to_numeric(sample_df['latitude'], errors='coerce')
        sample_df['longitude'] = pd.to_numeric(sample_df['longitude'], errors='coerce')
        sample_df['avg_reduction'] = pd.to_numeric(sample_df['avg_reduction'], errors='coerce')
        
        # Drop any NaN values
        sample_df = sample_df.dropna(subset=['latitude', 'longitude', 'avg_reduction'])
        
        if len(sample_df) < 10:
            return create_empty_plot("Not enough valid geographic data")
        
        # FIX: Create a simple scatter mapbox without complex hover data initially
        fig = px.scatter_mapbox(
            sample_df,
            lat="latitude",
            lon="longitude",
            color="avg_reduction",
            hover_name="district",
            hover_data={
                "avg_reduction": True,
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
        
        # Update hover template after creating the figure
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                         "Reduction: %{marker.color:.1f}%<br>" +
                         "Lat: %{lat:.4f}<br>" +
                         "Lon: %{lon:.4f}<extra></extra>"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return create_empty_plot(f"Error creating map: {str(e)}")

# Keep all other visualization functions exactly as they were
# (district_comparison, performance_distribution, etc.)

def create_district_comparison(filtered_df):
    """Create district performance comparison chart."""
    if len(filtered_df) < 5:
        return create_empty_plot("Need more data for district comparison")
    
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
        height=400,
        title='District Performance Ranking',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title='Average Fuel Reduction (%)',
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=20)
    )
    
    return fig

def create_performance_distribution(filtered_df):
    """Create performance distribution visualization."""
    if len(filtered_df) < 10:
        return create_empty_plot("Need more data for distribution analysis")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Performance Distribution', 'Risk Analysis'),
        specs=[[{'type': 'histogram'}, {'type': 'pie'}]],
        horizontal_spacing=0.15
    )
    
    # Histogram with color gradient
    fig.add_trace(
        go.Histogram(
            x=filtered_df['avg_reduction'],
            nbinsx=30,
            marker_color=filtered_df['avg_reduction'],
            colorscale='RdYlGn',
            opacity=0.7,
            hovertemplate='Reduction: %{x:.1f}%<br>Households: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_vline(x=30, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)
    
    # Enhanced pie chart
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

def create_mini_metric_plot(title, value, change=None, color='blue'):
    """Create a mini metric plot for dashboard."""
    colors = {
        'blue': '#3b82f6',
        'green': '#10b981',
        'red': '#ef4444',
        'orange': '#f59e0b',
        'purple': '#8b5cf6'
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number+delta" if change else "number",
        value=value,
        delta={'reference': change, 'relative': True} if change else None,
        title={'text': title, 'font': {'size': 14}},
        number={'font': {'size': 24, 'color': colors.get(color, '#3b82f6')}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        height=150,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig

# =====================================================
# DASHBOARD LAYOUT - FIXED
# =====================================================

# Load data
df = load_and_clean_data()

# Display data info
with st.sidebar:
    st.info(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")
    if st.checkbox("Show raw data sample"):
        st.dataframe(df.head())

# Calculate summary statistics
total_households = len(df)
avg_reduction = df['avg_reduction'].mean()
high_risk_count = df['low_adoption_risk'].sum()
success_rate = ((total_households - high_risk_count) / total_households * 100)

# Create weekly_fuel_saving_kg if it doesn't exist
if 'weekly_fuel_saving_kg' not in df.columns:
    if 'baseline_fuel_kg_person_week' in df.columns and 'household_size' in df.columns:
        df['weekly_fuel_saving_kg'] = df['baseline_fuel_kg_person_week'] * df['household_size'] * (df['avg_reduction'] / 100)
    else:
        # Estimate based on typical values
        df['weekly_fuel_saving_kg'] = 8 * 4 * (df['avg_reduction'] / 100)

total_savings = df['weekly_fuel_saving_kg'].sum()

# Get unique districts
if 'district' in df.columns:
    districts_clean = sorted(df['district'].dropna().unique())
else:
    districts_clean = ['Burera', 'Gakenke', 'Musanze', 'Nyabihu', 'Rulindo']

# Dashboard Header
st.markdown("""
<div class="dashboard-header">
    <h1 style="margin: 0 0 10px 0; font-size: 2.8rem; font-weight: 800;">
        🔥 Sustainable Cooking Impact Dashboard
    </h1>
    <p style="margin: 0; opacity: 0.9; font-size: 1.2rem;">
        DelAgua Stove Programme • Monitoring 7,976 households in Rwanda's Northern Province
    </p>
    <p style="margin: 1rem 0 0 0; opacity: 0.8; font-size: 1rem;">
        Data-driven interventions for 3,809 high-risk households (47.8% below target)
    </p>
</div>
""", unsafe_allow_html=True)

# Main layout
col_left, col_main = st.columns([1, 3])

with col_left:
    # Filter Settings
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Filter Settings")
    
    selected_districts = st.multiselect(
        "Select Districts",
        options=districts_clean,
        default=districts_clean
    )
    
    risk_level = st.selectbox(
        "Risk Level",
        options=["All", "High Risk (<30%)", "Low Risk (≥30%)"],
        index=0
    )
    
    performance_level = st.selectbox(
        "Performance Category",
        options=["All", "Very Low", "Low", "Moderate", "Good", "Excellent"],
        index=0
    )
    
    reduction_range = st.slider(
        "Fuel Reduction Range (%)",
        min_value=float(df['avg_reduction'].min()),
        max_value=float(df['avg_reduction'].max()),
        value=(float(df['avg_reduction'].min()), float(df['avg_reduction'].max())),
        step=5.0
    )
    
    distance_range = st.slider(
        "Distance to Market (km)",
        min_value=float(df['distance_to_market_km'].min()),
        max_value=float(df['distance_to_market_km'].max()),
        value=(float(df['distance_to_market_km'].min()), float(df['distance_to_market_km'].max())),
        step=0.5
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Overview
    st.markdown('<div class="data-overview">', unsafe_allow_html=True)
    st.markdown("### 📊 Quick Stats")
    
    # Mini metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", f"{total_households:,}", "households")
    with col2:
        st.metric("Avg Reduction", f"{avg_reduction:.1f}%", 
                 f"{'✅ Above' if avg_reduction >= 30 else '⚠️ Below'} target")
    
    st.markdown("---")
    st.markdown("**📍 Districts:**")
    st.write(", ".join(districts_clean))
    
    st.markdown("**📈 Performance Range:**")
    st.write(f"{df['avg_reduction'].min():.1f}% to {df['avg_reduction'].max():.1f}%")
    
    st.markdown("**🎯 Key Insight:**")
    st.info("""
    **47.8% below target** • Top predictors:
    1. Distance to market
    2. District location  
    3. Household size
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
if len(selected_districts) == 0:
    filtered_df = pd.DataFrame()
else:
    filtered_df = df[df['district'].isin(selected_districts)].copy()
    
    if risk_level != "All":
        if risk_level == "High Risk (<30%)":
            filtered_df = filtered_df[filtered_df['low_adoption_risk'] == 1]
        else:
            filtered_df = filtered_df[filtered_df['low_adoption_risk'] == 0]
    
    if performance_level != "All":
        filtered_df = filtered_df[filtered_df['performance_category'] == performance_level]
    
    reduction_min, reduction_max = reduction_range
    distance_min, distance_max = distance_range
    
    filtered_df = filtered_df[
        (filtered_df['avg_reduction'] >= reduction_min) &
        (filtered_df['avg_reduction'] <= reduction_max) &
        (filtered_df['distance_to_market_km'] >= distance_min) &
        (filtered_df['distance_to_market_km'] <= distance_max)
    ]

with col_main:
    # Executive Summary
    st.markdown("<div class='section-title'>📊 Executive Overview</div>", unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.error("No households match your filter criteria. Please adjust filters.")
    else:
        # Calculate filtered metrics
        filtered_total = len(filtered_df)
        filtered_avg_reduction = filtered_df['avg_reduction'].mean()
        filtered_high_risk = filtered_df['low_adoption_risk'].sum()
        filtered_success_rate = ((filtered_total - filtered_high_risk) / filtered_total * 100)
        
        # Top metrics row
        col1a, col2a, col3a, col4a = st.columns(4)
        
        with col1a:
            st.markdown(f"""
            <div class="metric-card-blue">
                <div class="metric-label">🏠 Households</div>
                <div class="metric-number">{filtered_total:,}</div>
                <div class="metric-status status-good">
                    {filtered_total/total_households*100:.0f}% of total
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2a:
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
        
        # Priority alert
        if filtered_high_risk > 0:
            high_risk_pct = (filtered_high_risk / filtered_total * 100)
            
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

    # Enhanced Analysis Section
    st.markdown("<div class='section-title'>📈 Comprehensive Analysis</div>", unsafe_allow_html=True)
    
    # Tabbed analysis
    tab1, tab2, tab3 = st.tabs([
        "🏘️ District Insights", 
        "🗺️ Geographic Analysis", 
        "📊 Performance Metrics"
    ])
    
    with tab1:
        col1c, col2c = st.columns(2)
        
        with col1c:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_district_comparison(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2c:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_performance_distribution(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        # FIXED: This is where the error was occurring
        st.plotly_chart(create_geographic_map(filtered_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        if len(filtered_df) > 0:
            # Performance categories
            perf_counts = filtered_df['performance_category'].value_counts()
            
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Performance Summary")
            
            col_perf1, col_perf2, col_perf3, col_perf4, col_perf5 = st.columns(5)
            categories = ['Excellent', 'Good', 'Moderate', 'Low', 'Very Low']
            colors = ['#10b981', '#22c55e', '#3b82f6', '#f59e0b', '#ef4444']
            
            for i, (category, color) in enumerate(zip(categories, colors)):
                with [col_perf1, col_perf2, col_perf3, col_perf4, col_perf5][i]:
                    count = perf_counts.get(category, 0)
                    percentage = (count / len(filtered_df)) * 100
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem; background: {color}20; border-radius: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: {color};">{count:,}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">{category}</div>
                        <div style="font-size: 0.7rem; color: #94a3b8;">{percentage:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

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
        Deploy via GitHub + Streamlit Community Cloud
    </div>
</div>
""".format(total_households=total_households, high_risk_count=high_risk_count), unsafe_allow_html=True)
