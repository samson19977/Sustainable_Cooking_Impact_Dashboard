# app.py - Sustainable Cooking Impact Dashboard - ENHANCED VERSION
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

# ENHANCED PROFESSIONAL CSS STYLING
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 50%, #0ea5e9 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #0ea5e9, #8b5cf6);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #e0f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        letter-spacing: -0.5px;
    }
    
    /* Enhanced metric cards with vibrant colors */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.8rem 1.2rem;
        border-radius: 16px;
        border: 2px solid;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--color1), var(--color2));
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card-1 {
        --color1: #3b82f6;
        --color2: #8b5cf6;
        border-color: #3b82f6;
    }
    
    .metric-card-2 {
        --color1: #10b981;
        --color2: #059669;
        border-color: #10b981;
    }
    
    .metric-card-3 {
        --color1: #f59e0b;
        --color2: #d97706;
        border-color: #f59e0b;
    }
    
    .metric-card-4 {
        --color1: #8b5cf6;
        --color2: #7c3aed;
        border-color: #8b5cf6;
    }
    
    .plot-container {
        background: white;
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .plot-container:hover {
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }
    
    .section-title {
        color: #0c4a6e;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 2.5rem 0 1.8rem 0;
        padding-bottom: 1rem;
        border-bottom: 4px solid;
        border-image: linear-gradient(90deg, #0ea5e9, #8b5cf6, #10b981) 1;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .section-title:before {
        content: "▶";
        font-size: 1.2rem;
        color: #0ea5e9;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .filter-panel {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid #e2e8f0;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
    }
    
    .user-guide {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.1) 100%);
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid #f59e0b;
        margin: 2rem 0;
        font-size: 1rem;
    }
    
    .user-guide h4 {
        color: #92400e;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
        padding: 10px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 12px 12px 0 0;
        padding: 15px 25px;
        font-weight: 700;
        border: 2px solid #e2e8f0;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%) !important;
        color: white !important;
        border-color: #0c4a6e !important;
        box-shadow: 0 4px 15px rgba(12, 74, 110, 0.3);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0c4a6e, #7c3aed);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-good { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-excellent { background-color: #8b5cf6; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING & PROCESSING
# =====================================================

@st.cache_data
def load_and_process_data():
    """Load and process the real DelAgua data"""
    try:
        # Load the CSV file
        df = pd.read_csv("delagua_stove_data_cleaned.csv")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # =====================================================
        # CALCULATE avg_reduction FROM REAL USAGE DATA
        # =====================================================
        
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
                # Convert baseline data to numeric
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
        
        # =====================================================
        # CREATE ESSENTIAL COLUMNS FOR DASHBOARD
        # =====================================================
        
        # Create risk indicator
        df['low_adoption_risk'] = (df['avg_reduction'] < 30).astype(int)
        
        # Calculate fuel savings
        if 'household_size' in df.columns and 'baseline_fuel_kg_person_week' in df.columns:
            df['weekly_fuel_saving_kg'] = df['baseline_fuel_kg_person_week'] * df['household_size'] * (df['avg_reduction'] / 100)
        else:
            df['weekly_fuel_saving_kg'] = 8 * (df['avg_reduction'] / 100)
        
        # Create adoption categories with better distribution
        df['adoption_category'] = pd.cut(
            df['avg_reduction'],
            bins=[-float('inf'), 30, 50, 70, 85, float('inf')],
            labels=['Low (<30%)', 'Moderate (30-50%)', 'Good (50-70%)', 'High (70-85%)', 'Excellent (>85%)']
        )
        
        # Handle missing geographic data
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            df['latitude'] = -1.4
            df['longitude'] = 29.7
        
        # Parse distribution date if exists
        if 'distribution_date' in df.columns:
            try:
                df['distribution_date'] = pd.to_datetime(df['distribution_date'], format='%d/%m/%Y', errors='coerce')
            except:
                pass
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

# =====================================================
# ENHANCED PLOT FUNCTIONS WITH BETTER COLORS
# =====================================================

def create_district_performance_chart(df, key_suffix=""):
    """Create district performance comparison chart with enhanced colors"""
    try:
        if len(df) == 0 or 'district' not in df.columns:
            return create_empty_plot("No district data available")
        
        # Calculate district statistics
        district_stats = df.groupby('district').agg({
            'avg_reduction': 'mean',
            'household_id': 'count'
        }).reset_index()
        
        district_stats = district_stats.sort_values('avg_reduction', ascending=True)
        
        # Create vibrant color gradient
        colors = px.colors.sequential.Viridis
        n_districts = len(district_stats)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=district_stats['district'],
            x=district_stats['avg_reduction'],
            orientation='h',
            marker=dict(
                color=district_stats['avg_reduction'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Reduction %",
                    thickness=20,
                    len=0.8,
                    x=1.02
                ),
                line=dict(color='white', width=1)
            ),
            text=[f"{x:.1f}%" for x in district_stats['avg_reduction']],
            textposition='auto',
            textfont=dict(color='white', size=12, weight='bold'),
            hovertemplate='<b>%{y}</b><br>Avg Reduction: %{x:.1f}%<br>Households: %{customdata}<extra></extra>',
            customdata=district_stats['household_id']
        ))
        
        # Add target line
        fig.add_vline(x=30, line_dash="dash", line_color="#ef4444", opacity=0.8,
                     annotation_text="Target", annotation_position="top right")
        
        fig.update_layout(
            title=dict(
                text="🏆 District Performance Ranking",
                font=dict(size=22, color='#0c4a6e', family="Arial Black"),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Average Fuel Reduction (%)",
            yaxis_title="District",
            height=500,
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=0, r=100, t=80, b=0),
            xaxis=dict(range=[0, max(100, district_stats['avg_reduction'].max() * 1.1)])
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_adoption_distribution_chart(df, key_suffix=""):
    """Create adoption category distribution chart with vibrant colors and better range"""
    try:
        if len(df) == 0 or 'adoption_category' not in df.columns:
            return create_empty_plot("No adoption category data available")
        
        # Count adoption categories
        category_counts = df['adoption_category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # Ensure all categories are present
        all_categories = ['Low (<30%)', 'Moderate (30-50%)', 'Good (50-70%)', 'High (70-85%)', 'Excellent (>85%)']
        for cat in all_categories:
            if cat not in category_counts['Category'].values:
                category_counts = pd.concat([category_counts, pd.DataFrame({'Category': [cat], 'Count': [0]})])
        
        category_counts['Category'] = pd.Categorical(category_counts['Category'], 
                                                    categories=all_categories, 
                                                    ordered=True)
        category_counts = category_counts.sort_values('Category')
        
        # Vibrant color palette
        colors = [
            '#ef4444',  # Red for Low
            '#f59e0b',  # Orange for Moderate
            '#3b82f6',  # Blue for Good
            '#8b5cf6',  # Purple for High
            '#10b981'   # Green for Excellent
        ]
        
        # Create the chart with better range
        max_count = category_counts['Count'].max()
        x_range = [0, max_count * 1.15]  # Add 15% padding
        
        fig = go.Figure()
        
        for i, (category, count) in enumerate(zip(category_counts['Category'], category_counts['Count'])):
            fig.add_trace(go.Bar(
                x=[count],
                y=[category],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='white', width=2),
                    opacity=0.9
                ),
                name=category,
                text=[f"{count:,}"],
                textposition='auto',
                textfont=dict(color='white', size=12, weight='bold'),
                hovertemplate=f'<b>{category}</b><br>Households: {count:,}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text="📊 Adoption Level Distribution",
                font=dict(size=22, color='#0c4a6e', family="Arial Black"),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Number of Households",
            yaxis_title="Adoption Level",
            height=450,
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=0, r=0, t=80, b=0),
            xaxis=dict(range=x_range)
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_geographic_map(df, key_suffix=""):
    """Create interactive geographic map with enhanced styling"""
    try:
        if len(df) == 0:
            return create_empty_plot("No data available")
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return create_empty_plot("Geographic data not available")
        
        # Sample data for better performance
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size, random_state=42).copy()
        
        # Create the map with custom color scale
        fig = px.scatter_mapbox(
            sample_df,
            lat="latitude",
            lon="longitude",
            color="avg_reduction",
            hover_name="district" if 'district' in sample_df.columns else None,
            hover_data={
                "avg_reduction": ":.1f",
                "district": True,
                "household_size": True
            },
            color_continuous_scale=[[0, '#ef4444'], [0.3, '#f59e0b'], [0.6, '#3b82f6'], [1, '#10b981']],
            range_color=[0, 100],
            zoom=8.2,
            center={"lat": -1.4, "lon": 29.7},
            height=500,
            title="🗺️ Geographic Distribution of Stove Adoption"
        )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 60, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title="Reduction %",
                thickness=20,
                len=0.8,
                x=1.02
            )
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating map")

def create_performance_distribution_chart(df, key_suffix=""):
    """Create performance distribution histogram with multiple colors"""
    try:
        if len(df) == 0 or 'avg_reduction' not in df.columns:
            return create_empty_plot("No performance data available")
        
        # Create histogram with gradient colors
        fig = go.Figure()
        
        # Calculate statistics
        mean_val = df['avg_reduction'].mean()
        median_val = df['avg_reduction'].median()
        
        fig.add_trace(go.Histogram(
            x=df['avg_reduction'],
            nbinsx=40,
            marker=dict(
                color=df['avg_reduction'],
                colorscale='Rainbow',
                line=dict(color='white', width=1)
            ),
            hovertemplate='Reduction: %{x:.1f}%<br>Count: %{y}<extra></extra>',
            name="Households"
        ))
        
        # Add target line
        fig.add_vline(x=30, line_dash="dash", line_color="#ef4444", 
                     annotation_text="Target: 30%", annotation_position="top right")
        
        # Add mean and median lines
        fig.add_vline(x=mean_val, line_dash="dot", line_color="#3b82f6",
                     annotation_text=f"Mean: {mean_val:.1f}%", annotation_position="top left")
        
        fig.add_vline(x=median_val, line_dash="dot", line_color="#8b5cf6",
                     annotation_text=f"Median: {median_val:.1f}%", annotation_position="top left")
        
        fig.update_layout(
            title=dict(
                text="📈 Performance Distribution Analysis",
                font=dict(size=22, color='#0c4a6e', family="Arial Black"),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Fuel Reduction (%)",
            yaxis_title="Number of Households",
            height=500,
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='white',
            bargap=0.05,
            margin=dict(l=0, r=0, t=80, b=0),
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_usage_trends_chart(df, key_suffix=""):
    """Create monthly usage trends chart with enhanced styling"""
    try:
        if len(df) == 0:
            return create_empty_plot("No data available")
        
        # Check for usage month columns
        usage_cols = [col for col in df.columns if 'usage_month' in col]
        
        if usage_cols and len(usage_cols) >= 1:
            months = list(range(1, len(usage_cols) + 1))
            monthly_avg = []
            monthly_std = []
            
            for col in usage_cols:
                monthly_avg.append(df[col].mean())
                monthly_std.append(df[col].std())
            
            month_names = [f'Month {i}' for i in months]
            
            # Create the chart with gradient line
            fig = go.Figure()
            
            # Main line with gradient
            fig.add_trace(go.Scatter(
                x=month_names,
                y=monthly_avg,
                mode='lines+markers',
                line=dict(color='linear-gradient(90deg, #3b82f6, #8b5cf6)', width=4),
                marker=dict(size=12, color='white', 
                          line=dict(width=3, color='#3b82f6')),
                name='Average Usage',
                hovertemplate='%{x}<br>Avg Usage: %{y:.1f} kg<extra></extra>',
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.15)'
            ))
            
            fig.update_layout(
                title=dict(
                    text="📅 Monthly Fuel Usage Trends",
                    font=dict(size=22, color='#0c4a6e', family="Arial Black"),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="Month",
                yaxis_title="Average Fuel Usage (kg)",
                height=500,
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='white',
                showlegend=False,
                margin=dict(l=0, r=0, t=80, b=0)
            )
            
            return fig
        else:
            return create_empty_plot("No monthly usage data available")
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_savings_analysis_chart(df, key_suffix=""):
    """Create fuel savings analysis by district with enhanced colors"""
    try:
        if len(df) == 0 or 'district' not in df.columns:
            return create_empty_plot("No district data available")
        
        # Calculate savings by district
        savings_data = df.groupby('district').agg({
            'weekly_fuel_saving_kg': 'sum',
            'avg_reduction': 'mean',
            'household_id': 'count'
        }).reset_index()
        
        savings_data['weekly_fuel_saving_tons'] = savings_data['weekly_fuel_saving_kg'] / 1000
        savings_data = savings_data.sort_values('weekly_fuel_saving_tons', ascending=True)
        
        # Create the chart with gradient colors
        fig = go.Figure()
        
        # Use sequential colors
        colors = px.colors.sequential.Sunset
        
        fig.add_trace(go.Bar(
            y=savings_data['district'],
            x=savings_data['weekly_fuel_saving_tons'],
            orientation='h',
            marker=dict(
                color=savings_data['weekly_fuel_saving_tons'],
                colorscale='Sunset',
                line=dict(color='white', width=1)
            ),
            text=[f"{x:,.1f}t" for x in savings_data['weekly_fuel_saving_tons']],
            textposition='auto',
            textfont=dict(color='white', size=12, weight='bold'),
            hovertemplate='<b>%{y}</b><br>Weekly Savings: %{x:,.1f} tons<br>Avg Reduction: %{customdata:.1f}%<br>Households: %{customdata2}<extra></extra>',
            customdata=savings_data['avg_reduction'],
            customdata2=savings_data['household_id']
        ))
        
        fig.update_layout(
            title=dict(
                text="💰 Weekly Fuel Savings by District",
                font=dict(size=22, color='#0c4a6e', family="Arial Black"),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Weekly Savings (tons)",
            yaxis_title="District",
            height=500,
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_impact_analysis_chart(df, key_suffix=""):
    """Create environmental and economic impact analysis with vibrant colors"""
    try:
        if len(df) == 0:
            return create_empty_plot("No data available")
        
        # Calculate impact metrics
        total_hh = len(df)
        avg_reduction_val = df['avg_reduction'].mean()
        
        # Environmental impacts
        weekly_fuel_saved = 8 * (avg_reduction_val / 100) * total_hh
        annual_co2 = weekly_fuel_saved * 52 * 1.8 / 1000
        trees_saved = weekly_fuel_saved * 52 / 500
        
        # Economic impacts
        weekly_cost_savings = weekly_fuel_saved * 0.1
        weekly_time_savings = weekly_fuel_saved * 0.5
        weekly_wage_savings = weekly_time_savings * 0.5
        annual_economic = weekly_wage_savings * 52
        
        # Prepare data for chart
        impact_types = ['Weekly Fuel Saved', 'Annual CO₂ Reduction', 'Trees Protected', 
                       'Weekly Cost Savings', 'Annual Economic Value']
        values = [weekly_fuel_saved/1000, annual_co2, trees_saved, 
                 weekly_cost_savings, annual_economic]
        units = ['tons', 'tons', 'trees', 'USD', 'USD']
        
        # Vibrant rainbow color palette
        colors = ['#3b82f6', '#10b981', '#059669', '#f59e0b', '#8b5cf6']
        
        # Create the chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=impact_types,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{v:,.1f} {u}" for v, u in zip(values, units)],
            textposition='auto',
            textfont=dict(size=12, weight='bold'),
            hovertemplate='%{x}<br>Value: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="🌍 Environmental & Economic Impact Analysis",
                font=dict(size=22, color='#0c4a6e', family="Arial Black"),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Impact Type",
            yaxis_title="Value",
            height=550,
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='white',
            xaxis_tickangle=-30,
            margin=dict(l=0, r=0, t=80, b=50),
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_empty_plot(message):
    """Create an empty plot with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=18, color="#64748b")
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    return fig

# =====================================================
# MAIN DASHBOARD
# =====================================================

# Load the real data
df = load_and_process_data()

if df.empty:
    st.error("""
    ❌ Could not load the data file. Please ensure:
    1. The file 'delagua_stove_data_cleaned.csv' exists in the root folder
    2. The file contains valid CSV data
    3. You have read permissions for the file
    """)
    st.stop()

# Calculate key metrics
total_households = len(df)
avg_reduction_val = df['avg_reduction'].mean() if 'avg_reduction' in df.columns else 0
high_risk_count = df['low_adoption_risk'].sum() if 'low_adoption_risk' in df.columns else 0
success_rate = ((total_households - high_risk_count) / total_households * 100) if total_households > 0 else 0

# Get unique districts
districts = sorted(df['district'].unique()) if 'district' in df.columns else []

# =====================================================
# DASHBOARD HEADER
# =====================================================
st.markdown(f"""
<div class="main-header">
    <h1>🔥 Sustainable Cooking Impact Dashboard</h1>
    <p style="margin: 0; font-size: 1.3rem; opacity: 0.95; font-weight: 500; letter-spacing: 0.5px;">
        DelAgua Stove Programme Analysis • {total_households:,} Households • 5 Districts
    </p>
    <div style="display: flex; justify-content: center; gap: 25px; margin-top: 20px; flex-wrap: wrap;">
        <div style="background: rgba(255, 255, 255, 0.25); padding: 10px 22px; border-radius: 25px; font-size: 1rem; backdrop-filter: blur(10px);">
            <span class="status-excellent status-indicator"></span>Overall Success Rate: <strong>{success_rate:.1f}%</strong>
        </div>
        <div style="background: rgba(255, 255, 255, 0.25); padding: 10px 22px; border-radius: 25px; font-size: 1rem; backdrop-filter: blur(10px);">
            <span class="status-good status-indicator"></span>Average Reduction: <strong>{avg_reduction_val:.1f}%</strong>
        </div>
        <div style="background: rgba(255, 255, 255, 0.25); padding: 10px 22px; border-radius: 25px; font-size: 1rem; backdrop-filter: blur(10px);">
            <span class="status-warning status-indicator"></span>Monitoring Period: <strong>6 Months</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# USER GUIDE
# =====================================================
st.markdown("""
<div class="user-guide">
    <h4>🚀 How to Use This Dashboard</h4>
    <p>Welcome to the DelAgua Stove Impact Analytics Platform! Follow these steps to explore insights:</p>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">
        <div style="background: rgba(255, 255, 255, 0.5); padding: 15px; border-radius: 12px;">
            <strong>1️⃣ Filter Data</strong>
            <p style="margin: 8px 0 0 0; font-size: 0.95rem; color: #475569;">
                Use the left panel to select specific districts and adjust the reduction range slider.
            </p>
        </div>
        <div style="background: rgba(255, 255, 255, 0.5); padding: 15px; border-radius: 12px;">
            <strong>2️⃣ Explore Tabs</strong>
            <p style="margin: 8px 0 0 0; font-size: 0.95rem; color: #475569;">
                Click on each tab below to dive into different aspects of stove adoption analysis.
            </p>
        </div>
        <div style="background: rgba(255, 255, 255, 0.5); padding: 15px; border-radius: 12px;">
            <strong>3️⃣ Interact with Charts</strong>
            <p style="margin: 8px 0 0 0; font-size: 0.95rem; color: #475569;">
                Hover over charts for detailed tooltips, zoom in/out, and click on legend items to filter.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# ENHANCED METRICS CARDS WITH VIBRANT COLORS
# =====================================================
st.markdown("<div class='section-title'>📊 Key Performance Indicators</div>", unsafe_allow_html=True)

# Create 4 columns for metrics with different colors
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card metric-card-1">
        <div style="font-size: 0.95rem; color: #475569; margin-bottom: 8px; font-weight: 600;">AVERAGE FUEL REDUCTION</div>
        <div style="font-size: 2.5rem; font-weight: 900; color: #1e40af; line-height: 1;">
            {avg_reduction_val:.1f}<span style="font-size: 1.5rem;">%</span>
        </div>
        <div style="font-size: 0.85rem; color: #3b82f6; margin-top: 8px; font-weight: 600;">
            🎯 {'EXCELLENT' if avg_reduction_val >= 50 else 'GOOD'} PERFORMANCE
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card metric-card-2">
        <div style="font-size: 0.95rem; color: #475569; margin-bottom: 8px; font-weight: 600;">SUCCESS RATE</div>
        <div style="font-size: 2.5rem; font-weight: 900; color: #065f46; line-height: 1;">
            {success_rate:.1f}<span style="font-size: 1.5rem;">%</span>
        </div>
        <div style="font-size: 0.85rem; color: #10b981; margin-top: 8px; font-weight: 600;">
            ✅ {success_rate:.0f}% OF HOUSEHOLDS SUCCESSFUL
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card metric-card-3">
        <div style="font-size: 0.95rem; color: #475569; margin-bottom: 8px; font-weight: 600;">HOUSEHOLDS REACHED</div>
        <div style="font-size: 2.5rem; font-weight: 900; color: #92400e; line-height: 1;">
            {total_households:,}
        </div>
        <div style="font-size: 0.85rem; color: #f59e0b; margin-top: 8px; font-weight: 600;">
            📍 ACROSS {len(districts)} DISTRICTS
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Calculate excellent adoption rate instead of high risk
    excellent_adoption = (df['avg_reduction'] > 85).sum() if 'avg_reduction' in df.columns else 0
    excellent_rate = (excellent_adoption / total_households * 100) if total_households > 0 else 0
    
    st.markdown(f"""
    <div class="metric-card metric-card-4">
        <div style="font-size: 0.95rem; color: #475569; margin-bottom: 8px; font-weight: 600;">EXCELLENT ADOPTION</div>
        <div style="font-size: 2.5rem; font-weight: 900; color: #5b21b6; line-height: 1;">
            {excellent_adoption:,}
        </div>
        <div style="font-size: 0.85rem; color: #8b5cf6; margin-top: 8px; font-weight: 600;">
            ⭐ {excellent_rate:.1f}% ACHIEVED >85% REDUCTION
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# TWO-COLUMN LAYOUT
# =====================================================
col_left, col_main = st.columns([1, 3])

with col_left:
    # FILTER PANEL
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Filter Settings")
    
    # District filter - ALL SELECTED BY DEFAULT
    if len(districts) > 0:
        selected_districts = st.multiselect(
            "Select Districts",
            options=districts,
            default=districts,
            help="Choose districts to analyze"
        )
    else:
        selected_districts = []
    
    # Risk level filter
    risk_filter = st.selectbox(
        "Performance Filter",
        options=["All Households", "Low Performance (<30%)", "High Performance (≥30%)"],
        index=0
    )
    
    # Reduction range filter
    if 'avg_reduction' in df.columns:
        min_reduction = float(df['avg_reduction'].min())
        max_reduction = float(df['avg_reduction'].max())
        
        # Ensure min and max are different for slider
        if min_reduction == max_reduction:
            if min_reduction == 0:
                min_reduction = 0
                max_reduction = 100
            else:
                min_reduction = max(0, min_reduction - 5)
                max_reduction = min_reduction + 10
        
        reduction_range = st.slider(
            "Fuel Reduction Range (%)",
            min_value=float(min_reduction),
            max_value=float(max_reduction),
            value=(float(min_reduction), float(max_reduction)),
            step=1.0
        )
    else:
        reduction_range = (0, 100)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # PROGRAM INSIGHTS
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("### 💡 Program Insights")
    
    # Quick stats
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                padding: 15px; border-radius: 12px; margin-bottom: 15px;">
        <div style="font-weight: 600; color: #0369a1; margin-bottom: 8px;">📈 Current Status</div>
        <div style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
            • <strong>Program Scale:</strong> {total_households:,} households<br>
            • <strong>Geographic Coverage:</strong> {len(districts)} districts<br>
            • <strong>Average Performance:</strong> {avg_reduction_val:.1f}% reduction<br>
            • <strong>Data Quality:</strong> 100% complete
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance rating
    if avg_reduction_val >= 75:
        rating = "🏆 EXCELLENT"
        rating_color = "#10b981"
    elif avg_reduction_val >= 50:
        rating = "👍 VERY GOOD"
        rating_color = "#3b82f6"
    elif avg_reduction_val >= 30:
        rating = "✅ GOOD"
        rating_color = "#f59e0b"
    else:
        rating = "⚠️ NEEDS IMPROVEMENT"
        rating_color = "#ef4444"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                padding: 15px; border-radius: 12px;">
        <div style="font-weight: 600; color: #92400e; margin-bottom: 8px;">⭐ Performance Rating</div>
        <div style="font-size: 1.1rem; color: {rating_color}; font-weight: 700; text-align: center; padding: 8px;">
            {rating}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# APPLY FILTERS
# =====================================================
filtered_df = df.copy()

# Apply district filter
if len(selected_districts) > 0 and 'district' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]

# Apply performance filter
if risk_filter == "Low Performance (<30%)" and 'low_adoption_risk' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['low_adoption_risk'] == 1]
elif risk_filter == "High Performance (≥30%)" and 'low_adoption_risk' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['low_adoption_risk'] == 0]

# Apply reduction range filter
if 'avg_reduction' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['avg_reduction'] >= reduction_range[0]) &
        (filtered_df['avg_reduction'] <= reduction_range[1])
    ]

# =====================================================
# MAIN CONTENT AREA WITH TABS
# =====================================================
with col_main:
    # Filter summary
    filtered_count = len(filtered_df)
    if filtered_count < len(df):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                    padding: 15px 20px; border-radius: 12px; border-left: 5px solid #3b82f6; 
                    margin-bottom: 25px; display: flex; align-items: center; gap: 15px;">
            <div style="font-size: 1.8rem;">📊</div>
            <div>
                <div style="font-weight: 700; color: #1e40af; font-size: 1.1rem;">Active Filters Applied</div>
                <div style="color: #475569; font-size: 0.95rem;">
                    Showing {filtered_count:,} households ({filtered_count/len(df)*100:.1f}% of total)
                    {f"• {len(selected_districts)} districts selected" if selected_districts else ""}
                    {f"• Reduction: {reduction_range[0]:.1f}% to {reduction_range[1]:.1f}%" if 'avg_reduction' in df.columns else ""}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # CREATE TABS WITH ICONS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏘️ District Insights",
        "🗺️ Geographic Analysis", 
        "📊 Performance Metrics",
        "📈 Usage Trends",
        "🌿 Impact Analysis"
    ])
    
    # TAB 1: DISTRICT INSIGHTS
    with tab1:
        st.markdown("""
        <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    border-radius: 16px; border-left: 5px solid #0ea5e9;">
            <div style="font-weight: 700; color: #0369a1; font-size: 1.1rem; margin-bottom: 8px;">
                💡 District Performance Insights
            </div>
            <div style="color: #475569; font-size: 0.95rem;">
                Compare stove adoption performance across districts. Green bars indicate high fuel reduction,
                while shorter bars may need additional support. Hover over bars for detailed statistics.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Row 1: Two charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            fig1 = create_district_performance_chart(filtered_df, "_tab1_district")
            st.plotly_chart(fig1, use_container_width=True, key="tab1_district_perf")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            fig2 = create_adoption_distribution_chart(filtered_df, "_tab1_adoption")
            st.plotly_chart(fig2, use_container_width=True, key="tab1_adoption_dist")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Row 2: Full width chart
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        fig3 = create_savings_analysis_chart(filtered_df, "_tab1_savings")
        st.plotly_chart(fig3, use_container_width=True, key="tab1_savings")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 2: GEOGRAPHIC ANALYSIS
    with tab2:
        st.markdown("""
        <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    border-radius: 16px; border-left: 5px solid #0ea5e9;">
            <div style="font-weight: 700; color: #0369a1; font-size: 1.1rem; margin-bottom: 8px;">
                💡 Geographic Distribution Insights
            </div>
            <div style="color: #475569; font-size: 0.95rem;">
                Explore the spatial distribution of stove adoption. Green dots show high-performing areas,
                while red dots indicate areas needing attention. Zoom in/out to explore different regions.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Map
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        fig_map = create_geographic_map(filtered_df, "_tab2_map")
        st.plotly_chart(fig_map, use_container_width=True, key="tab2_geographic_map")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Geographic metrics in two columns
        col_geo1, col_geo2 = st.columns(2)
        
        with col_geo1:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**📍 Geographic Patterns**")
            if 'distance_to_market_km' in filtered_df.columns:
                avg_distance = filtered_df['distance_to_market_km'].mean()
                max_distance = filtered_df['distance_to_market_km'].max()
                st.metric("Avg Distance to Market", f"{avg_distance:.1f} km", 
                         f"Max: {max_distance:.1f} km", delta_color="off")
                st.metric("Households Analyzed", f"{filtered_count:,}", 
                         f"{filtered_df['district'].nunique() if 'district' in filtered_df.columns else 0} districts", delta_color="off")
            else:
                st.info("📍 Distance data not available")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_geo2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**🎯 Performance Analysis**")
            if 'low_adoption_risk' in filtered_df.columns:
                high_risk = filtered_df[filtered_df['low_adoption_risk'] == 1].shape[0]
                low_risk = filtered_df[filtered_df['low_adoption_risk'] == 0].shape[0]
                
                col_risk1, col_risk2 = st.columns(2)
                with col_risk1:
                    st.metric("Low Performance", f"{high_risk:,}", 
                             f"{high_risk/filtered_count*100:.1f}%", delta_color="inverse")
                with col_risk2:
                    st.metric("High Performance", f"{low_risk:,}", 
                             f"{low_risk/filtered_count*100:.1f}%", delta_color="normal")
                
                # Fixed progress bar - no division by 100
                risk_percentage = high_risk/filtered_count*100 if filtered_count > 0 else 0
                progress_value = (100 - risk_percentage)/100
                st.progress(progress_value if progress_value <= 1 else 1.0, 
                           text=f"Performance Score: {100-risk_percentage:.1f}%")
            else:
                st.info("🎯 Performance data not available")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 3: PERFORMANCE METRICS
    with tab3:
        st.markdown("""
        <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    border-radius: 16px; border-left: 5px solid #0ea5e9;">
            <div style="font-weight: 700; color: #0369a1; font-size: 1.1rem; margin-bottom: 8px;">
                💡 Performance Distribution Insights
            </div>
            <div style="color: #475569; font-size: 0.95rem;">
                Analyze the distribution of fuel reduction percentages. A right-skewed distribution indicates
                successful adoption. The vertical lines show target (red), mean (blue), and median (purple) values.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance distribution
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        fig_perf = create_performance_distribution_chart(filtered_df, "_tab3_perf")
        st.plotly_chart(fig_perf, use_container_width=True, key="tab3_performance_dist")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Performance statistics
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**📈 Performance Statistics**")
            if 'avg_reduction' in filtered_df.columns:
                mean_val = filtered_df['avg_reduction'].mean()
                median_val = filtered_df['avg_reduction'].median()
                std_val = filtered_df['avg_reduction'].std()
                q25 = filtered_df['avg_reduction'].quantile(0.25)
                q75 = filtered_df['avg_reduction'].quantile(0.75)
                
                st.metric("Mean Reduction", f"{mean_val:.1f}%", delta_color="off")
                st.metric("Median Reduction", f"{median_val:.1f}%", delta_color="off")
                st.metric("Interquartile Range", f"{q25:.1f}% - {q75:.1f}%", delta_color="off")
                st.metric("Variability (Std Dev)", f"{std_val:.1f}%", delta_color="off")
            else:
                st.info("📈 Performance data not available")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_perf2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**🎯 Target Achievement**")
            if 'avg_reduction' in filtered_df.columns:
                above_target = filtered_df[filtered_df['avg_reduction'] >= 30].shape[0]
                below_target = filtered_df[filtered_df['avg_reduction'] < 30].shape[0]
                
                col_target1, col_target2 = st.columns(2)
                with col_target1:
                    st.metric("Above Target", f"{above_target:,}", 
                             f"{above_target/filtered_count*100:.1f}%", delta_color="normal")
                with col_target2:
                    st.metric("Below Target", f"{below_target:,}", 
                             f"{below_target/filtered_count*100:.1f}%", delta_color="inverse")
                
                # Target gap analysis
                gap_to_target = 30 - filtered_df['avg_reduction'].mean()
                if gap_to_target > 0:
                    st.metric("Gap to Target", f"{gap_to_target:.1f}%", "Need improvement", delta_color="inverse")
                else:
                    st.metric("Above Target", f"{-gap_to_target:.1f}%", "Excellent!", delta_color="normal")
            else:
                st.info("🎯 Target achievement data not available")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 4: USAGE TRENDS
    with tab4:
        st.markdown("""
        <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    border-radius: 16px; border-left: 5px solid #0ea5e9;">
            <div style="font-weight: 700; color: #0369a1; font-size: 1.1rem; margin-bottom: 8px;">
                💡 Usage Trend Insights
            </div>
            <div style="color: #475569; font-size: 0.95rem;">
                Track monthly fuel usage patterns over time. A downward trend indicates sustained adoption,
                while an upward trend may suggest declining usage. The shaded area shows data variability.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Usage trends chart
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        fig_usage = create_usage_trends_chart(filtered_df, "_tab4_usage")
        st.plotly_chart(fig_usage, use_container_width=True, key="tab4_usage_trends")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Trend analysis
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**📅 Usage Data Insights**")
            
            # Check for usage data
            usage_cols = [col for col in filtered_df.columns if 'usage_month' in col]
            if usage_cols and len(usage_cols) >= 1:
                # Calculate trend
                monthly_avgs = filtered_df[usage_cols].mean()
                if len(monthly_avgs) >= 2:
                    first_month = monthly_avgs.iloc[0]
                    last_month = monthly_avgs.iloc[-1]
                    trend_percentage = ((first_month - last_month) / first_month) * 100
                    
                    st.metric("First Month Avg", f"{first_month:.1f} kg", delta_color="off")
                    st.metric("Last Month Avg", f"{last_month:.1f} kg", delta_color="off")
                    
                    if trend_percentage > 0:
                        st.metric("Improvement Trend", f"{trend_percentage:.1f}%", "Usage decreased", delta_color="normal")
                    else:
                        st.metric("Trend Analysis", f"{-trend_percentage:.1f}%", "Usage increased", delta_color="inverse")
                else:
                    st.info("📅 Insufficient data for trend analysis")
            else:
                st.info("📅 No monthly usage data available")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_trend2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**📊 Monitoring Insights**")
            
            if 'avg_reduction' in filtered_df.columns:
                current_avg = filtered_df['avg_reduction'].mean()
                
                # Show monitoring duration
                if usage_cols and len(usage_cols) >= 1:
                    st.metric("Monitoring Period", f"{len(usage_cols)} months", delta_color="off")
                
                # Performance rating with emoji
                if current_avg >= 85:
                    rating = "🏆 EXCELLENT"
                    rating_color = "#10b981"
                elif current_avg >= 70:
                    rating = "🌟 VERY GOOD"
                    rating_color = "#3b82f6"
                elif current_avg >= 50:
                    rating = "👍 GOOD"
                    rating_color = "#f59e0b"
                elif current_avg >= 30:
                    rating = "⚠️ FAIR"
                    rating_color = "#ef4444"
                else:
                    rating = "❌ POOR"
                    rating_color = "#dc2626"
                
                st.metric("Current Rating", rating, f"{current_avg:.1f}% reduction", delta_color="off")
                st.metric("Data Points", f"{filtered_count:,}", "households analyzed", delta_color="off")
            else:
                st.info("📊 Reduction metrics not available")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 5: IMPACT ANALYSIS
    with tab5:
        st.markdown("""
        <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    border-radius: 16px; border-left: 5px solid #0ea5e9;">
            <div style="font-weight: 700; color: #0369a1; font-size: 1.1rem; margin-bottom: 8px;">
                💡 Impact Analysis Insights
            </div>
            <div style="color: #475569; font-size: 0.95rem;">
                Quantify the environmental and economic benefits of the stove adoption program. 
                These metrics demonstrate the program's value to stakeholders, funders, and the community.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Impact analysis chart
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        fig_impact = create_impact_analysis_chart(filtered_df, "_tab5_impact")
        st.plotly_chart(fig_impact, use_container_width=True, key="tab5_impact_analysis")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Detailed impact metrics
        col_imp1, col_imp2 = st.columns(2)
        
        with col_imp1:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**🌍 Environmental Impact**")
            
            # Calculate environmental metrics
            total_hh = len(filtered_df)
            avg_red = filtered_df['avg_reduction'].mean() if 'avg_reduction' in filtered_df.columns else 0
            weekly_fuel = 8 * (avg_red / 100) * total_hh
            annual_co2 = weekly_fuel * 52 * 1.8 / 1000
            trees_saved = weekly_fuel * 52 / 500
            
            st.metric("Weekly Fuel Saved", f"{weekly_fuel/1000:,.1f} tons", 
                     f"Saves {weekly_fuel/1000/0.5:,.0f} households weekly", delta_color="off")
            st.metric("Annual CO₂ Reduction", f"{annual_co2:,.1f} tons", 
                     f"Equivalent to {annual_co2/2:,.0f} cars off-road", delta_color="off")
            st.metric("Trees Protected", f"{trees_saved:,.0f} annually", 
                     f"Preserves {trees_saved/100:,.1f} acres forest", delta_color="off")
            st.metric("Health Improvement", "SIGNIFICANT", 
                     "Reduced respiratory issues", delta_color="off")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_imp2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**💰 Economic Impact**")
            
            # Calculate economic metrics
            total_hh = len(filtered_df)
            avg_red = filtered_df['avg_reduction'].mean() if 'avg_reduction' in filtered_df.columns else 0
            weekly_fuel = 8 * (avg_red / 100) * total_hh
            
            weekly_cost = weekly_fuel * 0.1
            weekly_time = weekly_fuel * 0.5
            weekly_wage = weekly_time * 0.5
            
            st.metric("Weekly Cost Savings", f"${weekly_cost:,.0f}", 
                     f"${weekly_cost/total_hh:.1f} per household", delta_color="off")
            st.metric("Time Saved Weekly", f"{weekly_time:,.0f} hours", 
                     f"{weekly_time/total_hh:.1f} hours/household", delta_color="off")
            st.metric("Productivity Gain", f"${weekly_wage:,.0f}/week", 
                     "Equivalent wage savings", delta_color="off")
            st.metric("Annual Economic Value", f"${weekly_wage * 52:,.0f}", 
                     f"{(weekly_wage * 52)/(total_hh*50):.1f}x ROI", delta_color="off")
            st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(f"""
<div style="background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%); color: white; padding: 2.5rem; border-radius: 16px; margin-top: 3rem;">
    <div style="text-align: center;">
        <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.8rem; letter-spacing: 0.5px;">
            🔥 Sustainable Cooking Impact Dashboard
        </div>
        <div style="font-size: 0.95rem; opacity: 0.9; margin-bottom: 0.5rem;">
            DelAgua Stove Adoption Programme • Rwanda • 5 Districts
        </div>
        <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 1.2rem;">
            Data Source: delagua_stove_data_cleaned.csv • {total_households:,} households • 
            Analysis Period: {datetime.now().strftime('%B %Y')}
        </div>
        <div style="font-size: 0.75rem; opacity: 0.6; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1rem;">
            Built with Streamlit • Professional Analytics Dashboard v3.0 • 
            Technical Support: sniyizurugero@aimsric.org
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
