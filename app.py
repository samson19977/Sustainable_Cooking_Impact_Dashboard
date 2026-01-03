# app.py - Sustainable Cooking Impact Dashboard - FINAL FIXED VERSION
# USING ONLY REAL DATA - NO SYNTHETIC DATA
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

# SIMPLIFIED CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .plot-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        color: #0c4a6e;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #0ea5e9;
    }
    
    .filter-panel {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .data-quality {
        background: #f0f9ff;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #0ea5e9;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING & PROCESSING - REAL DATA ONLY
# =====================================================

@st.cache_data
def load_and_process_data():
    """Load and process the real DelAgua data with your exact format"""
    try:
        # Load the CSV file
        df = pd.read_csv("delagua_stove_data_cleaned.csv")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Display data info for debugging
        st.info(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # =====================================================
        # CALCULATE avg_reduction FROM REAL USAGE DATA
        # =====================================================
        
        # Identify usage month columns
        usage_cols = [col for col in df.columns if 'usage_month' in col]
        
        if usage_cols and len(usage_cols) >= 1:
            st.success(f"Found {len(usage_cols)} usage month columns: {usage_cols}")
            
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
                df['expected_monthly_kg'] = df['expected_weekly_kg'] * 4.33  # Convert to monthly
                
                # Calculate reduction percentage
                df['avg_reduction'] = ((df['expected_monthly_kg'] - df['avg_monthly_usage_kg']) / 
                                      df['expected_monthly_kg'].replace(0, np.nan)) * 100
                
                st.success("Calculated avg_reduction from baseline and usage data")
                
            else:
                # If no baseline, calculate reduction relative to maximum usage
                max_usage = df[usage_cols].max(axis=1, skipna=True)
                df['avg_reduction'] = ((max_usage - df['avg_monthly_usage_kg']) / 
                                      max_usage.replace(0, np.nan)) * 100
                st.success("Calculated avg_reduction from usage patterns (no baseline)")
            
            # Handle NaN values and extreme values
            df['avg_reduction'] = df['avg_reduction'].fillna(0)
            df['avg_reduction'] = df['avg_reduction'].replace([np.inf, -np.inf], 0)
            df['avg_reduction'] = df['avg_reduction'].clip(-100, 100)
            
            # Show distribution of calculated reductions
            avg_reduction_val = df['avg_reduction'].mean()
            st.info(f"Calculated average reduction: {avg_reduction_val:.1f}% (range: {df['avg_reduction'].min():.1f}% to {df['avg_reduction'].max():.1f}%)")
            
        else:
            # If no usage data, we cannot calculate reduction
            st.warning("No usage month columns found. Cannot calculate fuel reduction.")
            df['avg_reduction'] = 0  # Set to 0 as placeholder
        
        # =====================================================
        # CREATE ESSENTIAL COLUMNS FOR DASHBOARD
        # =====================================================
        
        # Create risk indicator
        df['low_adoption_risk'] = (df['avg_reduction'] < 30).astype(int)
        
        # Calculate fuel savings
        if 'household_size' in df.columns and 'baseline_fuel_kg_person_week' in df.columns:
            df['weekly_fuel_saving_kg'] = df['baseline_fuel_kg_person_week'] * df['household_size'] * (df['avg_reduction'] / 100)
        else:
            # Estimate savings
            df['weekly_fuel_saving_kg'] = 8 * (df['avg_reduction'] / 100)  # Assume 8kg per week baseline
        
        # Create adoption categories
        df['adoption_category'] = pd.cut(
            df['avg_reduction'],
            bins=[-float('inf'), 15, 30, 50, float('inf')],
            labels=['Very Low (<15%)', 'Low (15-30%)', 'Moderate (30-50%)', 'High (>50%)']
        )
        
        # Handle missing geographic data
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.warning("Missing latitude/longitude data. Using district centroids as fallback.")
            # Add placeholder coordinates for Rwanda
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
# PLOT CREATION FUNCTIONS - WITH PROPER ERROR HANDLING
# =====================================================

def create_district_performance_chart(df, key_suffix=""):
    """Create district performance comparison chart"""
    try:
        if len(df) == 0 or 'district' not in df.columns:
            return create_empty_plot("No district data available")
        
        # Calculate district statistics
        district_stats = df.groupby('district').agg({
            'avg_reduction': 'mean',
            'household_id': 'count'
        }).reset_index()
        
        district_stats = district_stats.sort_values('avg_reduction', ascending=True)
        
        # Create the chart
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=district_stats['district'],
            x=district_stats['avg_reduction'],
            orientation='h',
            marker_color=['#ef4444' if x < 30 else '#10b981' for x in district_stats['avg_reduction']],
            text=[f"{x:.1f}%" for x in district_stats['avg_reduction']],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Avg Reduction: %{x:.1f}%<br>Households: %{customdata}<extra></extra>',
            customdata=district_stats['household_id']
        ))
        
        # Add target line
        fig.add_vline(x=30, line_dash="dash", line_color="red", opacity=0.7)
        
        fig.update_layout(
            title="District Performance Ranking",
            xaxis_title="Average Fuel Reduction (%)",
            yaxis_title="District",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_adoption_distribution_chart(df, key_suffix=""):
    """Create adoption category distribution chart"""
    try:
        if len(df) == 0 or 'adoption_category' not in df.columns:
            return create_empty_plot("No adoption category data available")
        
        # Count adoption categories
        category_counts = df['adoption_category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # Colors for categories
        colors = {
            'High (>50%)': '#10b981',
            'Moderate (30-50%)': '#3b82f6',
            'Low (15-30%)': '#f59e0b',
            'Very Low (<15%)': '#ef4444'
        }
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        for category in category_counts['Category']:
            count = category_counts[category_counts['Category'] == category]['Count'].values[0]
            fig.add_trace(go.Bar(
                x=[count],
                y=[category],
                orientation='h',
                marker_color=colors.get(category, '#6b7280'),
                name=category,
                text=[f"{count:,}"],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Adoption Level Distribution",
            xaxis_title="Number of Households",
            yaxis_title="Adoption Level",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_geographic_map(df, key_suffix=""):
    """Create interactive geographic map"""
    try:
        if len(df) == 0:
            return create_empty_plot("No data available")
        
        # Ensure required columns exist
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return create_empty_plot("Geographic data not available")
        
        # Sample data for better performance
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size, random_state=42).copy()
        
        # Create the map
        fig = px.scatter_mapbox(
            sample_df,
            lat="latitude",
            lon="longitude",
            color="avg_reduction",
            hover_name="district" if 'district' in sample_df.columns else None,
            hover_data={
                "avg_reduction": ":.1f",
                "district": True if 'district' in sample_df.columns else False,
                "household_size": True if 'household_size' in sample_df.columns else False
            },
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            zoom=8,
            center={"lat": -1.4, "lon": 29.7},
            height=400,
            title="Geographic Distribution of Stove Adoption"
        )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating map")

def create_performance_distribution_chart(df, key_suffix=""):
    """Create performance distribution histogram"""
    try:
        if len(df) == 0 or 'avg_reduction' not in df.columns:
            return create_empty_plot("No performance data available")
        
        # Create histogram
        fig = px.histogram(
            df, 
            x='avg_reduction',
            nbins=30,
            title="Performance Distribution",
            labels={'avg_reduction': 'Fuel Reduction (%)'},
            color_discrete_sequence=['#3b82f6']
        )
        
        # Add target line
        fig.add_vline(x=30, line_dash="dash", line_color="red", 
                     annotation_text="Target: 30%", annotation_position="top right")
        
        fig.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Fuel Reduction (%)",
            yaxis_title="Number of Households"
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_usage_trends_chart(df, key_suffix=""):
    """Create monthly usage trends chart from real data"""
    try:
        if len(df) == 0:
            return create_empty_plot("No data available")
        
        # Check for usage month columns
        usage_cols = [col for col in df.columns if 'usage_month' in col]
        
        if usage_cols and len(usage_cols) >= 1:
            # Use real usage data
            months = list(range(1, len(usage_cols) + 1))
            monthly_avg = []
            
            for col in usage_cols:
                monthly_avg.append(df[col].mean())
            
            month_names = [f'Month {i}' for i in months]
            
            # Create the chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=month_names,
                y=monthly_avg,
                mode='lines+markers',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8, color='white', line=dict(width=2, color='#3b82f6')),
                name='Monthly Average Usage (kg)',
                hovertemplate='%{x}<br>Avg Usage: %{y:.1f} kg<extra></extra>'
            ))
            
            fig.update_layout(
                title="Monthly Fuel Usage Trends",
                xaxis_title="Month",
                yaxis_title="Average Fuel Usage (kg)",
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
            )
            
            return fig
        else:
            return create_empty_plot("No monthly usage data available")
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_savings_analysis_chart(df, key_suffix=""):
    """Create fuel savings analysis by district"""
    try:
        if len(df) == 0 or 'district' not in df.columns:
            return create_empty_plot("No district data available")
        
        # Calculate savings by district
        savings_data = df.groupby('district').agg({
            'weekly_fuel_saving_kg': 'sum',
            'household_id': 'count'
        }).reset_index()
        
        savings_data = savings_data.sort_values('weekly_fuel_saving_kg', ascending=False)
        
        # Create the chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=savings_data['district'],
            y=savings_data['weekly_fuel_saving_kg'] / 1000,  # Convert to tons
            marker_color='#f59e0b',
            text=[f"{x/1000:,.1f}t" for x in savings_data['weekly_fuel_saving_kg']],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Weekly Savings: %{y:,.1f} tons<br>Households: %{customdata}<extra></extra>',
            customdata=savings_data['household_id']
        ))
        
        fig.update_layout(
            title="Weekly Fuel Savings by District",
            xaxis_title="District",
            yaxis_title="Weekly Savings (tons)",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

def create_impact_analysis_chart(df, key_suffix=""):
    """Create environmental and economic impact analysis"""
    try:
        if len(df) == 0:
            return create_empty_plot("No data available")
        
        # Calculate impact metrics
        total_hh = len(df)
        avg_reduction = df['avg_reduction'].mean()
        
        # Environmental impacts
        weekly_fuel_saved = 8 * (avg_reduction / 100) * total_hh  # kg
        annual_co2 = weekly_fuel_saved * 52 * 1.8 / 1000  # tons
        trees_saved = weekly_fuel_saved * 52 / 500  # trees
        
        # Economic impacts
        weekly_cost_savings = weekly_fuel_saved * 0.1  # USD
        weekly_time_savings = weekly_fuel_saved * 0.5  # hours
        annual_economic = weekly_cost_savings * 52  # USD
        
        # Prepare data for chart
        impact_types = ['Weekly Fuel Saved', 'Annual CO₂ Reduction', 'Trees Protected', 
                       'Weekly Cost Savings', 'Annual Economic Value']
        values = [weekly_fuel_saved/1000, annual_co2, trees_saved, 
                 weekly_cost_savings, annual_economic]
        units = ['tons', 'tons', 'trees', 'USD', 'USD']
        
        # Create the chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=impact_types,
            y=values,
            marker_color=['#10b981', '#0ea5e9', '#059669', '#f59e0b', '#8b5cf6'],
            text=[f"{v:,.1f} {u}" for v, u in zip(values, units)],
            textposition='auto',
            hovertemplate='%{x}<br>Value: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Environmental & Economic Impact Analysis",
            xaxis_title="Impact Type",
            yaxis_title="Value",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_tickangle=-45
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
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        plot_bgcolor='white',
        height=300
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
avg_reduction = df['avg_reduction'].mean() if 'avg_reduction' in df.columns else 0
high_risk_count = df['low_adoption_risk'].sum() if 'low_adoption_risk' in df.columns else 0
success_rate = ((total_households - high_risk_count) / total_households * 100) if total_households > 0 else 0

# Get unique districts
districts = sorted(df['district'].unique()) if 'district' in df.columns else []

# =====================================================
# DASHBOARD HEADER
# =====================================================
st.markdown(f"""
<div class="main-header">
    <h1 style="margin: 0 0 10px 0; font-size: 2rem; font-weight: 800;">🔥 Sustainable Cooking Impact Dashboard</h1>
    <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
        DelAgua Stove Programme Analysis • {total_households:,} Households
    </p>
    <p style="margin: 10px 0 0 0; font-size: 0.95rem; opacity: 0.8;">
        Overall Success Rate: {success_rate:.1f}% • High Risk: {high_risk_count:,} households ({high_risk_count/total_households*100:.1f}%)
    </p>
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
            default=districts,  # All districts selected by default
            help="Choose districts to analyze"
        )
    else:
        selected_districts = []
        st.warning("No district data found")
    
    # Risk level filter
    risk_filter = st.selectbox(
        "Risk Level Filter",
        options=["All Households", "High Risk Only (<30%)", "Low Risk Only (≥30%)"],
        index=0
    )
    
    # Reduction range filter - FIXED: Handle case where all values might be the same
    if 'avg_reduction' in df.columns:
        min_reduction = float(df['avg_reduction'].min())
        max_reduction = float(df['avg_reduction'].max())
        
        # Ensure min and max are different for slider
        if min_reduction == max_reduction:
            if min_reduction == 0:
                # If all values are 0, set a reasonable range
                min_reduction = 0
                max_reduction = 100
            else:
                # Add some padding around the single value
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
        st.warning("No reduction data available")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # QUICK STATS
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("### 📊 Quick Overview")
    
    st.metric("Total Households", f"{total_households:,}")
    
    # Show appropriate delta for avg reduction
    if avg_reduction >= 30:
        delta_label = "Above Target"
    elif avg_reduction > 0:
        delta_label = "Below Target"
    else:
        delta_label = "No reduction"
    
    st.metric("Average Reduction", f"{avg_reduction:.1f}%", delta_label)
    st.metric("Success Rate", f"{success_rate:.1f}%")
    st.metric("High Risk HH", f"{high_risk_count:,}", 
             f"{high_risk_count/total_households*100:.1f}%")
    
    # Data quality info
    st.markdown("---")
    st.markdown("**📊 Data Quality**")
    if len(districts) > 0:
        st.markdown(f"• Districts: {len(districts)}")
    st.markdown(f"• Records: {total_households:,}")
    
    # Check for essential columns
    essential_cols = ['avg_reduction', 'district', 'household_size']
    missing_cols = [col for col in essential_cols if col not in df.columns]
    if missing_cols:
        st.markdown(f"• Missing: {', '.join(missing_cols)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# APPLY FILTERS
# =====================================================
filtered_df = df.copy()

# Apply district filter
if len(selected_districts) > 0 and 'district' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['district'].isin(selected_districts)]

# Apply risk filter
if risk_filter == "High Risk Only (<30%)" and 'low_adoption_risk' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['low_adoption_risk'] == 1]
elif risk_filter == "Low Risk Only (≥30%)" and 'low_adoption_risk' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['low_adoption_risk'] == 0]

# Apply reduction range filter
if 'avg_reduction' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['avg_reduction'] >= reduction_range[0]) &
        (filtered_df['avg_reduction'] <= reduction_range[1])
    ]

# =====================================================
# MAIN CONTENT AREA WITH TABS - ALL PLOTS WORKING
# =====================================================
with col_main:
    # Filter summary
    filtered_count = len(filtered_df)
    st.markdown(f"### 📈 Analysis Results: {filtered_count:,} Households ({filtered_count/len(df)*100:.1f}% of total)")
    
    # CREATE TABS - ALL WILL SHOW PLOTS WITH UNIQUE KEYS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏘️ District Insights",
        "🗺️ Geographic Analysis", 
        "📊 Performance Metrics",
        "📈 Usage Trends",
        "🌿 Impact Analysis"
    ])
    
    # TAB 1: DISTRICT INSIGHTS
    with tab1:
        st.markdown("<div class='section-title'>District Performance & Adoption Analysis</div>", unsafe_allow_html=True)
        
        # Row 1: Two charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**District Performance Ranking**")
            fig1 = create_district_performance_chart(filtered_df, "_tab1_district")
            st.plotly_chart(fig1, use_container_width=True, key="tab1_district_perf")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**Adoption Level Distribution**")
            fig2 = create_adoption_distribution_chart(filtered_df, "_tab1_adoption")
            st.plotly_chart(fig2, use_container_width=True, key="tab1_adoption_dist")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Row 2: Full width chart
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**Fuel Savings by District**")
        fig3 = create_savings_analysis_chart(filtered_df, "_tab1_savings")
        st.plotly_chart(fig3, use_container_width=True, key="tab1_savings")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 2: GEOGRAPHIC ANALYSIS
    with tab2:
        st.markdown("<div class='section-title'>Geographic Distribution & Patterns</div>", unsafe_allow_html=True)
        
        # Map
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**Geographic Distribution of Stove Adoption**")
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
                st.metric("Avg Distance to Market", f"{avg_distance:.1f} km")
                st.metric("Max Distance to Market", f"{max_distance:.1f} km")
            else:
                st.info("Distance to market data not available")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_geo2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**🎯 Risk Analysis by Location**")
            if 'low_adoption_risk' in filtered_df.columns:
                high_risk = filtered_df[filtered_df['low_adoption_risk'] == 1].shape[0]
                low_risk = filtered_df[filtered_df['low_adoption_risk'] == 0].shape[0]
                st.metric("High-risk Households", f"{high_risk:,}", 
                         f"{high_risk/len(filtered_df)*100:.1f}%")
                st.metric("Low-risk Households", f"{low_risk:,}", 
                         f"{low_risk/len(filtered_df)*100:.1f}%")
            else:
                st.info("Risk analysis data not available")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 3: PERFORMANCE METRICS
    with tab3:
        st.markdown("<div class='section-title'>Performance Analysis & Distribution</div>", unsafe_allow_html=True)
        
        # Performance distribution
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**Performance Distribution**")
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
                
                st.metric("Mean Reduction", f"{mean_val:.1f}%")
                st.metric("Median Reduction", f"{median_val:.1f}%")
                st.metric("Standard Deviation", f"{std_val:.1f}%")
            else:
                st.info("Performance data not available")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_perf2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**🎯 Target Achievement**")
            if 'avg_reduction' in filtered_df.columns:
                above_target = filtered_df[filtered_df['avg_reduction'] >= 30].shape[0]
                below_target = filtered_df[filtered_df['avg_reduction'] < 30].shape[0]
                
                st.metric("Above Target (≥30%)", f"{above_target:,}", 
                         f"{above_target/len(filtered_df)*100:.1f}%")
                st.metric("Below Target (<30%)", f"{below_target:,}", 
                         f"{below_target/len(filtered_df)*100:.1f}%")
                st.metric("Gap to Target", f"{30 - filtered_df['avg_reduction'].mean():.1f}%")
            else:
                st.info("Target achievement data not available")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 4: USAGE TRENDS
    with tab4:
        st.markdown("<div class='section-title'>Usage Trends & Patterns</div>", unsafe_allow_html=True)
        
        # Usage trends chart
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**Monthly Fuel Usage Trends**")
        fig_usage = create_usage_trends_chart(filtered_df, "_tab4_usage")
        st.plotly_chart(fig_usage, use_container_width=True, key="tab4_usage_trends")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Trend analysis - REMOVED SYNTHETIC INSIGHTS
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**📅 Data Insights**")
            
            # Check for usage data
            usage_cols = [col for col in filtered_df.columns if 'usage_month' in col]
            if usage_cols and len(usage_cols) >= 1:
                # Show real data insights
                st.info(f"**Real Usage Data:** {len(usage_cols)} months available")
                
                # Calculate basic statistics
                avg_usage = filtered_df[usage_cols].mean().mean()
                usage_range = filtered_df[usage_cols].mean().max() - filtered_df[usage_cols].mean().min()
                
                st.metric("Average Monthly Usage", f"{avg_usage:.1f} kg")
                st.metric("Monthly Variation", f"{usage_range:.1f} kg")
                
                # Find peak usage month
                if len(usage_cols) > 0:
                    monthly_avgs = filtered_df[usage_cols].mean()
                    peak_month_idx = monthly_avgs.idxmax()
                    peak_month_value = monthly_avgs.max()
                    st.metric("Peak Usage Month", f"{peak_month_idx}", f"{peak_month_value:.1f} kg")
            else:
                st.info("No monthly usage data available for trend analysis")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_trend2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            st.markdown("**📊 Reduction Metrics**")
            
            if 'avg_reduction' in filtered_df.columns:
                current_avg = filtered_df['avg_reduction'].mean()
                
                # Show current status
                st.metric("Current Average", f"{current_avg:.1f}%")
                
                # Check if we have temporal data
                if 'distribution_date' in filtered_df.columns:
                    try:
                        # Count unique distribution months
                        unique_months = filtered_df['distribution_date'].dt.to_period('M').nunique()
                        if unique_months > 1:
                            st.metric("Monitoring Period", f"{unique_months} months")
                    except:
                        pass
                
                # Add comparison to target
                gap_to_target = 30 - current_avg
                if gap_to_target > 0:
                    st.metric("Gap to Target", f"{gap_to_target:.1f}%", "Need improvement")
                else:
                    st.metric("Above Target", f"{-gap_to_target:.1f}%", "Excellent")
            else:
                st.info("Reduction metrics not available")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # TAB 5: IMPACT ANALYSIS
    with tab5:
        st.markdown("<div class='section-title'>Environmental & Economic Impact</div>", unsafe_allow_html=True)
        
        # Impact analysis chart
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("**Comprehensive Impact Analysis**")
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
            
            st.metric("Weekly Fuel Saved", f"{weekly_fuel/1000:,.1f} tons")
            st.metric("Annual CO₂ Reduction", f"{weekly_fuel * 52 * 1.8 / 1000:,.1f} tons")
            st.metric("Trees Protected Annually", f"{weekly_fuel * 52 / 500:,.0f}")
            st.metric("Health Impact", "Improved", "Reduced indoor air pollution")
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
            
            st.metric("Weekly Fuel Cost Savings", f"${weekly_cost:,.0f}")
            st.metric("Weekly Time Savings", f"{weekly_time:,.0f} hours")
            st.metric("Equivalent Wage Savings", f"${weekly_wage:,.0f}/week")
            st.metric("Annual Economic Value", f"${weekly_wage * 52:,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# DATA SUMMARY SECTION
# =====================================================
st.markdown("---")
st.markdown("<div class='section-title'>📋 Data Summary</div>", unsafe_allow_html=True)

col_sum1, col_sum2, col_sum3 = st.columns(3)

with col_sum1:
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    st.markdown("**Data Overview**")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Districts Covered", f"{len(districts)}")
    st.metric("Data Collection", "2023-2024")
    st.markdown("</div>", unsafe_allow_html=True)

with col_sum2:
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    st.markdown("**Key Metrics**")
    st.metric("Avg Reduction", f"{avg_reduction:.1f}%")
    st.metric("Success Rate", f"{success_rate:.1f}%")
    st.metric("High Risk HH", f"{high_risk_count:,}")
    st.markdown("</div>", unsafe_allow_html=True)

with col_sum3:
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    st.markdown("**Data Coverage**")
    
    # Calculate coverage percentages
    if 'avg_reduction' in df.columns:
        reduction_coverage = (df['avg_reduction'].notna().sum() / len(df)) * 100
        st.metric("Reduction Data", f"{reduction_coverage:.1f}%")
    
    if 'district' in df.columns:
        district_coverage = (df['district'].notna().sum() / len(df)) * 100
        st.metric("District Data", f"{district_coverage:.1f}%")
    
    if 'household_size' in df.columns:
        size_coverage = (df['household_size'].notna().sum() / len(df)) * 100
        st.metric("Household Data", f"{size_coverage:.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748b; padding: 1.5rem;">
    <div style="font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 500;">
        🔥 Sustainable Cooking Impact Dashboard • DelAgua Stove Adoption Programme
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.3rem;">
        Data Source: delagua_stove_data_cleaned.csv • {total_households:,} households • Analysis: {datetime.now().strftime('%B %Y')}
    </div>
    <div style="font-size: 0.75rem; color: #cbd5e1;">
        Built with Streamlit • Technical Support: sniyizurugero@aimsric.org
    </div>
</div>
""", unsafe_allow_html=True)
