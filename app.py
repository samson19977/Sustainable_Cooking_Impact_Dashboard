# app.py - Sustainable Cooking Impact Dashboard - STREAMLIT VERSION
# Enhanced with insights from DelAgua Strategic Analysis Report
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from utils.data_loader import load_and_clean_data, create_sample_data
from utils.visualizations import (
    create_district_performance_chart,
    create_geographic_map,
    create_performance_distribution,
    create_elevation_analysis,
    create_savings_analysis,
    create_temporal_trends,
    create_cluster_analysis,
    create_risk_correlation_matrix,
    create_priority_intervention_map
)

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
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data(ttl=3600, show_spinner="Loading and cleaning dataset...")
def get_data():
    """Load and cache data with enhanced error handling."""
    try:
        df = load_and_clean_data()
        
        # Calculate predictive risk score based on report insights
        df['risk_score'] = (
            df['distance_to_market_km'].fillna(0) * 0.4 +
            (df['household_size'] - df['household_size'].mean()) * 0.3 +
            np.where(df['district'].isin(['Rulindo', 'Musanze']), 0.3, 0)
        )
        
        # Normalize risk score
        df['risk_score'] = (df['risk_score'] - df['risk_score'].min()) / \
                          (df['risk_score'].max() - df['risk_score'].min())
        
        # Add intervention priority based on report findings
        df['intervention_priority'] = np.where(
            (df['low_adoption_risk'] == 1) & (df['risk_score'] > 0.7),
            'High Priority',
            np.where(df['low_adoption_risk'] == 1, 'Medium Priority', 'Low Priority')
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()

# Load data
df = get_data()
districts_clean = sorted(df['district'].unique())

# Calculate summary statistics
total_households = len(df)
avg_reduction = df['avg_reduction'].mean()
high_risk_count = df['low_adoption_risk'].sum()
success_rate = ((total_households - high_risk_count) / total_households * 100)
total_savings = df['weekly_fuel_saving_kg'].sum()
annual_savings_tons = (total_savings * 52) / 1000

# =====================================================
# DASHBOARD LAYOUT
# =====================================================

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

# Main layout with left panel and main content
col_left, col_main = st.columns([1, 3])

with col_left:  # Left panel with filters and data overview
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
    
    intervention_priority = st.selectbox(
        "Intervention Priority",
        options=["All", "High Priority", "Medium Priority", "Low Priority"],
        index=0
    )
    
    reduction_range = st.slider(
        "Fuel Reduction Range (%)",
        min_value=-100.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=5.0
    )
    
    distance_range = st.slider(
        "Distance to Market (km)",
        min_value=0.0,
        max_value=50.0,
        value=(0.0, 25.0),
        step=0.5
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Overview
    st.markdown('<div class="data-overview">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Overview")
    
    st.metric("Total Households", f"{total_households:,}")
    st.metric("Average Reduction", f"{avg_reduction:.1f}%")
    st.metric("High-Risk Households", f"{high_risk_count:,}", 
              delta=f"{high_risk_count/total_households*100:.1f}%")
    
    st.markdown("---")
    st.markdown("**📍 Districts:**")
    st.write(", ".join(districts_clean))
    
    st.markdown("**📈 Reduction Range:**")
    st.write(f"{df['avg_reduction'].min():.1f}% to {df['avg_reduction'].max():.1f}%")
    
    st.markdown("**🎯 Key Insight from Report:**")
    st.info("""
    **47.8% of households (3,809) are below the 30% adoption target.** 
    Top predictors: Distance to market, district location, household size.
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
    
    if intervention_priority != "All":
        filtered_df = filtered_df[filtered_df['intervention_priority'] == intervention_priority]
    
    reduction_min, reduction_max = reduction_range
    distance_min, distance_max = distance_range
    
    filtered_df = filtered_df[
        (filtered_df['avg_reduction'] >= reduction_min) &
        (filtered_df['avg_reduction'] <= reduction_max) &
        (filtered_df['distance_to_market_km'] >= distance_min) &
        (filtered_df['distance_to_market_km'] <= distance_max)
    ]

with col_main:  # Main content area
    # Executive Summary Cards
    st.markdown("<div class='section-title'>📊 Executive Summary</div>", unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.error("No households match your filter criteria. Please adjust filters.")
    else:
        # Calculate filtered metrics
        filtered_total = len(filtered_df)
        filtered_avg_reduction = filtered_df['avg_reduction'].mean()
        filtered_high_risk = filtered_df['low_adoption_risk'].sum()
        filtered_success_rate = ((filtered_total - filtered_high_risk) / filtered_total * 100)
        filtered_savings = filtered_df['weekly_fuel_saving_kg'].sum()
        filtered_annual_savings_tons = (filtered_savings * 52) / 1000
        high_priority_count = filtered_df[filtered_df['intervention_priority'] == 'High Priority'].shape[0]
        
        # Create amazing cards
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
            st.markdown(f"""
            <div class="metric-card-orange">
                <div class="metric-label">🎯 Priority Interventions</div>
                <div class="metric-number">{high_priority_count:,}</div>
                <div class="metric-label">High-priority households</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Priority alert based on report findings
        if filtered_high_risk > 0:
            high_risk_pct = (filtered_high_risk / filtered_total * 100)
            
            # Calculate expected impact based on report
            additional_savings = filtered_high_risk * 0.171  # 171kg per household per year
            co2_reduction = additional_savings * 1.8 / 1000  # Convert to tons CO₂
            
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        border-radius: 10px; border-left: 4px solid #ef4444;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="font-size: 1.5rem;">⚠️</div>
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: #7f1d1d;">Priority Intervention Needed</div>
                        <div style="color: #991b1b; font-size: 0.9rem; margin-top: 0.3rem;">
                            {filtered_high_risk} households ({high_risk_pct:.1f}%) are below the 30% fuel reduction target
                        </div>
                        <div style="color: #92400e; font-size: 0.85rem; margin-top: 0.5rem;">
                            🎯 Targeted intervention could save: 
                            <strong>{additional_savings:.0f} tons of fuel</strong> and 
                            <strong>{co2_reduction:.0f} tons of CO₂</strong> annually
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Main Analysis Section - Enhanced with report insights
    st.markdown("<div class='section-title'>📈 Performance Analysis</div>", unsafe_allow_html=True)
    
    # Enhanced tabbed analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏘️ District Comparison", 
        "🗺️ Geographic Insights", 
        "📊 Predictive Analytics",
        "📈 Temporal Trends",
        "🌿 Environmental Impact"
    ])
    
    with tab1:
        col1b, col2b = st.columns(2)
        
        with col1b:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_district_performance_chart(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2b:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_performance_distribution(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # District-level insights from report
        if len(filtered_df) > 0:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 🎯 District Priority Ranking (Based on Report)")
            
            district_priorities = filtered_df[filtered_df['low_adoption_risk'] == 1]\
                .groupby('district').size().reset_index(name='high_risk_count')
            district_priorities = district_priorities.sort_values('high_risk_count', ascending=False)
            
            if len(district_priorities) > 0:
                st.markdown("**Top districts needing intervention:**")
                for idx, row in district_priorities.head(3).iterrows():
                    percentage = (row['high_risk_count'] / filtered_total * 100)
                    st.markdown(f"- **{row['district']}**: {row['high_risk_count']:,} high-risk households ({percentage:.1f}%)")
                
                # Resource allocation suggestion
                if len(district_priorities) >= 3:
                    st.info("""
                    **Report Recommendation:** Allocate 65% of field resources to top 3 districts: 
                    {}, {}, and {} for maximum impact.
                    """.format(
                        district_priorities.iloc[0]['district'],
                        district_priorities.iloc[1]['district'],
                        district_priorities.iloc[2]['district']
                    ))
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        col1c, col2c = st.columns(2)
        
        with col1c:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_geographic_map(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2c:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_priority_intervention_map(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Cluster analysis insights
        if len(filtered_df) > 100:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 📍 Geographic Cluster Analysis")
            st.plotly_chart(create_cluster_analysis(filtered_df), use_container_width=True)
            
            # Report insight
            st.info("""
            **Report Finding:** DBSCAN analysis identified 2 major clusters containing 
            3,791 low-adoption households (only 0.5% noise). Cluster 0 has 3,763 households 
            with 9.9% average reduction - **primary intervention zone**.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        col1d, col2d = st.columns(2)
        
        with col1d:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_risk_correlation_matrix(filtered_df), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2d:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("### 🎯 Predictive Risk Factors")
            
            if len(filtered_df) > 10:
                # Top risk factors from report
                st.markdown("""
                **Top 3 Predictors of Low Adoption (Logistic Regression Model):**
                1. **Distance to Market** (Strongest factor: farther = higher risk)
                2. **District Location** (Rulindo, Musanze show elevated risk)
                3. **Household Size** (Larger families struggle more)
                """)
                
                # Model performance metrics from report
                st.markdown("---")
                st.markdown("**Model Performance Metrics:**")
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric("Accuracy", "61.8%")
                
                with col_metric2:
                    st.metric("ROC-AUC", "66.2%")
                
                with col_metric3:
                    st.metric("Recall", "61.9%")
                
                st.caption("*Based on logistic regression model using pre-distribution data only*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Elevation analysis
        if len(filtered_df) > 10:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(create_elevation_analysis(filtered_df), use_container_width=True)
            
            # Correlation insight
            if len(filtered_df) > 2:
                elevation_corr = filtered_df['elevation_m'].corr(filtered_df['avg_reduction'])
                distance_corr = filtered_df['distance_to_market_km'].corr(filtered_df['avg_reduction'])
                
                st.markdown(f"""
                **Correlation Insights:**
                - **Elevation**: r = {elevation_corr:.3f} (weak but significant negative correlation)
                - **Market Distance**: r = {distance_corr:.3f} (moderate negative correlation)
                
                *Higher elevation and greater distance correlate with lower adoption rates*
                """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(create_temporal_trends(filtered_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        if len(filtered_df) > 0:
            col1e, col2e = st.columns(2)
            
            with col1e:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(create_savings_analysis(filtered_df), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2e:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("### 🌍 Environmental Impact Summary")
                
                # Calculate metrics
                weekly_savings_kg = filtered_df['weekly_fuel_saving_kg'].sum()
                annual_savings_kg = weekly_savings_kg * 52
                
                # Calculate potential additional savings from report
                potential_additional = filtered_high_risk * 171  # 171kg per high-risk household
                total_potential = annual_savings_kg + potential_additional
                
                # Use HTML grid for metrics
                st.markdown(f"""
                <div class="metrics-grid">
                    <div style="text-align: center; padding: 1rem; background: #f0f9ff; border-radius: 8px; border: 1px solid #bae6fd;">
                        <div style="font-size: 0.9rem; color: #0369a1; margin-bottom: 0.5rem;">Current Annual Savings</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #0c4a6e;">{annual_savings_kg/1000:,.0f} tons</div>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: #f0f9ff; border-radius: 8px; border: 1px solid #bae6fd;">
                        <div style="font-size: 0.9rem; color: #0369a1; margin-bottom: 0.5rem;">Potential Additional</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #0c4a6e;">{potential_additional/1000:,.0f} tons</div>
                        <div style="font-size: 0.7rem; color: #64748b;">with targeted interventions</div>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: #f0f9ff; border-radius: 8px; border: 1px solid #bae6fd;">
                        <div style="font-size: 0.9rem; color: #0369a1; margin-bottom: 0.5rem;">CO₂ Reduction</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #0c4a6e;">{(total_potential * 1.8)/1000:,.0f} tons</div>
                        <div style="font-size: 0.7rem; color: #64748b;">1.8kg CO₂ per kg fuelwood</div>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: #f0f9ff; border-radius: 8px; border: 1px solid #bae6fd;">
                        <div style="font-size: 0.9rem; color: #0369a1; margin-bottom: 0.5rem;">Trees Saved</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #0c4a6e;">{(total_potential/500):,.0f}</div>
                        <div style="font-size: 0.7rem; color: #64748b;">≈500kg fuelwood per tree</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Report impact statement
                st.markdown("---")
                st.success(f"""
                **Report Impact Projection:** Targeted interventions for {filtered_high_risk:,} high-risk households could:
                - Save an additional **{potential_additional/1000:,.0f} tons** of fuel annually
                - Reduce **{(potential_additional * 1.8)/1000:,.0f} tons** of CO₂ emissions
                - Improve adoption by **+15 percentage points** for struggling households
                - Increase resource efficiency by **37%** through geographic clustering
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.info("No data available with current filters")
            st.markdown('</div>', unsafe_allow_html=True)

# Actionable Recommendations Section
st.markdown("<div class='section-title'>🎯 Actionable Recommendations</div>", unsafe_allow_html=True)

# Tiered recommendations from report
rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    st.markdown("""
    <div style="padding: 1.5rem; background: white; border-radius: 12px; border: 1px solid #e2e8f0; height: 100%;">
        <h4 style="color: #0c4a6e; margin-top: 0;">🚨 IMMEDIATE (0-30 Days)</h4>
        <ul style="color: #475569; font-size: 0.95rem; padding-left: 1.2rem;">
            <li>Deploy teams to 2,393 highest-risk households</li>
            <li>Focus on households >5km from markets</li>
            <li>Allocate 65% resources to Gakenke, Nyabihu, Burera</li>
            <li>Implement distance-based prioritization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col2:
    st.markdown("""
    <div style="padding: 1.5rem; background: white; border-radius: 12px; border: 1px solid #e2e8f0; height: 100%;">
        <h4 style="color: #0c4a6e; margin-top: 0;">📈 OPTIMIZATION (30-90 Days)</h4>
        <ul style="color: #475569; font-size: 0.95rem; padding-left: 1.2rem;">
            <li>Develop "Market Access Kits" for remote households</li>
            <li>Create distance-tiered training programs</li>
            <li>Pilot mobile training units for largest clusters</li>
            <li>Establish feedback loops from field teams</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col3:
    st.markdown("""
    <div style="padding: 1.5rem; background: white; border-radius: 12px; border: 1px solid #e2e8f0; height: 100%;">
        <h4 style="color: #0c4a6e; margin-top: 0;">🔄 SYSTEMIC (Ongoing)</h4>
        <ul style="color: #475569; font-size: 0.95rem; padding-left: 1.2rem;">
            <li>Integrate predictive scoring into distribution</li>
            <li>Monthly monitoring of 276 high-risk grid cells</li>
            <li>Quarterly model retraining with new data</li>
            <li>Develop performance dashboards for field teams</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer with deployment info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
        🔥 Sustainable Cooking Impact Dashboard • DelAgua Stove Adoption Programme
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;">
        Data updated: March 2024 • 7,976 households analyzed • 3,809 high-risk households identified
    </div>
    <div style="font-size: 0.75rem; color: #cbd5e1;">
        Interactive version of the DelAgua Strategic Analysis Report • Deploy via GitHub + Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
