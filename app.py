# app.py - Main dashboard application
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import from utils
from utils.data_loader import load_and_process_data

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Sustainable Cooking Impact Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS STYLING
# =====================================================
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# =====================================================
# CHART FUNCTIONS
# =====================================================

def create_district_performance_chart(df):
    """Create district performance comparison chart"""
    try:
        if len(df) == 0 or 'district' not in df.columns:
            return create_empty_plot("No district data available")
        
        district_stats = df.groupby('district').agg({
            'avg_reduction': 'mean',
            'household_id': 'count'
        }).reset_index()
        
        district_stats = district_stats.sort_values('avg_reduction', ascending=True)
        
        colors = []
        for reduction in district_stats['avg_reduction']:
            if reduction < 30:
                colors.append('#ef4444')
            elif reduction < 50:
                colors.append('#f59e0b')
            elif reduction < 70:
                colors.append('#3b82f6')
            elif reduction < 85:
                colors.append('#8b5cf6')
            else:
                colors.append('#10b981')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=district_stats['district'],
            x=district_stats['avg_reduction'],
            orientation='h',
            marker_color=colors,
            text=[f"{x:.1f}%" for x in district_stats['avg_reduction']],
            textposition='auto',
            textfont=dict(color='white', size=12, weight='bold'),
            hovertemplate='<b>%{y}</b><br>Avg Reduction: %{x:.1f}%<br>Households: %{customdata}<extra></extra>',
            customdata=district_stats['household_id']
        ))
        
        fig.add_vline(x=30, line_dash="dash", line_color="#ef4444", opacity=0.8,
                     annotation_text="Target", annotation_position="top right")
        
        fig.update_layout(
            title=dict(
                text="🏆 District Performance Ranking",
                font=dict(size=22, color='#0c4a6e'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Average Fuel Reduction (%)",
            yaxis_title="District",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        return fig
        
    except Exception as e:
        return create_empty_plot("Error creating chart")

# Add other chart functions here (similar to previous code)
# ... [Include all other chart functions from previous code]

# =====================================================
# MAIN DASHBOARD
# =====================================================

def main():
    # Load data
    df = load_and_process_data()
    
    if df.empty:
        st.error("❌ Could not load data file. Please check if 'delagua_stove_data_cleaned.csv' exists.")
        return
    
    # Calculate metrics
    total_households = len(df)
    avg_reduction_val = df['avg_reduction'].mean() if 'avg_reduction' in df.columns else 0
    districts = sorted(df['district'].unique()) if 'district' in df.columns else []
    
    # Dashboard header
    st.markdown(f"""
    <div class="main-header">
        <h1>🔥 Sustainable Cooking Impact Dashboard</h1>
        <p style="margin: 0; font-size: 1.3rem; opacity: 0.95; font-weight: 500; letter-spacing: 0.5px;">
            DelAgua Stove Programme Analysis • {total_households:,} Households • {len(districts)} Districts
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    st.markdown("<div class='section-title'>📊 Key Performance Indicators</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-color: #3b82f6;">
            <div style="font-size: 0.95rem; color: #475569; margin-bottom: 8px; font-weight: 600;">AVERAGE FUEL REDUCTION</div>
            <div style="font-size: 2.5rem; font-weight: 900; color: #1e40af; line-height: 1;">
                {avg_reduction_val:.1f}<span style="font-size: 1.5rem;">%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ... Add other KPI cards
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏘️ District Insights(click)",
        "🗺️ Geographic Analysis(click)", 
        "📊 Performance Metrics(click)",
        "📈 Usage Trends(click)",
        "🌿 Impact Analysis(click)"
    ])
    
    with tab1:
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        fig = create_district_performance_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
