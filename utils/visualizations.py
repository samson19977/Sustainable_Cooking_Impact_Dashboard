# utils/visualizations.py - Enhanced visualization functions
# Including insights from DelAgua Strategic Analysis Report
# =====================================================

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def create_district_performance_chart(filtered_df):
    """Create district performance comparison chart with report insights."""
    if len(filtered_df) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Need more data for analysis", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    district_stats = filtered_df.groupby('district').agg({
        'avg_reduction': 'mean',
        'household_id': 'count',
        'low_adoption_risk': 'mean'
    }).reset_index()
    
    # Sort by performance
    district_stats = district_stats.sort_values('avg_reduction', ascending=True)
    
    # Color based on performance
    colors = []
    for reduction in district_stats['avg_reduction']:
        if reduction >= 30:
            colors.append('#10b981')  # Green for success
        elif reduction >= 20:
            colors.append('#f59e0b')  # Yellow for moderate
        else:
            colors.append('#ef4444')  # Red for low
    
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
                     'Risk Rate: %{customdata[1]:.1%}<extra></extra>',
        customdata=np.column_stack([
            district_stats['household_id'],
            district_stats['low_adoption_risk']
        ])
    ))
    
    fig.add_vline(x=30, line_dash="dash", line_color="red", opacity=0.7)
    
    fig.update_layout(
        height=400,
        title='District Performance Comparison',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title='Average Fuel Reduction (%)',
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=20)
    )
    
    return fig

def create_geographic_map(filtered_df):
    """Create interactive geographic map."""
    if len(filtered_df) < 5 or 'latitude' not in filtered_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Geographic data not available", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500
        )
        return fig
    
    sample_df = filtered_df.sample(min(500, len(filtered_df)), random_state=42)
    
    fig = px.scatter_mapbox(
        sample_df,
        lat="latitude",
        lon="longitude",
        color="avg_reduction",
        size=np.where(sample_df['low_adoption_risk'] == 1, 15, 8),
        hover_name="district",
        hover_data={
            "avg_reduction": ":.1f%",
            "distance_to_market_km": ":.1f km",
            "elevation_m": ":.0f m",
            "intervention_priority": True
        },
        color_continuous_scale="RdYlGn",
        size_max=20,
        zoom=8.5,
        center=dict(lat=-1.4, lon=29.7),
        height=500
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 10, "t": 10, "l": 10, "b": 10},
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_performance_distribution(filtered_df):
    """Create performance distribution visualization."""
    if len(filtered_df) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Need more data for distribution", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Performance Distribution', 'Risk Analysis'),
        specs=[[{'type': 'histogram'}, {'type': 'pie'}]],
        horizontal_spacing=0.15
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=filtered_df['avg_reduction'],
            nbinsx=30,
            marker_color='#3b82f6',
            opacity=0.7,
            name='Distribution',
            hovertemplate='Reduction: %{x:.1f}%<br>Households: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_vline(x=30, line_dash="dash", line_color="red", opacity=0.7, row=1, col=1)
    
    # Pie chart for risk distribution
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
            hovertemplate='%{label}<br>Households: %{value}<extra></extra>'
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

def create_elevation_analysis(filtered_df):
    """Create elevation impact analysis."""
    if len(filtered_df) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Need more data for analysis", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=filtered_df['elevation_m'],
        y=filtered_df['avg_reduction'],
        mode='markers',
        marker=dict(
            size=8,
            color=filtered_df['avg_reduction'],
            colorscale='RdYlGn',
            showscale=True,
            opacity=0.6
        ),
        hovertemplate='Elevation: %{x:.0f}m<br>Reduction: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        height=400,
        title='Elevation Impact on Fuel Reduction',
        xaxis_title='Elevation (meters)',
        yaxis_title='Fuel Reduction (%)',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.7, annotation_text="Target 30%")
    
    return fig

def create_savings_analysis(filtered_df):
    """Create fuel savings analysis."""
    if len(filtered_df) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Need more data for analysis", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    # Calculate savings by performance category
    savings_by_category = filtered_df.groupby('performance_category')['weekly_fuel_saving_kg'].sum().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=savings_by_category['performance_category'],
        y=savings_by_category['weekly_fuel_saving_kg'] / 1000,  # Convert to tons
        marker_color=['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6'],
        hovertemplate='Category: %{x}<br>Weekly Savings: %{y:,.0f} tons<extra></extra>'
    ))
    
    fig.update_layout(
        height=400,
        title='Weekly Fuel Savings by Performance Category',
        xaxis_title='Performance Category',
        yaxis_title='Weekly Savings (tons)',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_temporal_trends(filtered_df):
    """Create temporal trends analysis based on distribution dates."""
    if len(filtered_df) < 10 or 'distribution_year' not in filtered_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Temporal data not available", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    # Group by month and year
    if 'distribution_month' in filtered_df.columns:
        temporal_data = filtered_df.groupby(['distribution_year', 'distribution_month']).agg({
            'avg_reduction': 'mean',
            'household_id': 'count'
        }).reset_index()
        
        # Create date string for x-axis
        temporal_data['date'] = pd.to_datetime(
            temporal_data['distribution_year'].astype(str) + '-' + 
            temporal_data['distribution_month'].astype(str) + '-01'
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=temporal_data['date'],
            y=temporal_data['avg_reduction'],
            mode='lines+markers',
            name='Avg Reduction',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8),
            hovertemplate='Date: %{x|%b %Y}<br>Reduction: %{y:.1f}%<br>Households: %{customdata}<extra></extra>',
            customdata=temporal_data['household_id']
        ))
        
        fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.7)
        
        fig.update_layout(
            height=400,
            title='Monthly Adoption Trends',
            xaxis_title='Month',
            yaxis_title='Average Fuel Reduction (%)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    else:
        # Fallback to yearly trends
        yearly_data = filtered_df.groupby('distribution_year').agg({
            'avg_reduction': 'mean',
            'household_id': 'count'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=yearly_data['distribution_year'],
            y=yearly_data['avg_reduction'],
            marker_color=['#3b82f6', '#10b981'],
            hovertemplate='Year: %{x}<br>Reduction: %{y:.1f}%<br>Households: %{customdata}<extra></extra>',
            customdata=yearly_data['household_id']
        ))
        
        fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.7)
        
        fig.update_layout(
            height=400,
            title='Yearly Adoption Trends',
            xaxis_title='Year',
            yaxis_title='Average Fuel Reduction (%)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    
    return fig

def create_cluster_analysis(filtered_df):
    """Create cluster analysis visualization based on report DBSCAN findings."""
    if len(filtered_df) < 100 or 'latitude' not in filtered_df.columns or 'longitude' not in filtered_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Need more data for cluster analysis", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    try:
        # Sample data for performance
        sample_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
        
        # Perform DBSCAN clustering (as in report)
        coords = sample_df[['latitude', 'longitude']].values
        
        # Scale coordinates
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=10)
        clusters = dbscan.fit_predict(coords_scaled)
        
        sample_df['cluster'] = clusters
        
        # Create scatter plot
        fig = px.scatter(
            sample_df,
            x='longitude',
            y='latitude',
            color='cluster',
            size='avg_reduction',
            hover_data=['district', 'avg_reduction', 'distance_to_market_km'],
            title='Geographic Clusters of Low Adoption (DBSCAN Analysis)',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Cluster analysis failed: {str(e)}", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig

def create_risk_correlation_matrix(filtered_df):
    """Create correlation matrix visualization for risk factors."""
    if len(filtered_df) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Need more data for correlation analysis", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    # Select numeric columns for correlation
    numeric_cols = ['avg_reduction', 'distance_to_market_km', 'elevation_m', 
                    'household_size', 'baseline_fuel_kg_person_week']
    
    # Filter to available columns
    available_cols = [col for col in numeric_cols if col in filtered_df.columns]
    
    if len(available_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns for correlation", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        return fig
    
    # Calculate correlation matrix
    corr_matrix = filtered_df[available_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=available_cols,
        y=available_cols,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        height=400,
        title='Correlation Matrix of Risk Factors',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_priority_intervention_map(filtered_df):
    """Create priority intervention map based on report findings."""
    if len(filtered_df) < 5 or 'latitude' not in filtered_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Geographic data not available", showarrow=False)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500
        )
        return fig
    
    # Create priority categories
    sample_df = filtered_df.sample(min(500, len(filtered_df)), random_state=42).copy()
    
    fig = px.scatter_mapbox(
        sample_df,
        lat="latitude",
        lon="longitude",
        color="intervention_priority",
        size=np.where(sample_df['intervention_priority'] == 'High Priority', 20,
                     np.where(sample_df['intervention_priority'] == 'Medium Priority', 12, 8)),
        hover_name="district",
        hover_data={
            "avg_reduction": ":.1f%",
            "distance_to_market_km": ":.1f km",
            "risk_score": ":.2f"
        },
        color_discrete_map={
            'High Priority': '#ef4444',
            'Medium Priority': '#f59e0b',
            'Low Priority': '#10b981'
        },
        category_orders={"intervention_priority": ["High Priority", "Medium Priority", "Low Priority"]},
        size_max=25,
        zoom=8.5,
        center=dict(lat=-1.4, lon=29.7),
        height=500,
        title="Priority Intervention Map"
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 10, "t": 40, "l": 10, "b": 10},
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig
